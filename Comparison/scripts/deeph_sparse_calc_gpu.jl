#!/usr/bin/env julia
"""Run DeepH sparse_calc.jl with its shift-invert solve backed by NVIDIA cuDSS."""

using CUDA
using CUDA.CUSPARSE
using CUDSS

const DEEPH_SPARSE_CALC = get(
    ENV,
    "DEEPH_SPARSE_CALC",
    joinpath(dirname(@__DIR__), "..", "..", "DeepH-pack", "deeph", "inference", "sparse_calc.jl"),
)

source = read(DEEPH_SPARSE_CALC, String)
if haskey(ENV, "DEEPH_UNFOLD_MAP")
    include(joinpath(@__DIR__, "deeph_unfolding_weights.jl"))
    source = patch_sparse_calc_for_unfolding(source)
end
if haskey(ENV, "DEEPH_MULLIKEN_GROUPS")
    include(joinpath(@__DIR__, "deeph_mulliken_weights.jl"))
    source = patch_sparse_calc_for_mulliken(source)
end
source_without_main = replace(source, r"\nmain\(\)\s*$" => "")
source == source_without_main && error("DeepH sparse_calc.jl no longer ends in main(); refusing an unsafe wrapper")
include_string(Main, source_without_main, DEEPH_SPARSE_CALC)

mutable struct CudssShiftInvertState
    solver
    rowptr_gpu
    colval_gpu
    nzval_gpu
    solution_gpu
    rhs_gpu
    solution_cpu
    rhs_cpu
    maximum_relative_residual
    validate_residual::Bool
    analysis_seconds::Float64
    factorization_seconds::Float64
    solve_seconds
    solve_count
    released::Bool
end

function release!(state::CudssShiftInvertState)
    state.released && return
    CUDA.synchronize()
    finalize(state.solver.matrix)
    finalize(state.solver.data)
    finalize(state.solver.config)
    CUDA.unsafe_free!(state.rowptr_gpu)
    CUDA.unsafe_free!(state.colval_gpu)
    CUDA.unsafe_free!(state.nzval_gpu)
    CUDA.unsafe_free!(state.solution_gpu)
    CUDA.unsafe_free!(state.rhs_gpu)
    state.validate_residual &&
        println("cuDSS maximum relative solve residual: ", state.maximum_relative_residual[])
    println(
        "cuDSS timings: analysis_seconds=", state.analysis_seconds,
        " factorization_seconds=", state.factorization_seconds,
        " solve_seconds=", state.solve_seconds[],
        " solve_count=", state.solve_count[],
    )
    state.released = true
    CUDA.reclaim()
end

function Pardiso.set_phase!(state::CudssShiftInvertState, phase)
    phase == Pardiso.RELEASE_ALL || error("Unsupported cuDSS compatibility phase: $phase")
    release!(state)
end

Pardiso.pardiso(::CudssShiftInvertState) = nothing

function construct_linear_map(H, S; out_of_core=false)
    CUDA.functional() || error("gpu_cudss requested but CUDA is not functional")
    matrix_cpu = SparseMatrixCSC{ComplexF64,Int64}(sparse(tril(parent(H))))
    matrix_transpose = sparse(transpose(matrix_cpu))
    rowptr_gpu = CuVector(Int64.(matrix_transpose.colptr))
    colval_gpu = CuVector(Int64.(matrix_transpose.rowval))
    nzval_gpu = CuVector(ComplexF64.(matrix_transpose.nzval))
    solution_gpu = CUDA.zeros(ComplexF64, size(H, 1))
    rhs_gpu = similar(solution_gpu)
    solution_cpu = zeros(ComplexF64, size(H, 1))
    rhs_cpu = similar(solution_cpu)
    maximum_relative_residual = Ref(0.0)
    validate_residual = get(ENV, "CUDSS_VALIDATE_RESIDUAL", "0") == "1"
    solver = CudssSolver(rowptr_gpu, colval_gpu, nzval_gpu, "H", 'L')
    cudss_set(solver, "host_nthreads", parse(Int, get(ENV, "CUDSS_HOST_THREADS", "1")))

    hybrid = get(ENV, "CUDSS_HYBRID_MEMORY", "1") == "1"
    if hybrid
        cudss_set(solver, "hybrid_memory_mode", 1)
    else
        cudss_set(solver, "deterministic_mode", 1)
    end
    started = time()
    cudss("analysis", solver, solution_gpu, rhs_gpu; asynchronous=false)
    analysis_seconds = time() - started
    if hybrid
        minimum = cudss_get(solver, "hybrid_device_memory_min")
        limit = parse(Int64, get(ENV, "CUDSS_DEVICE_MEMORY_LIMIT_BYTES", string(28 * 1024^3)))
        limit >= minimum || error("cuDSS requires at least $minimum GPU bytes; configured limit is $limit")
        cudss_set(solver, "hybrid_device_memory_limit", limit)
    end
    started = time()
    cudss("factorization", solver, solution_gpu, rhs_gpu; asynchronous=false)
    factorization_seconds = time() - started
    positive_inertia, negative_inertia = cudss_get(solver, "inertia")
    println("cuDSS inertia: positive=", positive_inertia, " negative=", negative_inertia)
    solve_seconds = Ref(0.0)
    solve_count = Ref(0)
    state = CudssShiftInvertState(
        solver,
        rowptr_gpu,
        colval_gpu,
        nzval_gpu,
        solution_gpu,
        rhs_gpu,
        solution_cpu,
        rhs_cpu,
        maximum_relative_residual,
        validate_residual,
        analysis_seconds,
        factorization_seconds,
        solve_seconds,
        solve_count,
        false,
    )
    linear_map = LinearMap{ComplexF64}(
        (y, x) -> begin
            mul!(rhs_cpu, S, x)
            copyto!(rhs_gpu, rhs_cpu)
            started = time()
            cudss("solve", solver, solution_gpu, rhs_gpu; asynchronous=false)
            solve_seconds[] += time() - started
            solve_count[] += 1
            copyto!(solution_cpu, solution_gpu)
            if validate_residual
                denominator = norm(rhs_cpu)
                residual = norm(H * solution_cpu - rhs_cpu) / (denominator == 0 ? 1 : denominator)
                maximum_relative_residual[] = max(maximum_relative_residual[], residual)
            end
            copyto!(y, solution_cpu)
        end,
        size(H, 1);
        ismutating=true,
    )
    println(
        "cuDSS factorization ready: matrix=ComplexF64/Int64, structure=Hermitian-indefinite, hybrid=",
        hybrid,
    )
    return linear_map, state
end

main()
