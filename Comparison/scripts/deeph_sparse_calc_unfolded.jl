#!/usr/bin/env julia
"""Run DeepH sparse_calc.jl and stream layer-resolved LCAO unfolding weights."""

const DEEPH_SPARSE_CALC = get(
    ENV,
    "DEEPH_SPARSE_CALC",
    joinpath(dirname(@__DIR__), "..", "..", "DeepH-pack", "deeph", "inference", "sparse_calc.jl"),
)

source = read(DEEPH_SPARSE_CALC, String)
needle = "egval_sub = real(1 ./ egval_sub_inv) .+ (fermi_level)\n\n                        # orthogonalize"
replacement = "egval_sub = real(1 ./ egval_sub_inv) .+ (fermi_level)\n" *
              "                        record_unfolding!(parsed_args[\"output_dir\"], idx_k, egval_sub, egvec_sub)\n\n" *
              "                        # orthogonalize"
patched = replace(source, needle => replacement; count=1)
patched == source && error("DeepH band eigensolver changed; refusing an unsafe unfolding wrapper")
patched_without_main = replace(patched, r"\nmain\(\)\s*$" => "")
patched == patched_without_main && error("DeepH sparse_calc.jl no longer ends in main()")
include_string(Main, patched_without_main, DEEPH_SPARSE_CALC)

const UNFOLDING_MAP = Ref{Any}(nothing)
const UNFOLDING_KPOINTS = Ref{Any}(nothing)

function unfolding_inputs()
    if UNFOLDING_MAP[] === nothing
        UNFOLDING_MAP[] = readdlm(ENV["DEEPH_UNFOLD_MAP"], Int)
        UNFOLDING_KPOINTS[] = readdlm(ENV["DEEPH_UNFOLD_KPOINTS"], Float64)
    end
    return UNFOLDING_MAP[], UNFOLDING_KPOINTS[]
end

function record_unfolding!(output_dir, k_index, energies, vectors)
    mapping, primitive_kpoints = unfolding_inputs()
    primitive_k = primitive_kpoints[k_index, :]
    channels = maximum(mapping[:, 2])
    amplitudes = zeros(ComplexF64, channels, size(vectors, 2))
    for row in axes(mapping, 1)
        orbital, channel, r1, r2 = mapping[row, :]
        phase = exp(-2π * im * (primitive_k[1] * r1 + primitive_k[2] * r2))
        amplitudes[channel, :] .+= phase .* vectors[orbital, :]
    end
    cell_count = div(size(mapping, 1), channels)
    weights = vec(sum(abs2, amplitudes; dims=1)) ./ cell_count
    payload = Dict(
        "k_index" => k_index - 1,
        "primitive_k" => collect(primitive_k),
        "energies_eV" => collect(real(energies)),
        "spectral_weights" => collect(real(weights)),
        "normalization" => "layer_LCAO_coefficient_Fourier_weight_per_primitive_cell",
    )
    open(joinpath(output_dir, "unfolding_$(lpad(k_index - 1, 3, '0')).json"), "w") do handle
        JSON.print(handle, payload, 2)
    end
end

main()
