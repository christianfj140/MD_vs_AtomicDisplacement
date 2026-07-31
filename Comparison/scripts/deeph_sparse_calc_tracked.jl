#!/usr/bin/env julia
"""Run DeepH sparse_calc.jl while recording adjacent-k eigenvector overlaps."""

const DEEPH_SPARSE_CALC = get(
    ENV,
    "DEEPH_SPARSE_CALC",
    joinpath(dirname(@__DIR__), "..", "..", "DeepH-pack", "deeph", "inference", "sparse_calc.jl"),
)

source = read(DEEPH_SPARSE_CALC, String)
needle = "egval_sub = real(1 ./ egval_sub_inv) .+ (fermi_level)\n\n                        # orthogonalize"
replacement = "egval_sub = real(1 ./ egval_sub_inv) .+ (fermi_level)\n" *
              "                        record_band_overlap!(parsed_args[\"output_dir\"], idx_k, egval_sub, egvec_sub)\n\n" *
              "                        # orthogonalize"
patched = replace(source, needle => replacement; count=1)
patched == source && error("DeepH band eigensolver changed; refusing an unsafe tracking wrapper")
patched_without_main = replace(patched, r"\nmain\(\)\s*$" => "")
patched == patched_without_main && error("DeepH sparse_calc.jl no longer ends in main()")
include_string(Main, patched_without_main, DEEPH_SPARSE_CALC)

const PREVIOUS_BAND_VECTORS = Ref{Any}(nothing)

function record_band_overlap!(output_dir, k_index, energies, vectors)
    previous = PREVIOUS_BAND_VECTORS[]
    overlap = previous === nothing ? nothing : abs2.(previous' * vectors)
    payload = Dict(
        "k_index" => k_index - 1,
        "energies_eV" => collect(real(energies)),
        "overlap_from_previous" => (
            overlap === nothing ? nothing : [collect(overlap[row, :]) for row in axes(overlap, 1)]
        ),
    )
    open(joinpath(output_dir, "band_tracking_$(lpad(k_index - 1, 3, '0')).json"), "w") do handle
        JSON.print(handle, payload, 2)
    end
    PREVIOUS_BAND_VECTORS[] = copy(vectors)
end

main()
