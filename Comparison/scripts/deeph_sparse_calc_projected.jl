#!/usr/bin/env julia
"""Run DeepH sparse_calc.jl and stream S-aware Mulliken projections."""

const DEEPH_SPARSE_CALC = get(
    ENV,
    "DEEPH_SPARSE_CALC",
    joinpath(dirname(@__DIR__), "..", "..", "DeepH-pack", "deeph", "inference", "sparse_calc.jl"),
)

include(joinpath(@__DIR__, "deeph_mulliken_weights.jl"))
patched = patch_sparse_calc_for_mulliken(read(DEEPH_SPARSE_CALC, String))
patched_without_main = replace(patched, r"\nmain\(\)\s*$" => "")
patched == patched_without_main && error("DeepH sparse_calc.jl no longer ends in main()")
include_string(Main, patched_without_main, DEEPH_SPARSE_CALC)

main()
