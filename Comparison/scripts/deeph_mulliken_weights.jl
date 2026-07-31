"""Patch DeepH's sparse eigensolver and stream S-aware Mulliken projections."""

function patch_sparse_calc_for_mulliken(source)
    band_eigen_needle = "                        egval_sub = real(1 ./ egval_sub_inv) .+ (fermi_level)\n\n                        # orthogonalize the eigenvectors"
    band_eigen_replacement = "                        egval_sub = real(1 ./ egval_sub_inv) .+ (fermi_level)\n" *
                             "                        physical_egvec = egvec_sub\n\n" *
                             "                        # orthogonalize the eigenvectors"
    dos_eigen_needle = "                egval_sub = real(1 ./ egval_sub_inv) .+ (fermi_level)\n\n                # orthogonalize the eigenvectors"
    dos_eigen_replacement = "                egval_sub = real(1 ./ egval_sub_inv) .+ (fermi_level)\n" *
                            "                physical_egvec = egvec_sub\n\n" *
                            "                # orthogonalize the eigenvectors"
    all(occursin(needle, source) for needle in (band_eigen_needle, dos_eigen_needle)) ||
        error("DeepH eigensolver changed; expected band and DOS projection insertion points")
    patched = replace(source, band_eigen_needle => band_eigen_replacement; count=1)
    patched = replace(patched, dos_eigen_needle => dos_eigen_replacement; count=1)

    band_projected_needle = "                            egvec = egvec_sub * egvec\n"
    dos_projected_needle = "                    egvec = egvec_sub * egvec\n"
    patched = replace(
        patched,
        band_projected_needle => band_projected_needle * "                            physical_egvec = egvec\n";
        count=1,
    )
    patched = replace(
        patched,
        dos_projected_needle => dos_projected_needle * "                    physical_egvec = egvec\n";
        count=1,
    )

    band_needle = "                    egvals[:, idx_k] = egval\n                    if which_k == 0"
    band_replacement = "                    record_mulliken!(parsed_args[\"output_dir\"], idx_k, egval, physical_egvec, H_k, S_k)\n" *
                       band_needle
    occursin(band_needle, patched) || error("DeepH band output insertion point changed")
    patched = replace(patched, band_needle => band_replacement; count=1)

    dos_needle = "            egvals[:, idx_k] = egval\n            println(\"Time for solving No.\$idx_k eigenvalues"
    dos_replacement = "            record_mulliken!(parsed_args[\"output_dir\"], idx_k, egval, physical_egvec, H_k, S_k)\n" *
                      dos_needle
    occursin(dos_needle, patched) || error("DeepH DOS output insertion point changed")
    patched = replace(patched, dos_needle => dos_replacement; count=1)

    quality_needle = "                    S_k = (S_k + S_k') / 2"
    patched = replace(
        patched,
        quality_needle => "                    record_matrix_quality!(parsed_args[\"output_dir\"], idx_k, H_k, S_k)\n" * quality_needle;
        count=1,
    )
    quality_dos_needle = "            S_k = (S_k + S_k') / 2"
    patched = replace(
        patched,
        quality_dos_needle => "            record_matrix_quality!(parsed_args[\"output_dir\"], idx_k, H_k, S_k)\n" * quality_dos_needle;
        count=1,
    )
    return patched
end

const MULLIKEN_INPUT = Ref{Any}(nothing)

function mulliken_input()
    if MULLIKEN_INPUT[] === nothing
        payload = JSON.parsefile(ENV["DEEPH_MULLIKEN_GROUPS"])
        payload["groups"] = Dict(
            name => Int.(indices) for (name, indices) in payload["groups"]
        )
        MULLIKEN_INPUT[] = payload
    end
    return MULLIKEN_INPUT[]
end

function record_matrix_quality!(output_dir, k_index, H, S)
    relative_hermiticity(matrix) = norm(matrix - matrix') / max(norm(matrix), eps(Float64))
    payload = Dict(
        "k_index" => k_index - 1,
        "h_relative_hermiticity_before_solver_symmetrization" => relative_hermiticity(H),
        "s_relative_hermiticity_before_solver_symmetrization" => relative_hermiticity(S),
    )
    open(joinpath(output_dir, "matrix_quality_$(lpad(k_index - 1, 3, '0')).json"), "w") do handle
        JSON.print(handle, payload, 2)
    end
end

function record_mulliken!(output_dir, k_index, energies, vectors, H, S)
    length(energies) == size(vectors, 2) ||
        error("Cannot project eigenpairs after ill-conditioned modes were removed")
    input = mulliken_input()
    groups = input["groups"]
    norbits, nbands = size(vectors)
    all(all(index -> 1 <= index <= norbits, indices) for indices in Base.values(groups)) ||
        error("Mulliken group contains an out-of-range orbital index")

    raw_normalizations = zeros(Float64, nbands)
    normalizations = ones(Float64, nbands)
    residuals = zeros(Float64, nbands)
    weights = Dict(name => zeros(Float64, nbands) for name in Base.keys(groups))
    chunk_size = parse(Int, get(ENV, "DEEPH_PROJECTION_CHUNK_BANDS", "16"))
    for first_band in 1:chunk_size:nbands
        last_band = min(first_band + chunk_size - 1, nbands)
        columns = first_band:last_band
        chunk = @view vectors[:, columns]
        Schunk = S * chunk
        Hchunk = H * chunk
        for (column_index, band) in enumerate(columns)
            vector = @view chunk[:, column_index]
            Svector = @view Schunk[:, column_index]
            Hvector = @view Hchunk[:, column_index]
            raw_normalization = real(dot(vector, Svector))
            isfinite(raw_normalization) && raw_normalization > 0 ||
                error("Non-positive or non-finite C†SC normalization at k=$k_index band=$band")
            raw_normalizations[band] = raw_normalization
            scale = inv(sqrt(raw_normalization))
            vector .*= scale
            Svector .*= scale
            Hvector .*= scale
            normalization = real(dot(vector, Svector))
            normalizations[band] = normalization
            energy = real(energies[band])
            numerator = norm(Hvector - energy * Svector)
            residuals[band] = numerator / max(norm(Hvector) + abs(energy) * norm(Svector), eps(Float64))
            for (name, indices) in groups
                weights[name][band] = real(dot(@view(vector[indices]), @view(Svector[indices]))) / normalization
            end
        end
    end
    payload = Dict(
        "k_index" => k_index - 1,
        "energies_eV" => collect(real(energies)),
        "normalization_cdagger_s_c" => normalizations,
        "raw_solver_cdagger_s_c" => raw_normalizations,
        "generalized_relative_residual" => residuals,
        "mulliken_weights" => weights,
        "method" => "Re sum_{mu in A} conj(C_mu) (S C)_mu / (Cdagger S C)",
        "eigenvectors_persisted" => false,
    )
    open(joinpath(output_dir, "mulliken_$(lpad(k_index - 1, 3, '0')).json"), "w") do handle
        JSON.print(handle, payload, 2)
    end
end
