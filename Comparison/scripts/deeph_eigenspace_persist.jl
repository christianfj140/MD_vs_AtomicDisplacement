"""Persist S-normalised eigenspaces of the generalized problem (C15 / E-F_001-S18).

DeepH hands back ARPACK Ritz vectors. Those are S-orthogonal only for distinct,
converged eigenvalues and are an arbitrary basis inside a degenerate cluster,
which is precisely the case EPC has to contract (C_m' * Delta * C_n). So the
requested window is re-solved exactly once here:

    M = V'SV  ->  C = V M^(-1/2)      (Loewdin in the S metric)
    C' H C    ->  Rayleigh-Ritz       (eigenvalues + a fixed gauge in the window)

which makes C'SC = I to roundoff *by construction*, and hands Python the metric
condition number kappa(M) its identity tolerance is derived from. The window is
never cut through a degenerate cluster: it grows outwards until it closes on a
real gap, so a persisted subspace is a whole subspace.

Env: DEEPH_EIGENSPACE_WINDOW_EV (half width around the shift, 0 = every solved
state), DEEPH_EIGENSPACE_DEGENERACY_EV, DEEPH_EIGENSPACE_MIN_METRIC.
"""

const EIGENSPACE_SCHEMA = "persisted_eigenspace_v1"

# ponytail: same two anchors deeph_mulliken_weights.jl uses, duplicated behind an
# idempotence guard so either patch can run alone or after the other. Factor out
# only if a third consumer of the pre-QR Ritz vectors appears.
function ensure_physical_egvec(source)
    occursin("physical_egvec", source) && return source
    band_needle = "                        egval_sub = real(1 ./ egval_sub_inv) .+ (fermi_level)\n\n                        # orthogonalize the eigenvectors"
    dos_needle = "                egval_sub = real(1 ./ egval_sub_inv) .+ (fermi_level)\n\n                # orthogonalize the eigenvectors"
    all(occursin(needle, source) for needle in (band_needle, dos_needle)) ||
        error("DeepH eigensolver changed; expected band and DOS Ritz-vector capture points")
    patched = replace(
        source,
        band_needle => "                        egval_sub = real(1 ./ egval_sub_inv) .+ (fermi_level)\n" *
                       "                        physical_egvec = egvec_sub\n\n" *
                       "                        # orthogonalize the eigenvectors";
        count=1,
    )
    patched = replace(
        patched,
        dos_needle => "                egval_sub = real(1 ./ egval_sub_inv) .+ (fermi_level)\n" *
                      "                physical_egvec = egvec_sub\n\n" *
                      "                # orthogonalize the eigenvectors";
        count=1,
    )
    # The leading newline pins the indentation: without it the 20-space DOS
    # needle also matches 8 characters into the 28-space band line.
    for indent in ("                            ", "                    ")
        needle = "\n" * indent * "egvec = egvec_sub * egvec\n"
        occursin(needle, patched) || error("DeepH ill-projection insertion point changed")
        patched = replace(
            patched,
            needle => needle * indent * "physical_egvec = egvec\n";
            count=1,
        )
    end
    return patched
end

function patch_sparse_calc_for_eigenspace(source)
    patched = ensure_physical_egvec(source)
    band_needle = "                    egvals[:, idx_k] = egval\n                    if which_k == 0"
    occursin(band_needle, patched) || error("DeepH band output insertion point changed")
    patched = replace(
        patched,
        band_needle => "                    record_eigenspace!(parsed_args[\"output_dir\"], idx_k, [kx, ky, kz], egval, physical_egvec, H_k, S_k, fermi_level)\n" *
                       band_needle;
        count=1,
    )
    dos_needle = "            egvals[:, idx_k] = egval\n            println(\"Time for solving No.\$idx_k eigenvalues"
    occursin(dos_needle, patched) || error("DeepH DOS output insertion point changed")
    return replace(
        patched,
        dos_needle => "            record_eigenspace!(parsed_args[\"output_dir\"], idx_k, [kx, ky, kz], egval, physical_egvec, H_k, S_k, fermi_level)\n" *
                      dos_needle;
        count=1,
    )
end

function eigenspace_settings()
    return (
        half_width = parse(Float64, get(ENV, "DEEPH_EIGENSPACE_WINDOW_EV", "0")),
        degeneracy = parse(Float64, get(ENV, "DEEPH_EIGENSPACE_DEGENERACY_EV", "1e-6")),
        min_metric = parse(Float64, get(ENV, "DEEPH_EIGENSPACE_MIN_METRIC", "1e-10")),
    )
end

"""Ascending-energy band indices inside the window, never splitting a cluster."""
function eigenspace_window(energies, shift, half_width, degeneracy)
    order = sortperm(energies)
    half_width > 0 || return order
    inside = findall(position -> abs(energies[order[position]] - shift) <= half_width, eachindex(order))
    isempty(inside) && error("No solved state lies within $(half_width) eV of the shift $(shift) eV")
    first_position, last_position = first(inside), last(inside)
    while first_position > 1 &&
          abs(energies[order[first_position]] - energies[order[first_position - 1]]) <= degeneracy
        first_position -= 1
    end
    while last_position < length(order) &&
          abs(energies[order[last_position + 1]] - energies[order[last_position]]) <= degeneracy
        last_position += 1
    end
    return order[first_position:last_position]
end

function write_complex_matrix(path, matrix)
    open(path, "w") do handle
        write(handle, Array{ComplexF64}(matrix))
    end
end

function record_eigenspace!(output_dir, k_index, kfrac, energies, vectors, H, S, shift)
    length(energies) == size(vectors, 2) ||
        error("Cannot persist an eigenspace after ill-conditioned modes were removed")
    settings = eigenspace_settings()
    solver_energies = real(energies)
    indices = eigenspace_window(solver_energies, shift, settings.half_width, settings.degeneracy)

    V = Matrix{ComplexF64}(vectors[:, indices])
    SV = S * V
    metric = V' * SV
    metric_values, metric_vectors = eigen(Hermitian((metric + metric') / 2))
    minimum(metric_values) > settings.min_metric || error(
        "Window metric V'SV has eigenvalue $(minimum(metric_values)) <= $(settings.min_metric) at " *
        "k=$(k_index): the persisted vectors do not span an S-independent subspace"
    )
    loewdin = metric_vectors * Diagonal(metric_values .^ (-0.5)) * metric_vectors'
    C = V * loewdin
    SC = SV * loewdin

    HC = H * C
    subspace = C' * HC
    ritz_values, gauge = eigen(Hermitian((subspace + subspace') / 2))
    C = C * gauge
    SC = SC * gauge
    HC = HC * gauge

    overlap = C' * SC
    nstates = length(ritz_values)
    residuals = [
        norm(view(HC, :, n) - ritz_values[n] * view(SC, :, n)) /
        max(norm(view(HC, :, n)) + abs(ritz_values[n]) * norm(view(SC, :, n)), eps(Float64))
        for n in 1:nstates
    ]

    tag = lpad(k_index - 1, 3, '0')
    vectors_file = "eigenspace_vectors_$(tag).bin"
    overlap_file = "eigenspace_overlap_$(tag).bin"
    write_complex_matrix(joinpath(output_dir, vectors_file), C)
    write_complex_matrix(joinpath(output_dir, overlap_file), overlap - I)
    payload = Dict(
        "schema" => EIGENSPACE_SCHEMA,
        "k_index" => k_index - 1,
        "k_fractional" => collect(Float64, kfrac),
        "shift_eV" => shift,
        "window_half_width_eV" => settings.half_width > 0 ? settings.half_width : nothing,
        "degeneracy_tolerance_eV" => settings.degeneracy,
        "solver_band_indices" => [index - 1 for index in indices],
        "solver_energies_eV" => solver_energies[indices],
        "energies_eV" => ritz_values,
        "maximum_energy_shift_vs_solver_eV" => maximum(abs.(ritz_values .- solver_energies[indices])),
        "generalized_relative_residual" => residuals,
        "window_metric_minimum_eigenvalue" => minimum(metric_values),
        "window_metric_maximum_eigenvalue" => maximum(metric_values),
        "window_metric_condition_number" => maximum(metric_values) / minimum(metric_values),
        "norbits" => size(C, 1),
        "state_count" => nstates,
        "dtype" => "complex128",
        "storage_order" => "fortran",
        "vectors_file" => vectors_file,
        "overlap_minus_identity_file" => overlap_file,
        "method" => "loewdin_S_metric_then_rayleigh_ritz_in_window",
        "eigenvectors_persisted" => true,
    )
    open(joinpath(output_dir, "eigenspace_$(tag).json"), "w") do handle
        JSON.print(handle, payload, 2)
    end
end
