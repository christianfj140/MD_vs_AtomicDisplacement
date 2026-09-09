"""Claims require positive evidence; successful gate execution is not that evidence."""


def claim_policy(gates):
    prerequisites = ("gauge_derivative_audit", "graph2mat_target_gauge_contract", "numerical_PAO_convergence",
                     "checkpoint_lineage", "final_case_exclusion", "gpu_runtime", "GO-5", "GO-7")
    missing = [name for name in prerequisites if gates.get(name) != "PASS"]
    # Graph2Mat's target is H_abs - E_F(R).S; the primary PAO reference is
    # H_abs - c_vac(R).S. (c_vac - E_F)' = 3.4869 meV/Ang, not negligible
    # against the 5 meV/Ang convergence budget, so knowing the conversion
    # (gauge_derivative_audit = PASS) is not the same as having applied it.
    # Until the final comparator is verified to convert, a residual computed
    # across the two gauges is not a model error and cannot promote a claim.
    if gates.get("final_comparator_gauge_conversion_verified") is not True:
        missing = missing + ["final_comparator_gauge_conversion_verified"]
    return {
        "observable": "finite-PAO covariant response in PROD-SZ",
        "full_KS": "BLOCKED" if gates.get("delta_out_closure") != "PASS" else "UNADJUDICATED",
        "fine_tuning_candidate": "BLOCKED" if missing else "UNADJUDICATED",
        "final_claim": "BLOCKED" if missing else "UNADJUDICATED",
        "blocked_by": missing,
        "blind_holdout": False if gates.get("previously_inspected") is True else "UNPROVEN",
    }
