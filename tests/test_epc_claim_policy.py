import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'Comparison/scripts'))
from epc_claim_policy import claim_policy
from audit_epc_backends import audit_runtime_manifests


def test_missing_evidence_never_promotes_claims(tmp_path):
    for status in ('NO_GO', 'BLOCKED', None):
        p = claim_policy({'delta_out_closure': status, 'previously_inspected': True})
        assert p['full_KS'] == 'BLOCKED'
        assert p['fine_tuning_candidate'] == 'BLOCKED'
        assert p['blind_holdout'] is False
    assert not audit_runtime_manifests([tmp_path/'missing.json'])[0]['valid']
