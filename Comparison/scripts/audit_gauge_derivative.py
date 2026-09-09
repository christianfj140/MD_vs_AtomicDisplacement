#!/usr/bin/env python3
"""Audit the moving vacuum gauge on existing PROD-SZ five-point stencils."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "Comparison/scripts"), str(ROOT / "shared")]
from artifact_signature import file_sha256
from run_nested_basis_sentinel import _matrix_data, _production_run
from run_nested_basis_ladder import _rms
from onlys_semantic_preflight import _dense_k_points

WEIGHTS = {-2: 1., -1: -8., 1: 8., 2: -1.}
H = .005
LIMIT = .001  # existing connection budget, eV/Ang


def gauge_residual(vacuum, overlap, h=H):
    dc = sum(w * vacuum[j] for j, w in WEIGHTS.items()) / (12*h)
    difference = sum(-w*(vacuum[j]-vacuum[0])*overlap[j]
                     for j, w in WEIGHTS.items()) / (12*h)
    return difference + dc*overlap[0]


def main():
    import sisl
    rows = []
    source = Path(sisl.__file__).parent / 'io/siesta/_src/tshs_read.f90'
    semantics = 'H(:,is) = H(:,is) - Ef * S(:)' in source.read_text()
    for direction in ('C1_x', 'C1_z'):
        data = {j: _matrix_data(_production_run(direction, j)) for j in (-2,-1,0,1,2)}
        dc = sum(w * data[j][3] for j,w in WEIGHTS.items()) / (12*H)
        checks = []
        for k in _dense_k_points():
            s0 = np.asarray(data[0][1].Sk(k=k, format='array'))
            # Central K and connection terms are identical in the two gauges.
            residual = gauge_residual({j: data[j][3] for j in data},
                                      {j: np.asarray(data[j][1].Sk(k=k,format='array')) for j in data})
            rms, maximum = _rms(residual, (s0+s0.conj().T)/2)
            checks.append({'k': k.tolist(), 'identity_residual_ev_per_ang': rms,
                           'max_ev_per_ang': maximum})
        rows.append({'direction': direction, 'vacuum_ev': {str(j): data[j][3] for j in data},
                     'vacuum_derivative_ev_per_ang': dc, 'checks': checks,
                     'worst_residual_ev_per_ang': max(r['identity_residual_ev_per_ang'] for r in checks),
                     'inputs': {str(j): file_sha256(_production_run(direction,j)/f'{data[j][0]}.TSHS') for j in data}})
    report = {'gate': 'gauge_derivative_audit', 'schema': 'gauge_derivative_audit_v1',
              'verdict': 'PASS' if semantics and all(r['worst_residual_ev_per_ang'] < LIMIT for r in rows) else 'NO_GO',
              'threshold_ev_per_ang': LIMIT, 'primary_gauge': 'H_absolute-c_vac(R)*S(R)',
              'tshs_semantics': {'reader_subtracts_Ef_S': semantics, 'source': str(source),
                                 'source_sha256': file_sha256(source),
                                 'absolute_H': 'sisl.Hk + tshs.read_fermi_level()*Sk'},
              'finite_difference_note': 'D5[(c-c0)S] differs from D5[c]*S0 by the measured product-rule truncation residual',
              'directions': rows}
    out = ROOT/'Comparison/results/epc/pao_flow_audit'
    out.mkdir(parents=True, exist_ok=True)
    (out/'gauge_derivative_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['verdict'], [(r['direction'], r['vacuum_derivative_ev_per_ang'], r['worst_residual_ev_per_ang']) for r in rows])
    return 0 if report['verdict']=='PASS' else 1

if __name__ == '__main__':
    raise SystemExit(main())
