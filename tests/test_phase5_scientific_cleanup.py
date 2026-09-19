"""Regressions for k-point RMS aggregation and validation-only TBG selection."""
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Comparison/scripts'))
import evaluate_deeph_kpoint_metrics as deeph
import evaluate_hamiltonian_metrics as g2m
import run_tbg_registry_50_50 as tbg


def test_same_matrices_give_same_kpoint_rms_in_both_evaluators(tmp_path, monkeypatch):
    sample = NS(sample='sample', structure_path=None)
    processed = tmp_path / sample.sample
    processed.mkdir()
    for name in ('hamiltonians.h5', 'overlaps.h5', 'orbital_types.dat', 'prediction.h5'):
        (processed / name).touch()
    prediction = processed / 'prediction.h5'
    reference = processed / 'hamiltonians.h5'
    grid = g2m.MonkhorstPackKGrid((2, 1, 1), (0., 0., 0.), False, 'synthetic',
                                  ((0., 0., 0.), (0.5, 0., 0.)), (0.5, 0.5))
    def matrix(path, k):
        return np.array([[1. + (0.02 if path == prediction and k[0] else 0.)]])
    # Only input adapters are replaced; both evaluators run their real numerical paths.
    monkeypatch.setattr(deeph, 'parse_monkhorst_pack_kgrid', lambda _: grid)
    monkeypatch.setattr(deeph, 'assemble_hk', lambda path, _, k: matrix(path, k))
    monkeypatch.setattr(deeph, 'graph2mat_reference_path', lambda *a: None)
    monkeypatch.setattr(deeph, 'adapt_deeph_prediction_sample', lambda **kw: NS(
        metrics_ready=True, prediction_path=prediction, metric_fields=lambda: {}, to_dict=lambda: {}))
    args = NS(processed_dir=tmp_path, graph2mat_result_dir=tmp_path, predictions_dir=tmp_path,
              prediction_filename='prediction.h5', generate_predictions=False, output_dir=tmp_path/'deeph',
              disable_low_energy=False, low_energy_n_states=1, low_energy_alignment='none')
    deeph_rows = defaultdict(list)
    assert deeph.evaluate_sample(args, sample, deeph_rows)['status'] == 'ok'

    def read_matrix(path):
        return g2m.MatrixData(path, sparse.csr_matrix([[1.]]), sparse.eye(1, format='csr'),
                              np.array([1.]), None, None, False, True, None, spin_kind='unpolarized')
    monkeypatch.setattr(g2m, 'read_matrix', read_matrix)
    monkeypatch.setattr(g2m.sisl, 'get_sile', lambda path: NS(read_hamiltonian=lambda: NS(
        orthogonal=False, Hk=lambda k, **kw: matrix(Path(path), k), Sk=lambda k, **kw: np.eye(1))))
    g2m_rows = g2m.evaluate_kpoint_sample(sample.sample, prediction, reference, tmp_path/'g2m', grid,
        target_component_policy='h_only', low_energy_enabled=True, low_energy_n_states=1,
        low_energy_alignment='none')
    assert not g2m_rows['fatal_errors']
    values = []
    for rows in (g2m_rows, deeph_rows):
        per_k = [r['h_rmse_eV'] for r in rows['kpoint_matrix'] if r['row_type'] == 'per_k']
        assert per_k == pytest.approx([0., 0.020])
        value = next(r['h_rmse_eV'] for r in rows['kpoint_matrix'] if r['row_type'] == 'weighted_sample')
        assert value * 1000 == pytest.approx(14.142135623730951)
        assert value != pytest.approx(0.010)
        values.append(value)
    assert values[0] == pytest.approx(values[1])


def test_tbg_train_returns_validation_best_only(tmp_path, monkeypatch):
    # Exercise train's real return path without loading a model or running a trainer.
    selected = tmp_path / 'best.ckpt'
    selected.touch()
    last = tmp_path / 'training/checkpoints/last.ckpt'
    last.parent.mkdir(parents=True)
    last.touch()
    callback = MagicMock(return_value=NS(best_model_path=str(selected)))
    trainer = MagicMock()
    modules = {
        'pytorch_lightning': NS(Trainer=trainer, seed_everything=lambda *a, **kw: None),
        'pytorch_lightning.callbacks': NS(EarlyStopping=MagicMock, ModelCheckpoint=callback),
        'pytorch_lightning.loggers': NS(TensorBoardLogger=MagicMock()),
        'torch_geometric.loader': NS(DataLoader=MagicMock()),
        'graph2mat': NS(AtomicTableWithEdges=MagicMock()),
        'graph2mat.tools.lightning': NS(MatrixDataModule=MagicMock),
        'graph2mat.tools.lightning.models.mace': NS(LitMACEMatrixModel=MagicMock()),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    import torch
    monkeypatch.setattr(torch, 'set_float32_matmul_precision', lambda *a: None)
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', False)
    monkeypatch.setattr(torch.backends.cudnn, 'allow_tf32', False)
    monkeypatch.setattr(tbg, 'OUTPUT', tmp_path)
    monkeypatch.setattr(tbg, 'split_paths', lambda: (['md', 'registry'], [False, True], ['validation'], ['test']))
    monkeypatch.setattr(tbg, 'disk_guard', lambda: 90.)
    monkeypatch.setattr(tbg, 'update_status', lambda *a, **kw: None)
    args = NS(seed=0, learning_rate=5e-4, batch_size=40, loader_threads=0, patience=60,
              max_epochs=250, resume=False)
    assert tbg.train(args) == [selected]
    assert callback.call_args.kwargs['monitor'] == 'val_loss'
    assert callback.call_args.kwargs['mode'] == 'min'
    trainer.return_value.fit.assert_called_once()


def test_test_scores_cannot_change_validation_selected_checkpoint(tmp_path, monkeypatch):
    selected, loser = Path('best.ckpt'), Path('last.ckpt')
    test_scores = {selected.name: .005, loser.name: .020}
    calls = []
    def evaluate(command, **kw):
        candidates = [command[i+1] for i, arg in enumerate(command) if arg == '--checkpoint']
        assert candidates == [str(selected)]  # Losing checkpoint never reaches either evaluator.
        split = command[command.index('--split')+1]
        calls.append((split, tuple(candidates)))
        root = Path(command[command.index('--output-root')+1])
        root.mkdir(parents=True, exist_ok=True)
        error = .003 if split == 'validation' else test_scores[selected.name]
        payload = {'checkpoints': {selected.name: {'single_offset': {'offset_eV': 0.}, 'per_sample': [
            {'kpoint': 'K', 'frontier_n_states': 4, 'frontier_rmse_eV': error,
             'frontier_global_shift_eV': 0.}]}}, 'frozen_offsets_eV': {selected.name: 0.}}
        (root / f'checkpoint_spectral_metrics_{split}.json').write_text(json.dumps(payload))
    monkeypatch.setattr(tbg.subprocess, 'run', evaluate)
    monkeypatch.setattr(tbg, 'disk_guard', lambda: 90.)
    first = tbg.run_gate([selected], output=tmp_path)
    validation = (tmp_path/'eval_validation/checkpoint_spectral_metrics_validation.json').read_bytes()
    test_scores[selected.name], test_scores[loser.name] = test_scores[loser.name], test_scores[selected.name]
    second = tbg.run_gate([selected], output=tmp_path)
    assert first['checkpoint'] == second['checkpoint'] == selected.name
    assert first['status'] == 'passed' and second['status'] == 'failed'
    assert first['score_eV'] == pytest.approx(.005) and second['score_eV'] == pytest.approx(.020)
    assert validation == (tmp_path/'eval_validation/checkpoint_spectral_metrics_validation.json').read_bytes()
    assert calls == [('validation', ('best.ckpt',)), ('test', ('best.ckpt',))] * 2


@pytest.mark.parametrize('candidates', [[], [Path('best.ckpt'), Path('last.ckpt')]])
def test_gate_rejects_non_singleton_input_before_any_evaluation(candidates, tmp_path, monkeypatch):
    runner = MagicMock()
    monkeypatch.setattr(tbg.subprocess, 'run', runner)
    with pytest.raises(ValueError, match='exactly one checkpoint selected by validation'):
        tbg.run_gate(candidates, output=tmp_path)
    runner.assert_not_called()
    assert not list(tmp_path.iterdir())


def test_gate_rejects_report_with_unselected_checkpoint(tmp_path, monkeypatch):
    root = tmp_path/'eval_registry_holdout'
    root.mkdir()
    (root/'checkpoint_spectral_metrics_test.json').write_text(json.dumps({'checkpoints': {'last.ckpt': {}}}))
    monkeypatch.setattr(tbg.subprocess, 'run', lambda *a, **kw: None)
    monkeypatch.setattr(tbg, 'disk_guard', lambda: 90.)
    with pytest.raises(ValueError, match='only the validation-selected checkpoint'):
        tbg.run_gate([Path('best.ckpt')], output=tmp_path)
