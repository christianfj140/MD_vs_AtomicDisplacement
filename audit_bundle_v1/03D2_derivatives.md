# Dossier 3B — Derivadas FD y autograd

## Objeto de revisión

Auditar definición de dH/dR, stencils, signo, unidades, delta, ruido, soporte sparse, geometrías, sum rule traslacional y gates de comparabilidad.

## Condiciones del contexto

- Contexto ciego: no contiene informes de auditoría previos ni prompts históricos.
- Las líneas prefijadas `NNNNN |` conservan la numeración del archivo fuente.
- Los extractos Python omiten funciones operativas sin semántica científica directa.
- Ausencia de un archivo o función en este dossier significa `no evaluable`, no `correcto`.

## `Comparison/config/derivative_metrics_only_existing_artifacts.json`

SHA-256: `862d9f2c6942f52c69510a0c8d4e2cfe857468c4c56526df7a4edbefe4bddec8`

```json
00001 | {
00002 |   "workflow_mode": "derivative_metrics_only",
00003 |   "derivative": {
00004 |     "enabled": true,
00005 |     "result_dir": "Comparison/results/PLACEHOLDER_derivative_workflow",
00006 |     "method": "central",
00007 |     "base_split": "test",
00008 |     "skip_if_exists": true
00009 |   }
00010 | }
```

## `Comparison/config/derivative_stencils_only_minimal.json`

SHA-256: `18f8a856965ff2eaa564bc7339434064328a075e3abe3e8ec3a9cbc4fc1ad521`

```json
00001 | {
00002 |   "workflow_mode": "derivative_stencils_only",
00003 |   "derivative": {
00004 |     "enabled": true,
00005 |     "source_dataset_root": "Comparison/datasets/PLACEHOLDER_joint_dataset",
00006 |     "output_root": "Comparison/results/PLACEHOLDER_derivative_stencils_only",
00007 |     "method": "central",
00008 |     "delta_ang": [
00009 |       0.005,
00010 |       0.01
00011 |     ],
00012 |     "base_split": "test",
00013 |     "max_base_snapshots": 1,
00014 |     "atoms": [
00015 |       "0"
00016 |     ],
00017 |     "axes": [
00018 |       "x"
00019 |     ],
00020 |     "overwrite": false,
00021 |     "skip_if_exists": true
00022 |   }
00023 | }
```

## `Comparison/config/adaptive_derivative_selection_smoke.json`

SHA-256: `cc15929fafb4dd28e095b851b9490da8622a27fef29027b446455882cee80bac`

```json
00001 | {
00002 |   "schema": "adaptive_derivative_selection_smoke_v1",
00003 |   "output_root": "Comparison/results/adaptive_derivative_selection_smoke",
00004 |   "derivative": {
00005 |     "method": "central",
00006 |     "base_split": "test",
00007 |     "base_selection_policy": "adaptive_min_fraction",
00008 |     "min_base_snapshots": 20,
00009 |     "base_fraction": 0.2,
00010 |     "base_selection_seed": 1,
00011 |     "delta_ang": [0.01],
00012 |     "atoms": ["0"],
00013 |     "axes": ["x"]
00014 |   },
00015 |   "cases": [
00016 |     {
00017 |       "label": "n_test_10",
00018 |       "n_test": 10,
00019 |       "expected_selected_base_snapshots": 10
00020 |     },
00021 |     {
00022 |       "label": "n_test_80",
00023 |       "n_test": 80,
00024 |       "expected_selected_base_snapshots": 20
00025 |     },
00026 |     {
00027 |       "label": "n_test_110",
00028 |       "n_test": 110,
00029 |       "expected_selected_base_snapshots": 22
00030 |     }
00031 |   ]
00032 | }
```

## `Comparison/scripts/graph2mat_autograd_derivatives.py`

SHA-256: `e290a60e16625b76796a767c48b6ef21038f2c884bc36b5217d144bc32e3bcbf`

```py
00001 | #!/usr/bin/env python3
00002 | """Vectorized autograd derivatives of Graph2Mat outputs w.r.t. atomic positions.
00003 | 
00004 | This module computes ``J = d outputs_flat / d positions`` for a Graph2Mat
00005 | model loaded in memory, where ``outputs_flat`` is the concatenation of the
00006 | flattened ``node_labels`` and ``edge_labels`` predictions and ``positions``
00007 | are the cartesian atomic positions of a single structure.
00008 | 
00009 | Units
00010 | -----
00011 | Graph2Mat predicts Hamiltonian labels in eV from positions in Ang, so the
00012 | jacobian (and any derivative selected from it) is directly in eV/Ang. No unit
00013 | conversion is applied anywhere in this module.
00014 | 
00015 | Fixed neighbor topology
00016 | -----------------------
00017 | ``edge_index`` and ``shifts`` are taken from the input batch and kept constant
00018 | inside the differentiable closure. The jacobian is therefore the local
00019 | derivative of the model for the fixed neighbor topology of the base structure
00020 | (exact up to cutoff-induced connectivity changes, which are discontinuous
00021 | anyway).
00022 | 
00023 | Method trade-offs (measured on graphene 5x5, 50 atoms, 12800 outputs)
00024 | ---------------------------------------------------------------------
00025 | ``vmap_vjp_chunked`` (default)
00026 |     Classic autograd VJPs batched with ``torch.autograd.grad(...,
00027 |     is_grads_batched=True)`` over chunks of one-hot cotangents. Works with the
00028 |     real MACE-backed model and bounds memory: backward activations scale with
00029 |     ``chunk_size`` (chunk 512 exhausted 44 GiB on graphene 5x5; chunk 64 is
00030 |     safe). The final dense jacobian itself is tiny (~0.007 GiB float32).
00031 | ``jacrev`` / ``jacfwd``
00032 |     torch.func transforms. *Currently incompatible with MACE >= 0.3.x*: MACE's
00033 |     ``prepare_graph`` calls ``data["positions"].requires_grad_(True)`` inside
00034 |     the forward, which functorch transforms reject (RuntimeError). They remain
00035 |     available for functorch-compatible models (e.g. the fake models used in
00036 |     unit tests). ``jacfwd`` would otherwise be the natural choice here because
00037 |     ``n_inputs = n_atoms * 3`` (150) is much smaller than ``n_outputs``.
00038 | ``autograd_jacobian``
00039 |     ``torch.autograd.functional.jacobian(vectorize=True)``. Compatible with
00040 |     MACE, but it batches *all* ``n_outputs`` cotangents in a single backward,
00041 |     so memory explodes for realistic output counts. Only for small problems.
00042 | """
00043 | 
00044 | from __future__ import annotations
00045 | 
00046 | import sys
00047 | from copy import copy as _shallow_copy
00048 | from dataclasses import dataclass
00049 | from pathlib import Path
00050 | from typing import Any, Callable, Dict, Sequence
00051 | 
00052 | import torch
00053 | 
00054 | SCRIPT_DIR = Path(__file__).resolve().parent
00055 | if str(SCRIPT_DIR) not in sys.path:
00056 |     sys.path.insert(0, str(SCRIPT_DIR))
00057 | 
00058 | DEFAULT_OUTPUT_KEYS = ("node_labels", "edge_labels")
00059 | JACOBIAN_METHODS = (
00060 |     "auto",
00061 |     "jvp_double_backward",
00062 |     "vmap_vjp_chunked",
00063 |     "jacrev",
00064 |     "jacfwd",
00065 |     "autograd_jacobian",
00066 | )
00067 | # Backward-activation memory scales linearly with the cotangent chunk;
00068 | # 64 was measured safe on CPU for graphene 5x5 (512 triggered the OOM killer).
00069 | DEFAULT_JACOBIAN_CHUNK_SIZE = 64
00070 | 
00071 | 
00072 | class Graph2MatAutogradDerivativeError(RuntimeError):
00073 |     """Raised when the autograd derivative route cannot proceed safely."""
00074 | 
00075 | 
00076 | @dataclass(frozen=True)
00077 | class PredictionFlattenSpec:
00078 |     """Slicing/shape metadata to rebuild label dicts from a flat vector."""
00079 | 
00080 |     output_keys: tuple[str, ...]
00081 |     shapes: dict[str, tuple[int, ...]]
00082 |     slices: dict[str, tuple[int, int]]
00083 |     n_outputs: int
00084 | 
00085 | 
00086 | @dataclass(frozen=True)
00087 | class PositionJacobianResult:
00088 |     jacobian: torch.Tensor
00089 |     spec: PredictionFlattenSpec
00090 |     base_predictions: dict[str, torch.Tensor]
00091 |     method: str
00092 |     chunk_size: int | None
00093 |     n_atoms: int
00094 | 
00095 | 
00096 | def flatten_graph2mat_predictions(
00097 |     predictions: Dict[str, torch.Tensor],
00098 |     output_keys: Sequence[str] = DEFAULT_OUTPUT_KEYS,
00099 | ) -> tuple[torch.Tensor, PredictionFlattenSpec]:
00100 |     """Concatenate prediction tensors into a flat vector plus rebuild metadata."""
00101 | 
00102 |     flats: list[torch.Tensor] = []
00103 |     shapes: dict[str, tuple[int, ...]] = {}
00104 |     slices: dict[str, tuple[int, int]] = {}
00105 |     offset = 0
00106 |     for key in output_keys:
00107 |         if key not in predictions:
00108 |             raise Graph2MatAutogradDerivativeError(
00109 |                 f"Prediction output {key!r} is missing; available keys: {sorted(predictions)}"
00110 |             )
00111 |         tensor = predictions[key]
00112 |         if not isinstance(tensor, torch.Tensor):
00113 |             raise Graph2MatAutogradDerivativeError(
00114 |                 f"Prediction output {key!r} must be a torch.Tensor, got {type(tensor).__name__}."
00115 |             )
00116 |         flat = tensor.reshape(-1)
00117 |         shapes[key] = tuple(tensor.shape)
00118 |         slices[key] = (offset, offset + flat.numel())
00119 |         offset += flat.numel()
00120 |         flats.append(flat)
00121 |     outputs_flat = torch.cat(flats) if flats else torch.empty(0)
00122 |     spec = PredictionFlattenSpec(
00123 |         output_keys=tuple(output_keys),
00124 |         shapes=shapes,
00125 |         slices=slices,
00126 |         n_outputs=offset,
00127 |     )
00128 |     return outputs_flat, spec
00129 | 
00130 | 
00131 | def unflatten_graph2mat_prediction_vector(
00132 |     flat_vector: torch.Tensor,
00133 |     spec: PredictionFlattenSpec,
00134 | ) -> dict[str, torch.Tensor]:
00135 |     """Rebuild a predictions dict with the original keys/shapes from a flat vector."""
00136 | 
00137 |     if flat_vector.numel() != spec.n_outputs:
00138 |         raise Graph2MatAutogradDerivativeError(
00139 |             f"Flat vector has {flat_vector.numel()} elements, expected {spec.n_outputs}."
00140 |         )
00141 |     return {
00142 |         key: flat_vector[spec.slices[key][0] : spec.slices[key][1]].reshape(spec.shapes[key])
00143 |         for key in spec.output_keys
00144 |     }
00145 | 
00146 | 
00147 | def _clone_batch(batch: Any) -> Any:
00148 |     if hasattr(batch, "clone"):
00149 |         return batch.clone()
00150 |     if isinstance(batch, dict):
00151 |         return dict(batch)
00152 |     return _shallow_copy(batch)
00153 | 
00154 | 
00155 | def _batch_num_structures(batch: Any) -> int | None:
00156 |     num_graphs = getattr(batch, "num_graphs", None)
00157 |     if num_graphs is not None:
00158 |         return int(num_graphs)
00159 |     ptr = None
00160 |     try:
00161 |         ptr = batch["ptr"]
00162 |     except (KeyError, TypeError, IndexError):
00163 |         ptr = getattr(batch, "ptr", None)
00164 |     if ptr is not None:
00165 |         return max(int(len(ptr)) - 1, 0)
00166 |     return None
00167 | 
00168 | 
00169 | def require_single_structure_batch(batch: Any) -> None:
00170 |     """The derivative route maps one jacobian to one structure; reject batches > 1."""
00171 | 
00172 |     num_structures = _batch_num_structures(batch)
00173 |     if num_structures is not None and num_structures != 1:
00174 |         raise Graph2MatAutogradDerivativeError(
00175 |             "Autograd derivative predictions require batch_size=1 (one structure "
00176 |             f"per jacobian), got a batch with {num_structures} structures."
00177 |         )
00178 | 
00179 | 
00180 | def graph2mat_forward_labels(
00181 |     model: torch.nn.Module,
00182 |     batch: Any,
00183 |     positions: torch.Tensor,
00184 |     output_keys: Sequence[str] = DEFAULT_OUTPUT_KEYS,
00185 | ) -> tuple[torch.Tensor, PredictionFlattenSpec]:
00186 |     """Differentiable forward pass with ``positions`` substituted into the batch.
00187 | 
00188 |     Never uses ``torch.no_grad()``/``torch_predict()`` and never detaches or
00189 |     converts to numpy: the returned flat tensor stays connected to
00190 |     ``positions`` in the autograd graph. ``edge_index`` and ``shifts`` are
00191 |     left untouched (fixed topology).
00192 |     """
00193 | 
00194 |     model.eval()
00195 |     with torch.set_grad_enabled(True):
00196 |         data = _clone_batch(batch)
00197 |         data["positions"] = positions
00198 |         out = model(data)
00199 |         return flatten_graph2mat_predictions(out, output_keys=output_keys)
00200 | 
00201 | 
00202 | def _jacobian_vmap_vjp_chunked(
00203 |     forward: Callable[[torch.Tensor], torch.Tensor],
00204 |     positions: torch.Tensor,
00205 |     chunk_size: int,
00206 | ) -> tuple[torch.Tensor, torch.Tensor]:
00207 |     """Full jacobian via chunked batched VJPs (classic autograd + vmap backward).
00208 | 
00209 |     Chunks run over blocks of ``outputs_flat`` cotangents, never over single
00210 |     elements: each ``torch.autograd.grad(..., is_grads_batched=True)`` call is
00211 |     one vectorized backward for ``chunk_size`` one-hot cotangents.
00212 |     """
00213 | 
00214 |     positions_leaf = positions.detach().clone().requires_grad_(True)
00215 |     outputs = forward(positions_leaf)
00216 |     if not outputs.requires_grad:
00217 |         raise Graph2MatAutogradDerivativeError(
00218 |             "Model outputs are detached from positions; check for torch.no_grad()/"
00219 |             "detach() inside the forward pass."
00220 |         )
00221 |     n_outputs = outputs.numel()
00222 |     rows: list[torch.Tensor] = []
00223 |     for start in range(0, n_outputs, chunk_size):
00224 |         stop = min(start + chunk_size, n_outputs)
00225 |         basis = torch.zeros(
00226 |             (stop - start, n_outputs), dtype=outputs.dtype, device=outputs.device
00227 |         )
00228 |         basis[torch.arange(stop - start), torch.arange(start, stop)] = 1.0
00229 |         (vjp,) = torch.autograd.grad(
00230 |             outputs,
00231 |             positions_leaf,
00232 |             grad_outputs=basis,
00233 |             retain_graph=stop < n_outputs,
00234 |             is_grads_batched=True,
00235 |         )
00236 |         rows.append(vjp)
00237 |     jacobian = torch.cat(rows) if rows else torch.zeros((0, *positions.shape))
00238 |     return jacobian, outputs.detach()
00239 | 
00240 | 
00241 | def _jacobian_jvp_double_backward(
00242 |     forward: Callable[[torch.Tensor], torch.Tensor],
00243 |     positions: torch.Tensor,
00244 |     target_atoms: Sequence[int],
00245 | ) -> torch.Tensor:
00246 |     """Forward-mode jacobian columns for a few atoms, via forward-over-reverse.
00247 | 
00248 |     The finite-displacement stencils only request ``dH/dR`` for a handful of
00249 |     (atom, axis) pairs (typically atom 0, axes x/y/z), so computing the FULL
00250 |     reverse-mode jacobian over all ``n_outputs`` cotangents is wasteful: it
00251 |     scales with the number of Hamiltonian elements (~1e4), one backward per
00252 |     chunk. Forward-mode scales with the number of requested input directions
00253 |     instead (``len(target_atoms) * 3``).
00254 | 
00255 |     MACE calls ``data["positions"].requires_grad_(True)`` internally, which is
00256 |     incompatible with ``torch.func`` transforms (jvp/jacfwd). So we build the
00257 |     JVP by hand as a double backward: ``g = u^T J`` (u a differentiable dummy
00258 |     cotangent) is differentiated w.r.t. ``u`` along a one-hot input tangent,
00259 |     which yields ``J @ tangent`` — one column of the jacobian — using only
00260 |     ``torch.autograd.grad``. Numerically identical to the VJP route (verified:
00261 |     rel err ~4e-7, float32 noise) at ~4000x less cost.
00262 | 
00263 |     Returns a ``[n_outputs, n_atoms, 3]`` jacobian where only the columns of
00264 |     ``target_atoms`` are filled; all other atom columns are left as zeros (they
00265 |     are never selected downstream). The translation sum rule, which needs all
00266 |     atoms, is therefore only meaningful when every atom is a target.
00267 |     """
00268 | 
00269 |     positions_leaf = positions.detach().clone().requires_grad_(True)
00270 |     outputs = forward(positions_leaf)
00271 |     if not outputs.requires_grad:
00272 |         raise Graph2MatAutogradDerivativeError(
00273 |             "Model outputs are detached from positions; check for torch.no_grad()/"
00274 |             "detach() inside the forward pass."
00275 |         )
00276 |     n_outputs = int(outputs.numel())
00277 |     n_atoms = int(positions.shape[0])
00278 |     jacobian = torch.zeros(
00279 |         (n_outputs, n_atoms, 3), dtype=outputs.dtype, device=outputs.device
00280 |     )
00281 |     # u^T J: a differentiable dummy cotangent whose gradient reconstructs J columns.
00282 |     dummy = torch.zeros(n_outputs, dtype=outputs.dtype, device=outputs.device, requires_grad=True)
00283 |     (vjp_of_dummy,) = torch.autograd.grad(outputs, positions_leaf, grad_outputs=dummy, create_graph=True)
00284 |     unique_atoms = sorted({int(a) for a in target_atoms})
00285 |     for atom in unique_atoms:
00286 |         if not (0 <= atom < n_atoms):
00287 |             raise Graph2MatAutogradDerivativeError(
00288 |                 f"target atom {atom} is outside the structure with {n_atoms} atoms."
00289 |             )
00290 |         for axis in range(3):
00291 |             tangent = torch.zeros_like(positions_leaf)
00292 |             tangent[atom, axis] = 1.0
00293 |             # d(u^T J)/du contracted with the input tangent = J @ tangent = column.
00294 |             (column,) = torch.autograd.grad(
00295 |                 vjp_of_dummy, dummy, grad_outputs=tangent, retain_graph=True
00296 |             )
00297 |             jacobian[:, atom, axis] = column
00298 |     return jacobian
00299 | 
00300 | 
00301 | def compute_graph2mat_position_jacobian(
00302 |     model: torch.nn.Module,
00303 |     batch: Any,
00304 |     *,
00305 |     method: str = "auto",
00306 |     chunk_size: int | None = None,
00307 |     output_keys: Sequence[str] = DEFAULT_OUTPUT_KEYS,
00308 |     target_atoms: Sequence[int] | None = None,
00309 | ) -> PositionJacobianResult:
00310 |     """Compute ``J = d outputs_flat / d positions`` for a single-structure batch.
00311 | 
00312 |     Returns a jacobian of shape ``[n_outputs, n_atoms, 3]`` (same dtype/device
00313 |     as the batch positions), the flatten spec needed to rebuild label dicts,
00314 |     and the non-derived base predictions.
00315 | 
00316 |     ``target_atoms`` lists the atom indices whose derivative columns are
00317 |     actually consumed downstream (the stencil's requested atoms). When set and
00318 |     ``method`` resolves to forward-mode (``auto``/``jvp_double_backward``), only
00319 |     those columns are computed — orders of magnitude cheaper than the full
00320 |     reverse-mode jacobian. ``None`` means "all atoms" (full jacobian).
00321 |     """
00322 | 
00323 |     if method not in JACOBIAN_METHODS:
00324 |         raise Graph2MatAutogradDerivativeError(
00325 |             f"Unsupported jacobian method {method!r}. Use one of: {', '.join(JACOBIAN_METHODS)}."
00326 |         )
00327 |     require_single_structure_batch(batch)
00328 | 
00329 |     positions = batch["positions"]
00330 |     if positions.ndim != 2 or positions.shape[1] != 3:
00331 |         raise Graph2MatAutogradDerivativeError(
00332 |             f"positions must have shape [n_atoms, 3], got {tuple(positions.shape)}."
00333 |         )
00334 |     n_atoms = int(positions.shape[0])
00335 | 
00336 |     # Non-derived forward: base predictions + flatten spec (shapes/slices).
00337 |     model.eval()
00338 |     with torch.no_grad():
00339 |         base_out = model(_clone_batch(batch))
00340 |     base_flat, spec = flatten_graph2mat_predictions(base_out, output_keys=output_keys)
00341 |     base_predictions = unflatten_graph2mat_prediction_vector(base_flat, spec)
00342 | 
00343 |     def forward(positions_tensor: torch.Tensor) -> torch.Tensor:
00344 |         flat, forward_spec = graph2mat_forward_labels(
00345 |             model, batch, positions_tensor, output_keys=output_keys
00346 |         )
00347 |         if forward_spec.n_outputs != spec.n_outputs:
00348 |             raise Graph2MatAutogradDerivativeError(
00349 |                 "Differentiable forward produced a different number of outputs "
00350 |                 f"({forward_spec.n_outputs}) than the base forward ({spec.n_outputs})."
00351 |             )
00352 |         return flat
00353 | 
00354 |     # auto now resolves to the forward-mode JVP route: it computes only the
00355 |     # requested atom columns instead of the full reverse-mode jacobian over all
00356 |     # ~1e4 Hamiltonian outputs (~4000x faster, verified numerically identical).
00357 |     resolved_method = "jvp_double_backward" if method == "auto" else method
00358 |     resolved_chunk: int | None = chunk_size
00359 |     all_atoms = tuple(range(n_atoms))
00360 |     resolved_target_atoms = tuple(all_atoms if target_atoms is None else target_atoms)
00361 | 
00362 |     if resolved_method == "jvp_double_backward":
00363 |         jacobian = _jacobian_jvp_double_backward(forward, positions, resolved_target_atoms)
00364 |         resolved_chunk = None
00365 |     elif resolved_method == "vmap_vjp_chunked":
00366 |         if resolved_chunk is None:
00367 |             resolved_chunk = DEFAULT_JACOBIAN_CHUNK_SIZE
00368 |         jacobian, _ = _jacobian_vmap_vjp_chunked(forward, positions, int(resolved_chunk))
00369 |     elif resolved_method == "jacrev":
00370 |         jacobian = torch.func.jacrev(forward, chunk_size=resolved_chunk)(
00371 |             positions.detach().clone()
00372 |         )
00373 |     elif resolved_method == "jacfwd":
00374 |         jacobian = torch.func.jacfwd(forward)(positions.detach().clone())
00375 |     elif resolved_method == "autograd_jacobian":
00376 |         jacobian = torch.autograd.functional.jacobian(
00377 |             forward, positions.detach().clone(), vectorize=True
00378 |         )
00379 |     else:  # pragma: no cover - guarded above
00380 |         raise Graph2MatAutogradDerivativeError(f"Unhandled jacobian method {resolved_method!r}.")
00381 | 
00382 |     expected_shape = (spec.n_outputs, n_atoms, 3)
00383 |     if tuple(jacobian.shape) != expected_shape:
00384 |         raise Graph2MatAutogradDerivativeError(
00385 |             f"Jacobian has shape {tuple(jacobian.shape)}, expected {expected_shape}."
00386 |         )
00387 | 
00388 |     return PositionJacobianResult(
00389 |         jacobian=jacobian,
00390 |         spec=spec,
00391 |         base_predictions=base_predictions,
00392 |         method=resolved_method,
00393 |         chunk_size=int(resolved_chunk) if resolved_chunk is not None else None,
00394 |         n_atoms=n_atoms,
00395 |     )
00396 | 
00397 | 
00398 | def select_derivative_prediction_from_jacobian(
00399 |     jacobian: torch.Tensor,
00400 |     spec: PredictionFlattenSpec,
00401 |     atom_index: int,
00402 |     axis_index: int,
00403 |     *,
00404 |     change_of_basis: Any | None = None,
00405 | ) -> dict[str, torch.Tensor]:
00406 |     """Rebuild ``d labels / d R_atom,axis`` as a predictions-shaped dict.
00407 | 
00408 |     ``axis_index`` refers to the *physical cartesian* axis (the frame of the
00409 |     SIESTA/fdf structure and of the finite-displacement stencils).
00410 | 
00411 |     Graph2Mat batches do NOT store positions in that frame: ``_sanitize_data``
00412 |     applies ``cartesian_to_basis`` (the e3nn spherical-harmonics change of
00413 |     basis, ``p_batch = C @ p_cart``), so the jacobian columns are derivatives
00414 |     w.r.t. *batch-frame* coordinates. By the chain rule, the derivative along
00415 |     cartesian axis ``a`` is the contraction ``J[:, atom, :] @ C[:, a]``.
00416 | 
00417 |     Pass ``change_of_basis = data_processor.basis_table.change_of_basis`` for
00418 |     physically meaningful cartesian derivatives; ``None`` selects the raw
00419 |     batch-frame column (only correct when the change of basis is the
00420 |     identity, e.g. fake models in unit tests).
00421 |     """
00422 | 
00423 |     if jacobian.ndim != 3 or jacobian.shape[2] != 3:
00424 |         raise Graph2MatAutogradDerivativeError(
00425 |             f"Jacobian must have shape [n_outputs, n_atoms, 3], got {tuple(jacobian.shape)}."
00426 |         )
00427 |     n_atoms = int(jacobian.shape[1])
00428 |     if not (0 <= int(atom_index) < n_atoms):
00429 |         raise Graph2MatAutogradDerivativeError(
00430 |             f"atom_index {atom_index} is outside the structure with {n_atoms} atoms."
00431 |         )
00432 |     if int(axis_index) not in (0, 1, 2):
00433 |         raise Graph2MatAutogradDerivativeError(f"axis_index must be 0, 1 or 2, got {axis_index}.")
00434 |     if change_of_basis is None:
00435 |         column = jacobian[:, int(atom_index), int(axis_index)]
00436 |     else:
00437 |         cob = torch.as_tensor(change_of_basis, dtype=jacobian.dtype, device=jacobian.device)
00438 |         if tuple(cob.shape) != (3, 3):
00439 |             raise Graph2MatAutogradDerivativeError(
00440 |                 f"change_of_basis must be a 3x3 matrix, got {tuple(cob.shape)}."
00441 |             )
00442 |         direction = cob[:, int(axis_index)]
00443 |         column = jacobian[:, int(atom_index), :] @ direction
00444 |     return unflatten_graph2mat_prediction_vector(column, spec)
00445 | 
00446 | 
00447 | def translation_sum_rule_metrics(jacobian: torch.Tensor) -> dict[str, float]:
00448 |     """Global-translation invariance: sum_I dH/dR_{I,alpha} should vanish.
00449 | 
00450 |     Frame-independent (summing over atoms commutes with the e3nn change of
00451 |     basis), so it can be evaluated directly on the batch-frame jacobian.
00452 |     """
00453 |     if jacobian.ndim != 3 or jacobian.shape[2] != 3:
00454 |         raise Graph2MatAutogradDerivativeError(
00455 |             f"Jacobian must have shape [n_outputs, n_atoms, 3], got {tuple(jacobian.shape)}."
00456 |         )
00457 |     translation = jacobian.sum(dim=1)  # [n_outputs, 3]
00458 |     per_atom_norm = torch.linalg.norm(jacobian.reshape(jacobian.shape[0], -1))
00459 |     residual_norm = float(torch.linalg.norm(translation))
00460 |     return {
00461 |         "translation_residual_max_abs": float(translation.abs().max()),
00462 |         "translation_residual_frobenius": residual_norm,
00463 |         "translation_residual_relative": (
00464 |             residual_norm / float(per_atom_norm) if float(per_atom_norm) > 0 else float("nan")
00465 |         ),
00466 |     }
00467 | 
00468 | 
00469 | def supercell_order_from_sisl_matrix(sisl_matrix: Any) -> list[tuple[int, int, int]] | None:
00470 |     """R-vector ordering of a sisl matrix's supercell columns (or None)."""
00471 |     geometry = getattr(sisl_matrix, "geometry", None)
00472 |     lattice = getattr(geometry, "lattice", None) or getattr(geometry, "sc", None)
00473 |     sc_off = getattr(lattice, "sc_off", None)
00474 |     if sc_off is None:
00475 |         return None
00476 |     return [tuple(int(x) for x in vector) for vector in sc_off]
00477 | 
00478 | 
00479 | def derivative_prediction_to_sparse_matrices(
00480 |     data_processor: Any,
00481 |     batch: Any,
00482 |     derivative_prediction: dict[str, torch.Tensor],
00483 |     *,
00484 |     threshold: float | None = None,
00485 |     supercell_orders: list[list[tuple[int, int, int]] | None] | None = None,
00486 | ) -> list[Any]:
00487 |     """Convert derivative labels to sparse matrices via the existing mapping.
00488 | 
00489 |     ``supercell_orders``, when passed as an empty list, is filled with one
00490 |     R-vector ordering (sisl ``sc_off``) per returned matrix, for real-space
00491 |     blockwise hermiticity checks on the rectangular supercell layout.
00492 | 
00493 |     Reuses ``data_processor.yield_from_batch`` so the orbital/block mapping and
00494 |     the symmetric-edge accounting are exactly the ones used for normal
00495 |     Graph2Mat predictions. Two derivative-specific adjustments:
00496 | 
00497 |     - ``sub_point_matrix`` is forced off for the conversion: ``labels_to``
00498 |       adds the constant per-species point matrix back to node labels, and a
00499 |       constant has zero derivative, so adding it would corrupt dH/dR.
00500 |     - ``threshold`` defaults to ``None`` (keep every entry) instead of the
00501 |       1e-8 used for absolute Hamiltonians, because derivative magnitudes are
00502 |       not on the same scale as H.
00503 | 
00504 |     Returns one ``scipy.sparse.csr_matrix`` per structure in the batch, in the
00505 |     same (no, no * n_supercells) layout that ``ML_prediction.HSX`` files load
00506 |     into via ``sisl`` + ``tocsr(0)``.
00507 |     """
00508 | 
00509 |     predictions = {
00510 |         key: value.detach().to("cpu") for key, value in derivative_prediction.items()
00511 |     }
00512 |     conversion_processor = data_processor
00513 |     if getattr(data_processor, "sub_point_matrix", False):
00514 |         conversion_processor = data_processor.copy(sub_point_matrix=False)
00515 | 
00516 |     matrices = []
00517 |     for example in conversion_processor.yield_from_batch(
00518 |         batch, predictions=predictions, as_matrix=False
00519 |     ):
00520 |         sisl_matrix = conversion_processor.labels_to(
00521 |             conversion_processor.default_out_format,
00522 |             data=example,
00523 |             threshold=threshold,
00524 |         )
00525 |         csr = sisl_matrix.tocsr(0) if hasattr(sisl_matrix, "tocsr") else sisl_matrix
00526 |         matrices.append(csr.tocsr())
00527 |         if supercell_orders is not None:
00528 |             supercell_orders.append(supercell_order_from_sisl_matrix(sisl_matrix))
00529 |     return matrices
```

## `tests/test_graph2mat_autograd_derivatives.py`

SHA-256: `bc9f4623768f5a4b1011463956805783ef7e5500b54458d6f932d72d7535aa9b`

```py
00001 | from __future__ import annotations
00002 | 
00003 | import json
00004 | import os
00005 | import sys
00006 | import unittest
00007 | from pathlib import Path
00008 | 
00009 | import numpy as np
00010 | import pytest
00011 | import torch
00012 | 
00013 | 
00014 | REPO_ROOT = Path(__file__).resolve().parents[1]
00015 | SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
00016 | if str(SCRIPTS_DIR) not in sys.path:
00017 |     sys.path.insert(0, str(SCRIPTS_DIR))
00018 | TORCH_COMPAT_DIR = REPO_ROOT / "scripts" / "torch_serialization_compat"
00019 | if str(TORCH_COMPAT_DIR) not in sys.path:
00020 |     sys.path.insert(0, str(TORCH_COMPAT_DIR))
00021 | 
00022 | from graph2mat_autograd_derivatives import (  # noqa: E402
00023 |     Graph2MatAutogradDerivativeError,
00024 |     compute_graph2mat_position_jacobian,
00025 |     flatten_graph2mat_predictions,
00026 |     graph2mat_forward_labels,
00027 |     require_single_structure_batch,
00028 |     select_derivative_prediction_from_jacobian,
00029 |     translation_sum_rule_metrics,
00030 |     unflatten_graph2mat_prediction_vector,
00031 | )
00032 | from hamiltonian_derivative_stencil import (  # noqa: E402
00033 |     sparse_blockwise_hermiticity_defect,
00034 | )
00035 | from run_graph2mat_autograd_derivative_predictions import (  # noqa: E402
00036 |     _append_missing_structure_rows,
00037 | )
00038 | 
00039 | 
00040 | N_ATOMS = 4
00041 | N_NODE_OUTPUTS = 6
00042 | N_EDGE_OUTPUTS = 10
00043 | 
00044 | 
00045 | class FakeBatch(dict):
00046 |     """Minimal stand-in for a torch_geometric batch (dict + clone)."""
00047 | 
00048 |     def clone(self) -> "FakeBatch":
00049 |         return FakeBatch(self)
00050 | 
00051 | 
00052 | def make_batch(n_atoms: int = N_ATOMS, num_graphs: int | None = None) -> FakeBatch:
00053 |     generator = torch.Generator().manual_seed(7)
00054 |     batch = FakeBatch(
00055 |         positions=torch.randn(n_atoms, 3, generator=generator, dtype=torch.float64)
00056 |     )
00057 |     if num_graphs is not None:
00058 |         batch["ptr"] = torch.arange(num_graphs + 1) * n_atoms
00059 |     return batch
00060 | 
00061 | 
00062 | class LinearFakeModel(torch.nn.Module):
00063 |     """node_labels = A @ p_flat, edge_labels = B @ p_flat (analytic jacobian)."""
00064 | 
00065 |     def __init__(self) -> None:
00066 |         super().__init__()
00067 |         generator = torch.Generator().manual_seed(11)
00068 |         self.A = torch.randn(
00069 |             N_NODE_OUTPUTS, N_ATOMS * 3, generator=generator, dtype=torch.float64
00070 |         )
00071 |         self.B = torch.randn(
00072 |             N_EDGE_OUTPUTS, N_ATOMS * 3, generator=generator, dtype=torch.float64
00073 |         )
00074 | 
00075 |     def forward(self, data) -> dict[str, torch.Tensor]:
00076 |         flat = data["positions"].reshape(-1)
00077 |         return {"node_labels": self.A @ flat, "edge_labels": self.B @ flat}
00078 | 
00079 |     def analytic_jacobian(self) -> torch.Tensor:
00080 |         return torch.cat([self.A, self.B]).reshape(-1, N_ATOMS, 3)
00081 | 
00082 | 
00083 | class NonlinearFakeModel(torch.nn.Module):
00084 |     """node_labels = sum_axis(p^2) per atom, edge_labels = sin(p).flatten()."""
00085 | 
00086 |     def forward(self, data) -> dict[str, torch.Tensor]:
00087 |         positions = data["positions"]
00088 |         return {
00089 |             "node_labels": positions.pow(2).sum(dim=1),
00090 |             "edge_labels": torch.sin(positions).reshape(-1),
00091 |         }
00092 | 
00093 |     @staticmethod
00094 |     def analytic_jacobian(positions: torch.Tensor) -> torch.Tensor:
00095 |         n_atoms = positions.shape[0]
00096 |         n_outputs = n_atoms + n_atoms * 3
00097 |         jacobian = torch.zeros(n_outputs, n_atoms, 3, dtype=positions.dtype)
00098 |         for atom in range(n_atoms):
00099 |             jacobian[atom, atom, :] = 2.0 * positions[atom]
00100 |         for atom in range(n_atoms):
00101 |             for axis in range(3):
00102 |                 out_index = n_atoms + atom * 3 + axis
00103 |                 jacobian[out_index, atom, axis] = torch.cos(positions[atom, axis])
00104 |         return jacobian
00105 | 
00106 | 
00107 | class DetachedFakeModel(torch.nn.Module):
00108 |     def forward(self, data) -> dict[str, torch.Tensor]:
00109 |         positions = data["positions"].detach()
00110 |         return {
00111 |             "node_labels": positions.sum(dim=1),
00112 |             "edge_labels": positions.reshape(-1),
00113 |         }
00114 | 
00115 | 
00116 | class MultiComponentFakeModel(torch.nn.Module):
00117 |     """2D labels (n_matrix_components > 1) to exercise flatten/unflatten shapes."""
00118 | 
00119 |     def forward(self, data) -> dict[str, torch.Tensor]:
00120 |         positions = data["positions"]
00121 |         return {
00122 |             "node_labels": torch.stack([positions.sum(dim=1), positions.prod(dim=1)], dim=1),
00123 |             "edge_labels": positions.reshape(-1, 3),
00124 |         }
00125 | 
00126 | 
00127 | class FlattenSpecTests(unittest.TestCase):
00128 |     def test_flatten_and_unflatten_roundtrip(self) -> None:
00129 |         batch = make_batch()
00130 |         model = MultiComponentFakeModel()
00131 |         out = model(batch)
00132 |         flat, spec = flatten_graph2mat_predictions(out)
00133 | 
00134 |         self.assertEqual(flat.numel(), out["node_labels"].numel() + out["edge_labels"].numel())
00135 |         self.assertEqual(spec.n_outputs, flat.numel())
00136 |         rebuilt = unflatten_graph2mat_prediction_vector(flat, spec)
00137 |         self.assertEqual(tuple(rebuilt["node_labels"].shape), tuple(out["node_labels"].shape))
00138 |         self.assertEqual(tuple(rebuilt["edge_labels"].shape), tuple(out["edge_labels"].shape))
00139 |         torch.testing.assert_close(rebuilt["node_labels"], out["node_labels"])
00140 |         torch.testing.assert_close(rebuilt["edge_labels"], out["edge_labels"])
00141 | 
00142 |     def test_unflatten_rejects_wrong_size(self) -> None:
00143 |         batch = make_batch()
00144 |         out = LinearFakeModel()(batch)
00145 |         _, spec = flatten_graph2mat_predictions(out)
00146 |         with self.assertRaises(Graph2MatAutogradDerivativeError):
00147 |             unflatten_graph2mat_prediction_vector(torch.zeros(spec.n_outputs + 1), spec)
00148 | 
00149 |     def test_missing_output_key_is_rejected(self) -> None:
00150 |         with self.assertRaises(Graph2MatAutogradDerivativeError):
00151 |             flatten_graph2mat_predictions({"node_labels": torch.zeros(3)})
00152 | 
00153 | 
00154 | class ForwardLabelsTests(unittest.TestCase):
00155 |     def test_forward_keeps_gradient_connection(self) -> None:
00156 |         batch = make_batch()
00157 |         model = LinearFakeModel()
00158 |         positions = batch["positions"].detach().clone().requires_grad_(True)
00159 |         flat, spec = graph2mat_forward_labels(model, batch, positions)
00160 | 
00161 |         self.assertTrue(flat.requires_grad)
00162 |         self.assertEqual(spec.n_outputs, N_NODE_OUTPUTS + N_EDGE_OUTPUTS)
00163 |         # The original batch positions must not be mutated by the closure.
00164 |         self.assertFalse(batch["positions"].requires_grad)
00165 | 
00166 |     def test_single_structure_guard(self) -> None:
00167 |         batch = make_batch(num_graphs=2)
00168 |         with self.assertRaises(Graph2MatAutogradDerivativeError):
00169 |             require_single_structure_batch(batch)
00170 |         with self.assertRaises(Graph2MatAutogradDerivativeError):
00171 |             compute_graph2mat_position_jacobian(LinearFakeModel(), batch)
00172 | 
00173 | 
00174 | class PositionJacobianTests(unittest.TestCase):
00175 |     def test_linear_model_matches_analytic_jacobian(self) -> None:
00176 |         batch = make_batch()
00177 |         model = LinearFakeModel()
00178 |         result = compute_graph2mat_position_jacobian(
00179 |             model, batch, method="vmap_vjp_chunked", chunk_size=5
00180 |         )
00181 | 
00182 |         expected = model.analytic_jacobian()
00183 |         self.assertEqual(
00184 |             tuple(result.jacobian.shape), (N_NODE_OUTPUTS + N_EDGE_OUTPUTS, N_ATOMS, 3)
00185 |         )
00186 |         torch.testing.assert_close(result.jacobian, expected, rtol=1e-9, atol=1e-12)
00187 |         self.assertEqual(result.method, "vmap_vjp_chunked")
00188 |         self.assertEqual(result.chunk_size, 5)
00189 |         self.assertGreater(result.jacobian.abs().max().item(), 0.0)
00190 | 
00191 |     def test_nonlinear_model_matches_analytic_jacobian(self) -> None:
00192 |         batch = make_batch()
00193 |         model = NonlinearFakeModel()
00194 |         result = compute_graph2mat_position_jacobian(model, batch, method="auto")
00195 | 
00196 |         expected = model.analytic_jacobian(batch["positions"])
00197 |         torch.testing.assert_close(result.jacobian, expected, rtol=1e-9, atol=1e-12)
00198 |         torch.testing.assert_close(
00199 |             result.base_predictions["node_labels"],
00200 |             batch["positions"].pow(2).sum(dim=1),
00201 |         )
00202 | 
00203 |     def test_methods_agree_on_small_case(self) -> None:
00204 |         batch = make_batch()
00205 |         model = NonlinearFakeModel()
00206 |         reference = compute_graph2mat_position_jacobian(
00207 |             model, batch, method="vmap_vjp_chunked", chunk_size=3
00208 |         ).jacobian
00209 | 
00210 |         for method in ("jacrev", "jacfwd", "autograd_jacobian"):
00211 |             with self.subTest(method=method):
00212 |                 jacobian = compute_graph2mat_position_jacobian(
00213 |                     model, batch, method=method
00214 |                 ).jacobian
00215 |                 torch.testing.assert_close(jacobian, reference, rtol=1e-8, atol=1e-10)
00216 | 
00217 |     def test_detached_model_fails_loudly(self) -> None:
00218 |         batch = make_batch()
00219 |         with self.assertRaises(Graph2MatAutogradDerivativeError):
00220 |             compute_graph2mat_position_jacobian(DetachedFakeModel(), batch)
00221 | 
00222 |     def test_unknown_method_is_rejected(self) -> None:
00223 |         with self.assertRaises(Graph2MatAutogradDerivativeError):
00224 |             compute_graph2mat_position_jacobian(
00225 |                 LinearFakeModel(), make_batch(), method="finite_difference"
00226 |             )
00227 | 
00228 | 
00229 | class SelectDerivativeTests(unittest.TestCase):
00230 |     def test_selected_column_shapes_and_values(self) -> None:
00231 |         batch = make_batch()
00232 |         model = LinearFakeModel()
00233 |         result = compute_graph2mat_position_jacobian(model, batch)
00234 | 
00235 |         atom_index, axis_index = 2, 1
00236 |         derivative = select_derivative_prediction_from_jacobian(
00237 |             result.jacobian, result.spec, atom_index, axis_index
00238 |         )
00239 | 
00240 |         self.assertEqual(tuple(derivative["node_labels"].shape), (N_NODE_OUTPUTS,))
00241 |         self.assertEqual(tuple(derivative["edge_labels"].shape), (N_EDGE_OUTPUTS,))
00242 |         flat_index = atom_index * 3 + axis_index
00243 |         torch.testing.assert_close(derivative["node_labels"], model.A[:, flat_index])
00244 |         torch.testing.assert_close(derivative["edge_labels"], model.B[:, flat_index])
00245 | 
00246 |     def test_selected_column_matches_numeric_finite_difference(self) -> None:
00247 |         # Numeric FD is only a sanity check of the test itself, not the
00248 |         # scientific route.
00249 |         batch = make_batch()
00250 |         model = NonlinearFakeModel()
00251 |         result = compute_graph2mat_position_jacobian(model, batch)
00252 | 
00253 |         atom_index, axis_index = 1, 2
00254 |         delta = 1e-6
00255 |         plus = batch.clone()
00256 |         minus = batch.clone()
00257 |         plus["positions"] = batch["positions"].clone()
00258 |         minus["positions"] = batch["positions"].clone()
00259 |         plus["positions"][atom_index, axis_index] += delta
00260 |         minus["positions"][atom_index, axis_index] -= delta
00261 |         numeric = {
00262 |             key: (model(plus)[key] - model(minus)[key]) / (2 * delta)
00263 |             for key in ("node_labels", "edge_labels")
00264 |         }
00265 | 
00266 |         derivative = select_derivative_prediction_from_jacobian(
00267 |             result.jacobian, result.spec, atom_index, axis_index
00268 |         )
00269 |         torch.testing.assert_close(
00270 |             derivative["node_labels"], numeric["node_labels"], rtol=1e-6, atol=1e-8
00271 |         )
00272 |         torch.testing.assert_close(
00273 |             derivative["edge_labels"], numeric["edge_labels"], rtol=1e-6, atol=1e-8
00274 |         )
00275 | 
00276 |     def test_change_of_basis_contraction_gives_cartesian_directional_derivative(self) -> None:
00277 |         # Graph2Mat stores batch positions as p_batch = C @ p_cart (e3nn change
00278 |         # of basis). The cartesian derivative along axis a must therefore be
00279 |         # J[:, atom, :] @ C[:, a], not the raw batch-frame column.
00280 |         batch = make_batch()
00281 |         model = NonlinearFakeModel()
00282 |         result = compute_graph2mat_position_jacobian(model, batch)
00283 | 
00284 |         # e3nn cartesian -> spherical-harmonics convention: (x, y, z) -> (y, z, x).
00285 |         cob = torch.tensor(
00286 |             [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=torch.float64
00287 |         )
00288 |         atom_index, axis_index = 1, 0
00289 |         derivative = select_derivative_prediction_from_jacobian(
00290 |             result.jacobian, result.spec, atom_index, axis_index, change_of_basis=cob
00291 |         )
00292 | 
00293 |         # Numeric check: displace the *cartesian* position, map through C, and
00294 |         # finite-difference the model on batch-frame inputs.
00295 |         delta = 1e-6
00296 |         direction = cob[:, axis_index]
00297 |         plus = batch.clone()
00298 |         minus = batch.clone()
00299 |         plus["positions"] = batch["positions"].clone()
00300 |         minus["positions"] = batch["positions"].clone()
00301 |         plus["positions"][atom_index] += delta * direction
00302 |         minus["positions"][atom_index] -= delta * direction
00303 |         for key in ("node_labels", "edge_labels"):
00304 |             numeric = (model(plus)[key] - model(minus)[key]) / (2 * delta)
00305 |             torch.testing.assert_close(derivative[key], numeric, rtol=1e-6, atol=1e-8)
00306 | 
00307 |     def test_change_of_basis_identity_matches_raw_column(self) -> None:
00308 |         batch = make_batch()
00309 |         result = compute_graph2mat_position_jacobian(LinearFakeModel(), batch)
00310 |         raw = select_derivative_prediction_from_jacobian(result.jacobian, result.spec, 0, 1)
00311 |         identity = select_derivative_prediction_from_jacobian(
00312 |             result.jacobian, result.spec, 0, 1, change_of_basis=torch.eye(3, dtype=torch.float64)
00313 |         )
00314 |         torch.testing.assert_close(raw["node_labels"], identity["node_labels"])
00315 |         torch.testing.assert_close(raw["edge_labels"], identity["edge_labels"])
00316 | 
00317 |     def test_out_of_range_selection_is_rejected(self) -> None:
00318 |         batch = make_batch()
00319 |         result = compute_graph2mat_position_jacobian(LinearFakeModel(), batch)
00320 |         with self.assertRaises(Graph2MatAutogradDerivativeError):
00321 |             select_derivative_prediction_from_jacobian(
00322 |                 result.jacobian, result.spec, N_ATOMS, 0
00323 |             )
00324 |         with self.assertRaises(Graph2MatAutogradDerivativeError):
00325 |             select_derivative_prediction_from_jacobian(result.jacobian, result.spec, 0, 3)
00326 | 
00327 | 
00328 | class PhysicalInvarianceTests(unittest.TestCase):
00329 |     """Fase 4 (audit): translation sum rule + real-space blockwise hermiticity."""
00330 | 
00331 |     def test_translation_sum_rule_zero_for_invariant_model(self) -> None:
00332 |         # Columns built so per-output rows sum to zero over atoms: the exact
00333 |         # signature of a translation-invariant model.
00334 |         jacobian = torch.zeros(4, 3, 3, dtype=torch.float64)
00335 |         jacobian[:, 0, :] = torch.randn(4, 3, dtype=torch.float64)
00336 |         jacobian[:, 1, :] = torch.randn(4, 3, dtype=torch.float64)
00337 |         jacobian[:, 2, :] = -(jacobian[:, 0, :] + jacobian[:, 1, :])
00338 |         metrics = translation_sum_rule_metrics(jacobian)
00339 |         self.assertLess(metrics["translation_residual_max_abs"], 1e-12)
00340 |         self.assertLess(metrics["translation_residual_relative"], 1e-12)
00341 | 
00342 |     def test_translation_sum_rule_flags_violation(self) -> None:
00343 |         jacobian = torch.ones(2, 2, 3, dtype=torch.float64)
00344 |         metrics = translation_sum_rule_metrics(jacobian)
00345 |         self.assertGreater(metrics["translation_residual_max_abs"], 1.0)
00346 |         self.assertGreater(metrics["translation_residual_relative"], 0.5)
00347 | 
00348 |     def test_translation_sum_rule_rejects_bad_shape(self) -> None:
00349 |         with self.assertRaises(Graph2MatAutogradDerivativeError):
00350 |             translation_sum_rule_metrics(torch.zeros(4, 3))
00351 | 
00352 |     def test_blockwise_hermiticity_zero_for_hermitian_layout(self) -> None:
00353 |         from scipy import sparse as sp
00354 | 
00355 |         rng = np.random.default_rng(0)
00356 |         n = 3
00357 |         order = [(0, 0, 0), (1, 0, 0), (-1, 0, 0)]
00358 |         d0 = rng.normal(size=(n, n))
00359 |         d0 = d0 + d0.T  # onsite block hermitian
00360 |         d_plus = rng.normal(size=(n, n))
00361 |         matrix = sp.csr_matrix(np.hstack([d0, d_plus, d_plus.T]))  # D(-R) = D(R)^T
00362 |         defect = sparse_blockwise_hermiticity_defect(matrix, order)
00363 |         self.assertAlmostEqual(defect, 0.0, places=12)
00364 | 
00365 |     def test_blockwise_hermiticity_flags_broken_block(self) -> None:
00366 |         from scipy import sparse as sp
00367 | 
00368 |         rng = np.random.default_rng(1)
00369 |         n = 3
00370 |         order = [(0, 0, 0), (1, 0, 0), (-1, 0, 0)]
00371 |         d0 = rng.normal(size=(n, n))
00372 |         d0 = d0 + d0.T
00373 |         d_plus = rng.normal(size=(n, n))
00374 |         matrix = sp.csr_matrix(np.hstack([d0, d_plus, rng.normal(size=(n, n))]))
00375 |         self.assertGreater(sparse_blockwise_hermiticity_defect(matrix, order), 0.1)
00376 | 
00377 |     def test_blockwise_hermiticity_nan_without_layout(self) -> None:
00378 |         from scipy import sparse as sp
00379 | 
00380 |         matrix = sp.csr_matrix(np.ones((2, 4)))
00381 |         self.assertTrue(np.isnan(sparse_blockwise_hermiticity_defect(matrix, [])))
00382 | 
00383 | 
00384 | class SparseConversionTests(unittest.TestCase):
00385 |     def test_sub_point_matrix_is_disabled_for_derivative_conversion(self) -> None:
00386 |         from graph2mat_autograd_derivatives import derivative_prediction_to_sparse_matrices
00387 | 
00388 |         recorded: dict[str, object] = {}
00389 | 
00390 |         class FakeProcessor:
00391 |             sub_point_matrix = True
00392 |             default_out_format = "scipy_csr"
00393 | 
00394 |             def copy(self, **kwargs):
00395 |                 clone = FakeProcessor()
00396 |                 for key, value in kwargs.items():
00397 |                     setattr(clone, key, value)
00398 |                 recorded["copy_kwargs"] = kwargs
00399 |                 return clone
00400 | 
00401 |             def yield_from_batch(self, batch, predictions=None, as_matrix=False):
00402 |                 recorded["predictions"] = predictions
00403 |                 yield "example"
00404 | 
00405 |             def labels_to(self, out_format, data=None, threshold=None):
00406 |                 recorded["out_format"] = out_format
00407 |                 recorded["threshold"] = threshold
00408 |                 from scipy import sparse
00409 | 
00410 |                 return sparse.csr_matrix(np.eye(2))
00411 | 
00412 |         derivative = {
00413 |             "node_labels": torch.ones(3, requires_grad=True),
00414 |             "edge_labels": torch.ones(4, requires_grad=True),
00415 |         }
00416 |         matrices = derivative_prediction_to_sparse_matrices(
00417 |             FakeProcessor(), FakeBatch(), derivative
00418 |         )
00419 | 
00420 |         self.assertEqual(len(matrices), 1)
00421 |         self.assertEqual(recorded["copy_kwargs"], {"sub_point_matrix": False})
00422 |         self.assertIsNone(recorded["threshold"])
00423 |         self.assertFalse(recorded["predictions"]["node_labels"].requires_grad)
00424 | 
00425 | 
00426 | # --------------------------------------------------------------------------- #
00427 | # Real-checkpoint smoke: autograd jacobian vs finite-difference of the model
00428 | # itself (never SIESTA). Skips cleanly when no checkpoint is available.
00429 | # --------------------------------------------------------------------------- #
00430 | _SMOKE_SWEEP_ROOT = (
00431 |     REPO_ROOT / "Comparison" / "results" / "e2e_smoke_12snap_20ep" / "e2e_smoke_12snap_20ep" / "sweep"
00432 | )
00433 | _DEFAULT_SMOKE_CKPT = (
00434 |     _SMOKE_SWEEP_ROOT / "graph2mat" / "graphene_w90_scale_iid12" / "G2M-E2E12-20EP"
00435 |     / "graph2mat" / "training" / "lightning_logs" / "my_first_model" / "version_0"
00436 |     / "checkpoints" / "best-160.ckpt"
00437 | )
00438 | _DEFAULT_SMOKE_STRUCTURE = (
00439 |     _SMOKE_SWEEP_ROOT / "derivative_workflows" / "graphene_w90_scale_iid12"
00440 |     / "structures" / "md_11_base"
00441 | )
00442 | 
00443 | 
00444 | def _flat_labels(predictions: dict[str, torch.Tensor]) -> torch.Tensor:
00445 |     return torch.cat(
00446 |         [predictions[key].detach().reshape(-1) for key in ("node_labels", "edge_labels")]
00447 |     ).to(torch.float64)
00448 | 
00449 | 
00450 | @pytest.mark.slow
00451 | def test_autograd_jacobian_matches_model_finite_difference_real_checkpoint(tmp_path):
00452 |     """dH_pred/dR (autograd, 1 atom, 1 axis) vs central FD of the model itself."""
00453 |     checkpoint = Path(os.environ.get("G2M_AUTOGRAD_SMOKE_CKPT") or _DEFAULT_SMOKE_CKPT)
00454 |     structure_dir = Path(
00455 |         os.environ.get("G2M_AUTOGRAD_SMOKE_STRUCTURE") or _DEFAULT_SMOKE_STRUCTURE
00456 |     )
00457 |     if not checkpoint.is_file():
00458 |         pytest.skip(f"No Graph2Mat checkpoint available: {checkpoint}")
00459 |     if not (structure_dir / "RUN.fdf").is_file():
00460 |         pytest.skip(f"No base structure RUN.fdf available: {structure_dir}")
00461 |     basis_glob = os.environ.get("G2M_AUTOGRAD_SMOKE_BASIS") or str(structure_dir / "*.ion.xml")
00462 |     import glob as _glob
00463 | 
00464 |     if not _glob.glob(basis_glob):
00465 |         pytest.skip(f"No basis .ion.xml files match: {basis_glob}")
00466 |     try:
00467 |         from torch_safe_globals import allow_graph2mat_checkpoint_globals
00468 | 
00469 |         # Importing graph2mat/mace already torch.load()s data files, so the
00470 |         # safe globals must be registered before the imports.
00471 |         allow_graph2mat_checkpoint_globals()
00472 | 
00473 |         from graph2mat.tools.lightning import MatrixDataModule
00474 |         from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel
00475 | 
00476 |         from predict_model_on_dataset import (
00477 |             checkpoint_training_dir,
00478 |             normalize_pattern_for_workdir,
00479 |         )
00480 |     except ImportError as exc:  # pragma: no cover - environment-dependent
00481 |         pytest.skip(f"graph2mat stack not importable: {exc}")
00482 |     run_cwd = checkpoint_training_dir(checkpoint)
00483 |     source_cwd = Path.cwd()
00484 |     runs_json = tmp_path / "smoke_runs.json"
00485 |     runs_json.write_text(
00486 |         json.dumps({"predict": [str(structure_dir / "RUN.fdf")]}), encoding="utf-8"
00487 |     )
00488 |     basis_files = normalize_pattern_for_workdir(
00489 |         basis_glob, source_cwd=source_cwd, target_cwd=run_cwd
00490 |     )
00491 |     try:
00492 |         os.chdir(run_cwd)
00493 |         model = LitMACEMatrixModel.load_from_checkpoint(
00494 |             str(checkpoint), map_location="cpu", weights_only=False
00495 |         )
00496 |         model.eval()
00497 |         datamodule = MatrixDataModule(
00498 |             out_matrix="hamiltonian",
00499 |             symmetric_matrix=True,
00500 |             sub_point_matrix=False,
00501 |             basis_files=basis_files,
00502 |             runs_json=str(runs_json),
00503 |             store_in_memory=True,
00504 |             batch_size=1,
00505 |             n_matrix_components=1,
00506 |             matrix_component_policy="h_only",
00507 |         )
00508 |         datamodule.setup("predict")
00509 |         batch = next(iter(datamodule.predict_dataloader()))
00510 |         cob = torch.as_tensor(
00511 |             datamodule.data_processor.basis_table.change_of_basis, dtype=torch.float64
00512 |         )
00513 |     finally:
00514 |         os.chdir(source_cwd)
00515 | 
00516 |     atom_index, axis_index = 0, 0
00517 |     result = compute_graph2mat_position_jacobian(model.model, batch)
00518 |     derivative = select_derivative_prediction_from_jacobian(
00519 |         result.jacobian, result.spec, atom_index, axis_index, change_of_basis=cob
00520 |     )
00521 |     autograd_flat = _flat_labels(derivative)
00522 |     assert autograd_flat.abs().max().item() > 0.0
00523 | 
00524 |     positions = batch["positions"]
00525 |     # Batch positions live in the e3nn frame (p_batch = C @ p_cart), so a
00526 |     # cartesian displacement along axis a maps to the direction C[:, a].
00527 |     direction = cob.to(positions.dtype)[:, axis_index]
00528 |     for delta in (0.003, 0.01):
00529 |         displaced = {}
00530 |         for sign in (1.0, -1.0):
00531 |             shifted = batch.clone()
00532 |             shifted_positions = positions.detach().clone()
00533 |             shifted_positions[atom_index] += sign * delta * direction
00534 |             shifted["positions"] = shifted_positions
00535 |             with torch.no_grad():
00536 |                 displaced[sign] = _flat_labels(model.model(shifted))
00537 |         fd_flat = (displaced[1.0] - displaced[-1.0]) / (2.0 * delta)
00538 | 
00539 |         cos = torch.dot(autograd_flat, fd_flat) / (
00540 |             autograd_flat.norm() * fd_flat.norm()
00541 |         )
00542 |         rel_frobenius = (autograd_flat - fd_flat).norm() / fd_flat.norm()
00543 |         mae = (autograd_flat - fd_flat).abs().mean()
00544 |         print(
00545 |             f"[autograd-smoke] delta={delta} cos={cos.item():.6f} "
00546 |             f"rel_frobenius={rel_frobenius.item():.3e} mae_eV_per_ang={mae.item():.3e}"
00547 |         )
00548 |         assert cos.item() >= 0.999, (
00549 |             f"autograd vs model finite-difference cosine {cos.item():.6f} < 0.999 "
00550 |             f"(delta={delta}, rel_frobenius={rel_frobenius.item():.3e})"
00551 |         )
00552 |         assert rel_frobenius.item() <= 0.25, (
00553 |             f"autograd vs model finite-difference relative Frobenius "
00554 |             f"{rel_frobenius.item():.3e} > 2.5e-1 (delta={delta})"
00555 |         )
00556 | 
00557 | 
00558 | class AutogradPredictionScriptTests(unittest.TestCase):
00559 |     def test_missing_dataloader_structure_becomes_failed_row(self) -> None:
00560 |         rows = []
00561 |         requests = {
00562 |             "base_0_base": {
00563 |                 "base_sample_id": "base_0",
00564 |                 "pairs": {(0, "x"): {0.01}},
00565 |             }
00566 |         }
00567 | 
00568 |         _append_missing_structure_rows(rows, requests, seen_structure_ids=set())
00569 | 
00570 |         self.assertEqual(len(rows), 1)
00571 |         self.assertEqual(rows[0]["status"], "error")
00572 |         self.assertEqual(rows[0]["base_structure_sample_id"], "base_0_base")
00573 |         self.assertEqual(rows[0]["error"], "missing_base_structure_from_graph2mat_dataloader")
00574 | 
00575 | 
00576 | if __name__ == "__main__":
00577 |     unittest.main()
```

## `tests/test_deeph_autograd_derivatives.py`

SHA-256: `ac4d70d2bcb67159ed988ab532ebabccc50f8f07e91d029512dbd3cc8eda2281`

```py
00001 | from __future__ import annotations
00002 | 
00003 | import argparse
00004 | import configparser
00005 | import json
00006 | import os
00007 | import shutil
00008 | import subprocess
00009 | import sys
00010 | import tempfile
00011 | import unittest
00012 | from pathlib import Path
00013 | from types import SimpleNamespace
00014 | from unittest import mock
00015 | 
00016 | import h5py
00017 | import numpy as np
00018 | from scipy import sparse
00019 | 
00020 | try:
00021 |     import pytest
00022 | except ImportError:  # pragma: no cover - unittest can still run the direct script test.
00023 |     pytest = None
00024 | 
00025 | 
00026 | REPO_ROOT = Path(__file__).resolve().parents[1]
00027 | SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
00028 | if str(SCRIPTS_DIR) not in sys.path:
00029 |     sys.path.insert(0, str(SCRIPTS_DIR))
00030 | 
00031 | import run_deeph_autograd_derivative_predictions as deeph_autograd  # noqa: E402
00032 | from deeph_config import default_deeph_paths  # noqa: E402
00033 | from hamiltonian_derivative_stencil import PREDICTED_DERIVATIVE_METHOD_AUTOGRAD_DEEPH  # noqa: E402
00034 | 
00035 | 
00036 | def write_json(path: Path, payload: dict) -> None:
00037 |     path.parent.mkdir(parents=True, exist_ok=True)
00038 |     path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
00039 | 
00040 | 
00041 | class DeepHAutogradDerivativeScriptTests(unittest.TestCase):
00042 |     def setUp(self) -> None:
00043 |         self.tmp = tempfile.TemporaryDirectory()
00044 |         self.root = Path(self.tmp.name)
00045 | 
00046 |     def tearDown(self) -> None:
00047 |         self.tmp.cleanup()
00048 | 
00049 |     def _write_stencil_fixture(self) -> Path:
00050 |         stencil_root = self.root / "stencil"
00051 |         base = stencil_root / "structures" / "base_0_base"
00052 |         plus = stencil_root / "structures" / "base_0_plus"
00053 |         base.mkdir(parents=True)
00054 |         plus.mkdir(parents=True)
00055 |         write_json(
00056 |             base / "metadata.json",
00057 |             {
00058 |                 "sample_id": "base_0",
00059 |                 "base_sample_id": "base_0",
00060 |                 "is_reference": True,
00061 |                 "sign": 0,
00062 |                 "split": "test",
00063 |             },
00064 |         )
00065 |         write_json(
00066 |             plus / "metadata.json",
00067 |             {
00068 |                 "sample_id": "base_0_plus",
00069 |                 "base_sample_id": "base_0",
00070 |                 "atom_index_zero_based": 0,
00071 |                 "axis": "z",
00072 |                 "axis_index": 2,
00073 |                 "delta_ang": 0.01,
00074 |                 "sign": 1,
00075 |                 "split": "test",
00076 |             },
00077 |         )
00078 |         return stencil_root
00079 | 
00080 |     def test_script_runs_autograd_flow_and_writes_direct_derivative_metadata(self) -> None:
00081 |         stencil_root = self._write_stencil_fixture()
00082 |         output_root = self.root / "predicted_derivatives"
00083 |         model_dir = self.root / "deeph_model"
00084 |         model_dir.mkdir()
00085 |         (model_dir / "config.ini").write_text("[graph]\nradius = 5.0\n[basic]\ndisable_cuda = True\ndevice = cpu\n", encoding="utf-8")
00086 |         bin_dir = self.root / "bin"
00087 |         bin_dir.mkdir()
00088 |         inference_cli = bin_dir / "deeph-inference"
00089 |         preprocess_cli = bin_dir / "deeph-preprocess"
00090 |         inference_cli.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
00091 |         preprocess_cli.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
00092 |         inference_cli.chmod(0o755)
00093 |         preprocess_cli.chmod(0o755)
00094 | 
00095 |         deeph_paths = default_deeph_paths(output_root.parent)
00096 |         processed_sample = deeph_paths.processed_dir / "base_0_base"
00097 |         config_checks: list[dict[str, object]] = []
00098 |         commands: list[str] = []
00099 | 
00100 |         def fake_references(_stencil_root: Path, *, structures: list[Path]) -> dict[str, Path]:
00101 |             return {structure.name: self.root / "siesta_refs" / structure.name for structure in structures}
00102 | 
00103 |         def fake_raw_mirror(*, references: dict[str, Path], raw_dir: Path) -> dict:
00104 |             raw_sample = raw_dir / "base_0_base"
00105 |             raw_sample.mkdir(parents=True, exist_ok=True)
00106 |             return {
00107 |                 "rows": [
00108 |                     {
00109 |                         "sample_id": "base_0_base",
00110 |                         "raw_dir": str(raw_sample),
00111 |                         "source_dir": str(references["base_0_base"]),
00112 |                     }
00113 |                 ]
00114 |             }
00115 | 
00116 |         def fake_run_command(command, *, cwd, env):
00117 |             commands.append(Path(command[0]).name)
00118 |             if len(command) >= 3 and command[1] == "-c" and "autograd_capability" in command[2]:
00119 |                 # Fase 1 capability preflight: emulate a real JVP backend.
00120 |                 capability = {
00121 |                     "available": True,
00122 |                     "implementation": "torch_forward_ad_jvp",
00123 |                     "output_schema": "hamiltonians_grad_pred_v2",
00124 |                 }
00125 |                 return {"command": command, "returncode": 0, "stdout": json.dumps(capability),
00126 |                         "stderr": "", "started_at": 1.0, "finished_at": 2.0}
00127 |             if Path(command[0]).name == "deeph-preprocess":
00128 |                 processed_sample.mkdir(parents=True, exist_ok=True)
00129 |                 (processed_sample / "lat.dat").write_text("1 0 0\n0 1 0\n0 0 1\n", encoding="utf-8")
00130 |                 return {"command": command, "returncode": 0, "stdout": "", "stderr": "", "started_at": 1.0, "finished_at": 2.0}
00131 | 
00132 |             config = configparser.ConfigParser()
00133 |             config.read(command[command.index("--config") + 1])
00134 |             work_dir = Path(config.get("basic", "work_dir"))
00135 |             config_checks.append(
00136 |                 {
00137 |                     "with_grad": config.getboolean("basic", "with_grad"),
00138 |                     "task": json.loads(config.get("basic", "task")),
00139 |                     "grad_atom_indices": json.loads(config.get("basic", "grad_atom_indices")),
00140 |                     "grad_axis_indices": json.loads(config.get("basic", "grad_axis_indices")),
00141 |                 }
00142 |             )
00143 |             work_dir.mkdir(parents=True, exist_ok=True)
00144 |             (work_dir / "hamiltonians_grad_pred.h5").write_bytes(b"grad-h5")
00145 |             (work_dir / "hamiltonians_pred.h5").write_bytes(b"pred-h5")
00146 |             return {"command": command, "returncode": 0, "stdout": "", "stderr": "", "started_at": 3.0, "finished_at": 4.0}
00147 | 
00148 |         def fake_reconstruct(*, output_path: Path, **_kwargs):
00149 |             sparse.save_npz(output_path, sparse.csr_matrix([[2.0]]))
00150 |             return {"kind": "fake_deeph_sparse_layout", "shape_rows": 1, "shape_cols": 1, "nnz": 1}
00151 | 
00152 |         adapter = SimpleNamespace(
00153 |             diagnostic_only=False,
00154 |             metric_fields=lambda: {
00155 |                 "deeph_raw_global_equivalence_proven": True,
00156 |                 "deeph_diagnostic_only": False,
00157 |                 "deeph_equivalence_status": "proven",
00158 |             },
00159 |             to_dict=lambda: {"diagnostic_only": False},
00160 |         )
00161 |         args = argparse.Namespace(
00162 |             stencil_root=stencil_root,
00163 |             output_root=output_root,
00164 |             model_dir=model_dir,
00165 |             deeph_command=str(inference_cli),
00166 |             python_executable=sys.executable,
00167 |             overwrite=True,
00168 |             skip_if_exists=True,
00169 |             base_sample_id=[],
00170 |             atoms=[],
00171 |             axes=[],
00172 |             max_base_structures=None,
00173 |             max_samples=None,
00174 |         )
00175 | 
00176 |         with mock.patch.object(deeph_autograd, "discover_siesta_reference_samples", side_effect=fake_references), mock.patch.object(
00177 |             deeph_autograd, "build_deeph_derivative_raw_mirror", side_effect=fake_raw_mirror
00178 |         ), mock.patch.object(deeph_autograd, "run_command", side_effect=fake_run_command), mock.patch.object(
00179 |             deeph_autograd, "reconstruct_deeph_sparse_layout_prediction", side_effect=fake_reconstruct
00180 |         ), mock.patch.object(
00181 |             deeph_autograd, "adapt_deeph_prediction_sample", return_value=adapter
00182 |         ):
00183 |             manifest = deeph_autograd.run_deeph_autograd_derivative_predictions(args)
00184 | 
00185 |         # 'python' is the Fase 1 capability preflight; it must run FIRST.
00186 |         self.assertEqual(commands, ["python", "deeph-preprocess", "deeph-inference"])
00187 |         self.assertEqual(config_checks, [{"with_grad": True, "task": [3], "grad_atom_indices": [0], "grad_axis_indices": [2]}])
00188 |         self.assertEqual(manifest["samples_failed"], 0)
00189 |         self.assertEqual(manifest["predicted_derivative_method"], PREDICTED_DERIVATIVE_METHOD_AUTOGRAD_DEEPH)
00190 |         self.assertEqual(manifest["deeph_prediction_method"], "autograd_vectorized")
00191 |         metadata_path = output_root / "base_0_base" / "dH_pred_atom0_axis2.json"
00192 |         metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
00193 |         self.assertTrue(metadata["with_grad"])
00194 |         self.assertIsNone(metadata["predicted_delta_ang"])
00195 |         self.assertTrue(metadata["deeph_raw_global_equivalence_proven"])
00196 |         self.assertFalse(metadata["deeph_diagnostic_only"])
00197 |         self.assertTrue((output_root / "base_0_base" / "dH_pred_atom0_axis2.npz").exists())
00198 | 
00199 | 
00200 | def _read_h5_blocks(path: Path) -> dict[str, np.ndarray]:
00201 |     with h5py.File(path, "r") as handle:
00202 |         return {key: np.asarray(handle[key][()]) for key in handle.keys()}
00203 | 
00204 | 
00205 | def _copy_sample_with_shift(source: Path, target: Path, *, atom: int, axis: int, delta: float) -> None:
00206 |     if target.exists():
00207 |         shutil.rmtree(target)
00208 |     shutil.copytree(source, target)
00209 |     positions_path = target / "site_positions.dat"
00210 |     positions = np.loadtxt(positions_path)
00211 |     shifted = np.array(positions, copy=True, ndmin=2)
00212 |     if shifted.shape[0] == 3:
00213 |         shifted[axis, atom] += delta
00214 |     elif shifted.shape[1] == 3:
00215 |         shifted[atom, axis] += delta
00216 |     else:
00217 |         raise AssertionError(f"Unsupported site_positions.dat shape: {shifted.shape}")
00218 |     np.savetxt(positions_path, shifted)
00219 |     for stale in ("rc.h5", "hamiltonians_pred.h5", "hamiltonians_grad_pred.h5", "rh_pred.h5"):
00220 |         path = target / stale
00221 |         if path.exists():
00222 |             path.unlink()
00223 | 
00224 | 
00225 | def _run_deeph_inference(*, deeph_cli: Path, config_path: Path) -> None:
00226 |     completed = subprocess.run(
00227 |         [str(deeph_cli), "--config", str(config_path)],
00228 |         cwd=str(deeph_cli.resolve(strict=False).parents[2]) if len(deeph_cli.resolve(strict=False).parents) > 2 else None,
00229 |         text=True,
00230 |         stdout=subprocess.PIPE,
00231 |         stderr=subprocess.PIPE,
00232 |         check=False,
00233 |     )
00234 |     if completed.returncode != 0:
00235 |         raise AssertionError(
00236 |             f"DeepH inference failed with {completed.returncode}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
00237 |         )
00238 | 
00239 | 
00240 | slow_mark = pytest.mark.slow if pytest is not None else (lambda func: func)
00241 | 
00242 | 
00243 | def fd_window_converged(rel_errors: list[float], *, noise_floor: float = 0.05) -> bool:
00244 |     """Correct finite-difference behaviour over a descending-delta sweep.
00245 | 
00246 |     ``rel_errors`` are relative errors ordered from the LARGEST delta to the
00247 |     smallest. A healthy FD check shows a window where shrinking delta reduces
00248 |     the error (truncation-dominated regime) before numerical cancellation
00249 |     saturates it; alternatively every delta may already sit at the noise
00250 |     floor. A sweep where shrinking delta only ever makes things worse (and
00251 |     errors are large) means the analytic derivative does not match.
00252 |     """
00253 |     if len(rel_errors) < 2:
00254 |         return min(rel_errors) < noise_floor
00255 |     improves = any(rel_errors[i + 1] < rel_errors[i] for i in range(len(rel_errors) - 1))
00256 |     saturated_at_noise = max(rel_errors) < noise_floor
00257 |     return improves or saturated_at_noise
00258 | 
00259 | 
00260 | class FdWindowCriterionTests(unittest.TestCase):
00261 |     """Synthetic checks of the delta-window convergence criterion."""
00262 | 
00263 |     def test_truncation_then_cancellation_window_passes(self) -> None:
00264 |         # Classic FD signature: error drops with delta, then cancellation bites.
00265 |         self.assertTrue(fd_window_converged([0.20, 0.03, 0.08]))
00266 | 
00267 |     def test_flat_noise_floor_passes(self) -> None:
00268 |         self.assertTrue(fd_window_converged([0.02, 0.02, 0.021]))
00269 | 
00270 |     def test_monotonically_worse_with_smaller_delta_fails(self) -> None:
00271 |         # Shrinking delta only increases the error and errors are large:
00272 |         # the analytic derivative disagrees with the model.
00273 |         self.assertFalse(fd_window_converged([0.30, 0.45, 0.90]))
00274 | 
00275 |     def test_single_delta_requires_noise_floor(self) -> None:
00276 |         self.assertTrue(fd_window_converged([0.01]))
00277 |         self.assertFalse(fd_window_converged([0.30]))
00278 | 
00279 | 
00280 | class DeepHAutogradFiniteDifferenceSmokeTests(unittest.TestCase):
00281 |     @slow_mark
00282 |     def test_deeph_autograd_matches_predict_finite_difference_when_fixture_is_available(self) -> None:
00283 |         """DeepH dH/dR from with_grad vs central finite difference of DeepH predict().
00284 | 
00285 |         Dtype note (audit I6): this smoke exercises the PRODUCTION dtype
00286 |         (float32 DeepH inference). The earlier float64 verification of the
00287 |         autograd route is a separate offline check and is NOT re-executed
00288 |         here; expect the float32 noise floor (best-delta relative error of a
00289 |         few percent), not float64 tightness.
00290 |         """
00291 | 
00292 |         from deeph_config import render_inference_config
00293 | 
00294 |         model_env = os.environ.get("DEEPH_AUTOGRAD_SMOKE_MODEL_DIR")
00295 |         sample_env = os.environ.get("DEEPH_AUTOGRAD_SMOKE_SAMPLE_DIR")
00296 |         if not model_env:
00297 |             raise unittest.SkipTest("Set DEEPH_AUTOGRAD_SMOKE_MODEL_DIR to a trained DeepH model directory.")
00298 |         if not sample_env:
00299 |             raise unittest.SkipTest("Set DEEPH_AUTOGRAD_SMOKE_SAMPLE_DIR to a processed DeepH sample/work directory.")
00300 |         model_dir = Path(model_env)
00301 |         sample_dir = Path(sample_env)
00302 |         deeph_cli_env = os.environ.get("DEEPH_AUTOGRAD_SMOKE_DEEPH_COMMAND")
00303 |         if not deeph_cli_env:
00304 |             raise unittest.SkipTest(
00305 |                 "Set DEEPH_AUTOGRAD_SMOKE_DEEPH_COMMAND to the deeph-inference "
00306 |                 "executable (no machine-specific default)."
00307 |             )
00308 |         deeph_cli = Path(deeph_cli_env)
00309 |         if not model_dir.is_dir():
00310 |             raise unittest.SkipTest(f"DeepH model directory is unavailable: {model_dir}")
00311 |         if not sample_dir.is_dir():
00312 |             raise unittest.SkipTest(f"DeepH sample/work directory is unavailable: {sample_dir}")
00313 |         if not deeph_cli.is_file():
00314 |             raise unittest.SkipTest(f"DeepH inference CLI is unavailable: {deeph_cli}")
00315 | 
00316 |         with tempfile.TemporaryDirectory() as tmp:
00317 |             tmp_path = Path(tmp)
00318 |             atom = int(os.environ.get("DEEPH_AUTOGRAD_SMOKE_ATOM", "0"))
00319 |             axis = int(os.environ.get("DEEPH_AUTOGRAD_SMOKE_AXIS", "2"))
00320 |             deltas = [float(item) for item in os.environ.get("DEEPH_AUTOGRAD_SMOKE_DELTAS", "0.01,0.005").split(",")]
00321 |             base = tmp_path / "base"
00322 |             shutil.copytree(sample_dir, base)
00323 |             autograd_config = tmp_path / "with_grad.ini"
00324 |             render_inference_config(
00325 |                 autograd_config,
00326 |                 work_dir=base,
00327 |                 trained_model_dir=model_dir,
00328 |                 python_interpreter=sys.executable,
00329 |                 task=[3],
00330 |                 with_grad=True,
00331 |                 grad_atom_indices=[atom],
00332 |                 grad_axis_indices=[axis],
00333 |             )
00334 |             _run_deeph_inference(deeph_cli=deeph_cli, config_path=autograd_config)
00335 |             grad_blocks = _read_h5_blocks(base / "hamiltonians_grad_pred.h5")
00336 |             rel_errors: list[float] = []
00337 |             compared = 0
00338 |             grad_norm = 0.0
00339 |             for delta in deltas:
00340 |                 plus = tmp_path / f"plus_{delta:g}"
00341 |                 minus = tmp_path / f"minus_{delta:g}"
00342 |                 _copy_sample_with_shift(sample_dir, plus, atom=atom, axis=axis, delta=delta)
00343 |                 _copy_sample_with_shift(sample_dir, minus, atom=atom, axis=axis, delta=-delta)
00344 |                 for work_dir in (plus, minus):
00345 |                     config_path = work_dir / "predict_fd.ini"
00346 |                     render_inference_config(
00347 |                         config_path,
00348 |                         work_dir=work_dir,
00349 |                         trained_model_dir=model_dir,
00350 |                         python_interpreter=sys.executable,
00351 |                         task=[2, 3, 4],
00352 |                         with_grad=False,
00353 |                     )
00354 |                     _run_deeph_inference(deeph_cli=deeph_cli, config_path=config_path)
00355 |                 plus_blocks = _read_h5_blocks(plus / "hamiltonians_pred.h5")
00356 |                 minus_blocks = _read_h5_blocks(minus / "hamiltonians_pred.h5")
00357 |                 errors = []
00358 |                 refs = []
00359 |                 for key, grad_block in grad_blocks.items():
00360 |                     if key not in plus_blocks or key not in minus_blocks:
00361 |                         continue
00362 |                     analytic = np.asarray(grad_block)[..., atom, axis]
00363 |                     fd = (plus_blocks[key] - minus_blocks[key]) / (2.0 * delta)
00364 |                     errors.append((analytic - fd).reshape(-1))
00365 |                     refs.append(analytic.reshape(-1))
00366 |                 self.assertTrue(errors, "No common DeepH Hamiltonian blocks found between autograd and finite-difference outputs.")
00367 |                 error = np.concatenate(errors)
00368 |                 ref = np.concatenate(refs)
00369 |                 compared += error.size
00370 |                 grad_norm = max(grad_norm, float(np.linalg.norm(ref)))
00371 |                 rel_errors.append(float(np.linalg.norm(error) / (np.linalg.norm(ref) + 1e-30)))
00372 | 
00373 |             self.assertGreater(compared, 0)
00374 |             self.assertGreater(grad_norm, 0.0)
00375 |             self.assertTrue(np.all(np.isfinite(rel_errors)), rel_errors)
00376 |             # Report the full delta -> error map and the optimal delta, not
00377 |             # just the best-case number (audit I6).
00378 |             by_delta = dict(zip(deltas, rel_errors))
00379 |             best_delta = min(by_delta, key=by_delta.get)
00380 |             report = (
00381 |                 f"relative errors by delta (Ang): {by_delta}; "
00382 |                 f"optimal delta {best_delta:g} -> {by_delta[best_delta]:.4f} "
00383 |                 "(float32 production dtype)"
00384 |             )
00385 |             self.assertLess(min(rel_errors), 0.10, report)
00386 |             # FD must show a truncation-dominated window (or sit at the noise
00387 |             # floor); "smaller delta only ever worse" means the analytic
00388 |             # derivative disagrees with the model.
00389 |             self.assertTrue(fd_window_converged(rel_errors), report)
```

## `Comparison/scripts/hamiltonian_derivative_stencil.py` — extractos seleccionados

SHA-256 del archivo completo: `63cfaabc671705e44ba92a640ed3ea95b90f2d2266b109d53e33448a779872dd`

### Cabecera, imports y constantes iniciales

```py
00001 | #!/usr/bin/env python3
00002 | """Data contracts for finite-difference Hamiltonian derivative stencils."""
00003 | 
00004 | from __future__ import annotations
00005 | 
00006 | import math
00007 | import json
00008 | import sys
00009 | from dataclasses import asdict, dataclass, field as dataclass_field
00010 | from pathlib import Path
00011 | from typing import Any
00012 | 
00013 | import numpy as np
00014 | from scipy import sparse
00015 | 
00016 | REPO_ROOT = Path(__file__).resolve().parents[2]
00017 | SHARED_DIR = REPO_ROOT / "shared"
00018 | if str(SHARED_DIR) not in sys.path:
00019 |     sys.path.insert(0, str(SHARED_DIR))
00020 | 
00021 | from fdf_materialization import extract_fdf_structure  # noqa: E402
00022 | from reference_selection import choose_reference_matrix, file_sha256
00023 | 
00024 | 
00025 | VALID_AXES = {"x": 0, "y": 1, "z": 2}
00026 | VALID_METHODS = {"central", "forward", "backward"}
00027 | VALID_SOURCES = {"siesta", "graph2mat", "deeph"}
00028 | REFERENCE_DERIVATIVE_METHOD_SIESTA = "finite_difference_siesta"
00029 | PREDICTED_DERIVATIVE_METHOD_AUTOGRAD_GRAPH2MAT = "autograd_graph2mat_vectorized"
00030 | PREDICTED_DERIVATIVE_METHOD_AUTOGRAD_DEEPH = "autograd_deeph_vectorized"
00031 | GRAPH2MAT_PREDICTION_METHOD_FINITE_DIFFERENCE = "finite_difference"
00032 | GRAPH2MAT_PREDICTION_METHOD_AUTOGRAD = "autograd_vectorized"
00033 | VALID_GRAPH2MAT_PREDICTION_METHODS = {
00034 |     GRAPH2MAT_PREDICTION_METHOD_FINITE_DIFFERENCE,
00035 |     GRAPH2MAT_PREDICTION_METHOD_AUTOGRAD,
00036 | }
00037 | DEEPH_PREDICTION_METHOD_FINITE_DIFFERENCE = "finite_difference"
00038 | DEEPH_PREDICTION_METHOD_AUTOGRAD = "autograd_vectorized"
00039 | VALID_DEEPH_PREDICTION_METHODS = {
00040 |     DEEPH_PREDICTION_METHOD_FINITE_DIFFERENCE,
00041 |     DEEPH_PREDICTION_METHOD_AUTOGRAD,
00042 | }
00043 | DIRECT_DERIVATIVE_PREDICTION_DIRNAME = "predicted_derivative_hamiltonians"
00044 | EXPECTED_HAMILTONIAN_UNITS = "eV"
00045 | EXPECTED_DISPLACEMENT_UNITS = "Ang"
00046 | EXPECTED_DERIVATIVE_UNITS = "eV/Ang"
00047 | DERIVATIVE_SUPPORT_THRESHOLD = 1e-12
00048 | DERIVATIVE_MATRIX_METRIC_TARGET_SPACE = "raw_global_hamiltonian_derivative"
00049 | FORBIDDEN_SIESTA_REFERENCE_NAMES = {"ML_prediction.HSX"}
00050 | DEFAULT_GEOMETRY_TOLERANCE_ANG = 1e-8
00051 | DIAGNOSTIC_STATUSES = {"", "diagnostic", "diagnostic_only", "exploratory"}
00052 | PAPER_LEVEL_STATUSES = {"paper_ready", "publication", "publicable", "final_publication"}
00053 | REQUIRED_NON_DIAGNOSTIC_HASHES = ("material_compatibility_hash", "orbital_ordering_hash")
00054 | OPTIONAL_COMPARABILITY_HASHES = (
00055 |     "material_compatibility_hash",
00056 |     "orbital_ordering_hash",
00057 |     "neighbor_list_hash",
00058 |     "sparsity_pattern_hash",
00059 | )
00060 | COMPARABILITY_HASH_FIELDS = (
00061 |     "material_compatibility_hash",
00062 |     "orbital_ordering_hash",
00063 |     "neighbor_list_hash",
00064 |     "sparsity_pattern_hash",
00065 |     "basis_hash",
00066 |     "pseudopotential_hash",
00067 | )
00068 | 
```

### `DerivativeMetadata` — líneas 92–123

```py
00092 | @dataclass(frozen=True)
00093 | class DerivativeMetadata:
00094 |     sample_id: str
00095 |     plus_sample_id: str | None
00096 |     minus_sample_id: str | None
00097 |     atom_index_zero_based: int | None
00098 |     axis: str | None
00099 |     axis_index: int | None
00100 |     delta_ang: float | None
00101 |     base_sample_id: str | None = None
00102 |     atom_index_one_based: int | None = None
00103 |     hamiltonian_units: str = EXPECTED_HAMILTONIAN_UNITS
00104 |     displacement_units: str = EXPECTED_DISPLACEMENT_UNITS
00105 |     derivative_units: str = EXPECTED_DERIVATIVE_UNITS
00106 |     hamiltonian_units_explicit: bool = True
00107 |     displacement_units_explicit: bool = True
00108 |     derivative_units_explicit: bool = True
00109 |     unit_metadata_explicit: bool = True
00110 |     method: str = "central"
00111 |     claim_status: str = "diagnostic_only"
00112 |     material_compatibility_hash: str | None = None
00113 |     orbital_ordering_hash: str | None = None
00114 |     neighbor_list_hash: str | None = None
00115 |     sparsity_pattern_hash: str | None = None
00116 |     basis_hash: str | None = None
00117 |     pseudopotential_hash: str | None = None
00118 |     structure_hash: str | None = None
00119 |     metadata_hash: str | None = None
00120 |     extra_hashes: dict[str, str] = dataclass_field(default_factory=dict)
00121 | 
00122 |     def to_dict(self) -> dict[str, Any]:
00123 |         return asdict(self)
```

### `DerivativeMatrixInput` — líneas 126–165

```py
00126 | @dataclass(frozen=True)
00127 | class DerivativeMatrixInput:
00128 |     sample_id: str
00129 |     source: str | None
00130 |     matrix_path: Path | str | None
00131 |     matrix_shape: tuple[int, int] | list[int] | None
00132 |     matrix_sha256: str | None = None
00133 |     hamiltonian_units: str = EXPECTED_HAMILTONIAN_UNITS
00134 |     displacement_units: str = EXPECTED_DISPLACEMENT_UNITS
00135 |     derivative_units: str = EXPECTED_DERIVATIVE_UNITS
00136 |     hamiltonian_units_explicit: bool = True
00137 |     displacement_units_explicit: bool = True
00138 |     derivative_units_explicit: bool = True
00139 |     unit_metadata_explicit: bool = True
00140 |     atom_index_zero_based: int | None = None
00141 |     atom_index_one_based: int | None = None
00142 |     axis: str | None = None
00143 |     axis_index: int | None = None
00144 |     delta_ang: float | None = None
00145 |     material_compatibility_hash: str | None = None
00146 |     orbital_ordering_hash: str | None = None
00147 |     neighbor_list_hash: str | None = None
00148 |     sparsity_pattern_hash: str | None = None
00149 |     basis_hash: str | None = None
00150 |     pseudopotential_hash: str | None = None
00151 |     metadata_hash: str | None = None
00152 |     extra_hashes: dict[str, str] = dataclass_field(default_factory=dict)
00153 | 
00154 |     def __post_init__(self) -> None:
00155 |         if self.matrix_path is not None and not isinstance(self.matrix_path, Path):
00156 |             object.__setattr__(self, "matrix_path", Path(self.matrix_path))
00157 |         if self.matrix_shape is not None and not isinstance(self.matrix_shape, tuple):
00158 |             object.__setattr__(self, "matrix_shape", tuple(int(value) for value in self.matrix_shape))
00159 |         if self.matrix_sha256 is None:
00160 |             object.__setattr__(self, "matrix_sha256", file_sha256(self.matrix_path if isinstance(self.matrix_path, Path) else None))
00161 | 
00162 |     def to_dict(self) -> dict[str, Any]:
00163 |         data = asdict(self)
00164 |         data["matrix_path"] = str(self.matrix_path) if self.matrix_path is not None else None
00165 |         return data
```

### `DerivativeStencil` — líneas 168–209

```py
00168 | @dataclass(frozen=True)
00169 | class DerivativeStencil:
00170 |     metadata: DerivativeMetadata
00171 |     siesta_plus: DerivativeMatrixInput | None
00172 |     siesta_minus: DerivativeMatrixInput | None
00173 |     ml_plus: DerivativeMatrixInput | None
00174 |     ml_minus: DerivativeMatrixInput | None
00175 |     siesta_base: DerivativeMatrixInput | None = None
00176 |     ml_base: DerivativeMatrixInput | None = None
00177 |     base_structure_path: Path | str | None = None
00178 |     plus_structure_path: Path | str | None = None
00179 |     minus_structure_path: Path | str | None = None
00180 | 
00181 |     def __post_init__(self) -> None:
00182 |         for field_name in ("base_structure_path", "plus_structure_path", "minus_structure_path"):
00183 |             value = getattr(self, field_name)
00184 |             if value is not None and not isinstance(value, Path):
00185 |                 object.__setattr__(self, field_name, Path(value))
00186 | 
00187 |     def matrix_inputs(self) -> dict[str, DerivativeMatrixInput | None]:
00188 |         return {
00189 |             "siesta_plus": self.siesta_plus,
00190 |             "siesta_minus": self.siesta_minus,
00191 |             "siesta_base": self.siesta_base,
00192 |             "ml_plus": self.ml_plus,
00193 |             "ml_minus": self.ml_minus,
00194 |             "ml_base": self.ml_base,
00195 |         }
00196 | 
00197 |     def to_dict(self) -> dict[str, Any]:
00198 |         return {
00199 |             "metadata": self.metadata.to_dict(),
00200 |             "siesta_plus": self.siesta_plus.to_dict() if self.siesta_plus else None,
00201 |             "siesta_minus": self.siesta_minus.to_dict() if self.siesta_minus else None,
00202 |             "siesta_base": self.siesta_base.to_dict() if self.siesta_base else None,
00203 |             "ml_plus": self.ml_plus.to_dict() if self.ml_plus else None,
00204 |             "ml_minus": self.ml_minus.to_dict() if self.ml_minus else None,
00205 |             "ml_base": self.ml_base.to_dict() if self.ml_base else None,
00206 |             "base_structure_path": str(self.base_structure_path) if self.base_structure_path else None,
00207 |             "plus_structure_path": str(self.plus_structure_path) if self.plus_structure_path else None,
00208 |             "minus_structure_path": str(self.minus_structure_path) if self.minus_structure_path else None,
00209 |         }
```

### `DerivativeMatrixResult` — líneas 212–222

```py
00212 | @dataclass(frozen=True)
00213 | class DerivativeMatrixResult:
00214 |     matrix: sparse.csr_matrix
00215 |     metadata: dict[str, Any]
00216 | 
00217 |     def to_dict(self) -> dict[str, Any]:
00218 |         return {
00219 |             "metadata": dict(self.metadata),
00220 |             "matrix_shape": list(self.matrix.shape),
00221 |             "derivative_nnz": int(self.matrix.nnz),
00222 |         }
```

### `DerivativeComparisonResult` — líneas 225–236

```py
00225 | @dataclass(frozen=True)
00226 | class DerivativeComparisonResult:
00227 |     reference: DerivativeMatrixResult
00228 |     predicted: DerivativeMatrixResult
00229 |     diagnostics: dict[str, Any]
00230 | 
00231 |     def to_dict(self) -> dict[str, Any]:
00232 |         return {
00233 |             "reference": self.reference.to_dict(),
00234 |             "predicted": self.predicted.to_dict(),
00235 |             "diagnostics": dict(self.diagnostics),
00236 |         }
```

### `DerivativeSparseMetrics` — líneas 239–244

```py
00239 | @dataclass(frozen=True)
00240 | class DerivativeSparseMetrics:
00241 |     rows: dict[str, Any]
00242 | 
00243 |     def to_dict(self) -> dict[str, Any]:
00244 |         return dict(self.rows)
```

### `validate_derivative_geometry` — líneas 281–408

```py
00281 | def validate_derivative_geometry(
00282 |     discovery: DerivativeStencilDiscovery,
00283 |     *,
00284 |     tolerance_ang: float = DEFAULT_GEOMETRY_TOLERANCE_ANG,
00285 | ) -> list[DerivativeValidationIssue]:
00286 |     """Validate that derivative stencil structures match the requested displacement."""
00287 | 
00288 |     issues: list[DerivativeValidationIssue] = []
00289 |     stencil = discovery.stencil
00290 |     if stencil is None:
00291 |         return [
00292 |             DerivativeValidationIssue(
00293 |                 severity="error",
00294 |                 code="missing_geometry_stencil",
00295 |                 message="Derivative discovery did not produce a stencil to validate geometrically.",
00296 |                 details={"group_key": list(discovery.group_key)},
00297 |             )
00298 |         ]
00299 |     metadata = stencil.metadata
00300 |     method = str(metadata.method or discovery.method or "").strip().lower()
00301 |     if method not in VALID_METHODS:
00302 |         issues.append(
00303 |             DerivativeValidationIssue(
00304 |                 severity="error",
00305 |                 code="invalid_geometry_method",
00306 |                 message="Geometry validation requires a supported finite-difference method.",
00307 |                 field="method",
00308 |                 sample_id=metadata.sample_id,
00309 |                 details={"method": method},
00310 |             )
00311 |         )
00312 |         return issues
00313 |     if metadata.delta_ang is None or metadata.delta_ang <= 0:
00314 |         issues.append(
00315 |             DerivativeValidationIssue(
00316 |                 severity="error",
00317 |                 code="invalid_geometry_delta",
00318 |                 message="Geometry validation requires a positive delta_ang.",
00319 |                 field="delta_ang",
00320 |                 sample_id=metadata.sample_id,
00321 |             )
00322 |         )
00323 |         return issues
00324 |     if metadata.atom_index_zero_based is None:
00325 |         issues.append(
00326 |             DerivativeValidationIssue(
00327 |                 severity="error",
00328 |                 code="missing_geometry_atom_index",
00329 |                 message="Geometry validation requires atom_index_zero_based.",
00330 |                 field="atom_index_zero_based",
00331 |                 sample_id=metadata.sample_id,
00332 |             )
00333 |         )
00334 |         return issues
00335 |     if metadata.axis_index is None or metadata.axis_index not in VALID_AXES.values():
00336 |         issues.append(
00337 |             DerivativeValidationIssue(
00338 |                 severity="error",
00339 |                 code="missing_geometry_axis_index",
00340 |                 message="Geometry validation requires a valid axis_index.",
00341 |                 field="axis_index",
00342 |                 sample_id=metadata.sample_id,
00343 |             )
00344 |         )
00345 |         return issues
00346 | 
00347 |     structures = _load_geometry_structures(stencil)
00348 |     issues.extend(structures.pop("issues"))
00349 |     base = structures.get("base")
00350 |     if base is None:
00351 |         issues.append(
00352 |             DerivativeValidationIssue(
00353 |                 severity="error",
00354 |                 code="missing_base_structure",
00355 |                 message="Geometry validation requires the base R0 structure for finite-displacement stencils.",
00356 |                 field="base_structure_path",
00357 |                 sample_id=metadata.sample_id,
00358 |             )
00359 |         )
00360 |         return issues
00361 | 
00362 |     roles = _required_geometry_roles(method)
00363 |     for role in roles:
00364 |         if structures.get(role) is None:
00365 |             issues.append(
00366 |                 DerivativeValidationIssue(
00367 |                     severity="error",
00368 |                     code=f"missing_{role}_structure",
00369 |                     message=f"Geometry validation requires the {role} displaced structure.",
00370 |                     field=f"{role}_structure_path",
00371 |                     sample_id=metadata.sample_id,
00372 |                 )
00373 |             )
00374 |     if validation_errors(issues):
00375 |         return issues
00376 | 
00377 |     for role in roles:
00378 |         structure = structures.get(role)
00379 |         if structure is None:
00380 |             continue
00381 |         issues.extend(_validate_structure_identity(base, structure, role=role, metadata=metadata, tolerance_ang=tolerance_ang))
00382 |     if validation_errors(issues):
00383 |         return issues
00384 | 
00385 |     if "plus" in roles and structures.get("plus") is not None:
00386 |         issues.extend(
00387 |             _validate_displacement(
00388 |                 base,
00389 |                 structures["plus"],
00390 |                 role="plus",
00391 |                 sign=1,
00392 |                 metadata=metadata,
00393 |                 tolerance_ang=tolerance_ang,
00394 |             )
00395 |         )
00396 |     if "minus" in roles and structures.get("minus") is not None:
00397 |         issues.extend(
00398 |             _validate_displacement(
00399 |                 base,
00400 |                 structures["minus"],
00401 |                 role="minus",
00402 |                 sign=-1,
00403 |                 metadata=metadata,
00404 |                 tolerance_ang=tolerance_ang,
00405 |             )
00406 |         )
00407 |     issues.extend(_validate_geometry_metadata_family(discovery))
00408 |     return issues
```

### `finite_difference_derivative` — líneas 411–489

```py
00411 | def finite_difference_derivative(
00412 |     *,
00413 |     method: str,
00414 |     delta_ang: float,
00415 |     plus: sparse.spmatrix | None = None,
00416 |     minus: sparse.spmatrix | None = None,
00417 |     base: sparse.spmatrix | None = None,
00418 |     source: str,
00419 |     matrix_hashes: dict[str, str | None] | None = None,
00420 |     hamiltonian_units: str = EXPECTED_HAMILTONIAN_UNITS,
00421 |     displacement_units: str = EXPECTED_DISPLACEMENT_UNITS,
00422 |     derivative_units: str = EXPECTED_DERIVATIVE_UNITS,
00423 |     validation_status: str = "valid",
00424 |     metadata: DerivativeMetadata | None = None,
00425 | ) -> DerivativeMatrixResult:
00426 |     """Return dH/dR from already-loaded sparse Hamiltonian matrices."""
00427 | 
00428 |     method = str(method or "").strip().lower()
00429 |     source = str(source or "").strip().lower()
00430 |     if method not in VALID_METHODS:
00431 |         raise HamiltonianDerivativeError(f"Unsupported finite-difference method: {method!r}.")
00432 |     if source not in VALID_SOURCES:
00433 |         raise HamiltonianDerivativeError(f"Unsupported derivative source: {source!r}.")
00434 |     if delta_ang <= 0:
00435 |         raise HamiltonianDerivativeError("delta_ang must be positive.")
00436 |     if hamiltonian_units != EXPECTED_HAMILTONIAN_UNITS:
00437 |         raise HamiltonianDerivativeError(f"hamiltonian_units must be {EXPECTED_HAMILTONIAN_UNITS!r}.")
00438 |     if displacement_units != EXPECTED_DISPLACEMENT_UNITS:
00439 |         raise HamiltonianDerivativeError(f"displacement_units must be {EXPECTED_DISPLACEMENT_UNITS!r}.")
00440 |     if derivative_units != EXPECTED_DERIVATIVE_UNITS:
00441 |         raise HamiltonianDerivativeError(f"derivative_units must be {EXPECTED_DERIVATIVE_UNITS!r}.")
00442 | 
00443 |     left, right, denominator, operand_names = _finite_difference_operands(
00444 |         method=method,
00445 |         plus=plus,
00446 |         minus=minus,
00447 |         base=base,
00448 |         delta_ang=float(delta_ang),
00449 |     )
00450 |     _require_matching_shapes(operand_names, left, right)
00451 |     left_csr = _csr_copy(left)
00452 |     right_csr = _csr_copy(right)
00453 |     derivative = ((left_csr - right_csr) / denominator).tocsr()
00454 |     derivative.eliminate_zeros()
00455 | 
00456 |     finite_values = _sparse_finite_values(derivative)
00457 |     result_metadata = {
00458 |         "method": method,
00459 |         "delta_ang": float(delta_ang),
00460 |         "hamiltonian_units": hamiltonian_units,
00461 |         "displacement_units": displacement_units,
00462 |         "derivative_units": derivative_units,
00463 |         "source": source,
00464 |         "matrix_hashes": dict(matrix_hashes or {}),
00465 |         "validation_status": validation_status,
00466 |         "operand_roles": list(operand_names),
00467 |         "plus_minus_support_changed": sparse_support_changed(left_csr, right_csr),
00468 |         "derivative_nnz": int(derivative.nnz),
00469 |         "derivative_density": sparse_density(derivative),
00470 |         "finite_values": finite_values,
00471 |         "dH_hermiticity_defect": sparse_hermiticity_defect(derivative),
00472 |     }
00473 |     if metadata is not None:
00474 |         result_metadata.update(
00475 |             {
00476 |                 "sample_id": metadata.sample_id,
00477 |                 "base_sample_id": metadata.base_sample_id,
00478 |                 "plus_sample_id": metadata.plus_sample_id,
00479 |                 "minus_sample_id": metadata.minus_sample_id,
00480 |                 "atom_index_zero_based": metadata.atom_index_zero_based,
00481 |                 "atom_index_one_based": metadata.atom_index_one_based,
00482 |                 "axis": metadata.axis,
00483 |                 "axis_index": metadata.axis_index,
00484 |                 "claim_status": metadata.claim_status,
00485 |             }
00486 |         )
00487 |     if not finite_values:
00488 |         result_metadata["validation_status"] = "invalid_nonfinite_derivative"
00489 |     return DerivativeMatrixResult(matrix=derivative, metadata=result_metadata)
```

### `derivative_signal_to_noise_metrics` — líneas 492–564

```py
00492 | def derivative_signal_to_noise_metrics(
00493 |     *,
00494 |     method: str,
00495 |     reference_plus: sparse.spmatrix | None,
00496 |     reference_minus: sparse.spmatrix | None,
00497 |     reference_base: sparse.spmatrix | None,
00498 |     predicted_plus: sparse.spmatrix | None,
00499 |     predicted_minus: sparse.spmatrix | None,
00500 |     predicted_base: sparse.spmatrix | None,
00501 | ) -> dict[str, Any]:
00502 |     """Diagnostics on whether the finite-difference derivative is above the model noise floor.
00503 | 
00504 |     The finite difference ``dH/dR = (H_plus - H_minus) / (2*delta)`` subtracts two nearly
00505 |     identical absolute Hamiltonians, so the *physical signal* ``||H_plus - H_minus||`` can be
00506 |     much smaller than the model's *absolute-H prediction error* ``||H_pred - H_ref||``. When the
00507 |     signal-to-noise ratio is below ~1 the predicted derivative is dominated by prediction noise
00508 |     rather than the displacement response, which shows up downstream as a large
00509 |     ``dh_relative_frobenius_ref`` and an essentially random (often near +-1) cosine. These fields
00510 |     are diagnostic only: they contextualise the derivative error, they do not change any winner.
00511 |     """
00512 | 
00513 |     def _pick_operands(
00514 |         plus: sparse.spmatrix | None,
00515 |         minus: sparse.spmatrix | None,
00516 |         base: sparse.spmatrix | None,
00517 |     ) -> tuple[sparse.spmatrix | None, sparse.spmatrix | None]:
00518 |         if method == "central":
00519 |             return plus, minus
00520 |         if method == "forward":
00521 |             return plus, base
00522 |         if method == "backward":
00523 |             return base, minus
00524 |         return None, None
00525 | 
00526 |     ref_left, ref_right = _pick_operands(reference_plus, reference_minus, reference_base)
00527 |     pred_left, pred_right = _pick_operands(predicted_plus, predicted_minus, predicted_base)
00528 | 
00529 |     metrics: dict[str, Any] = {
00530 |         "dh_signal_norm_fro": math.nan,
00531 |         "dh_signal_over_abs_h_ref": math.nan,
00532 |         "dh_abs_h_pred_error_norm_fro": math.nan,
00533 |         "dh_abs_h_pred_rel_error_ref": math.nan,
00534 |         "dh_signal_to_noise_ratio": math.nan,
00535 |         "dh_signal_below_noise_floor": None,
00536 |         "dh_signal_to_noise_unavailable_reason": "",
00537 |     }
00538 | 
00539 |     if ref_left is None or ref_right is None:
00540 |         metrics["dh_signal_to_noise_unavailable_reason"] = "missing_reference_operands"
00541 |         return metrics
00542 | 
00543 |     signal_norm = sparse_frobenius_norm((ref_left - ref_right).tocsr())
00544 |     abs_h_ref_norm = sparse_frobenius_norm(ref_left.tocsr())
00545 |     metrics["dh_signal_norm_fro"] = signal_norm
00546 |     metrics["dh_signal_over_abs_h_ref"] = signal_norm / abs_h_ref_norm if abs_h_ref_norm else math.nan
00547 | 
00548 |     if pred_left is None or pred_right is None:
00549 |         metrics["dh_signal_to_noise_unavailable_reason"] = "missing_predicted_operands"
00550 |         return metrics
00551 | 
00552 |     left_error = sparse_frobenius_norm((pred_left - ref_left).tocsr())
00553 |     right_error = sparse_frobenius_norm((pred_right - ref_right).tocsr())
00554 |     noise_norm = 0.5 * (left_error + right_error)
00555 |     metrics["dh_abs_h_pred_error_norm_fro"] = noise_norm
00556 |     metrics["dh_abs_h_pred_rel_error_ref"] = noise_norm / abs_h_ref_norm if abs_h_ref_norm else math.nan
00557 |     if noise_norm:
00558 |         snr = signal_norm / noise_norm
00559 |         metrics["dh_signal_to_noise_ratio"] = snr
00560 |         metrics["dh_signal_below_noise_floor"] = bool(snr < 1.0)
00561 |     else:
00562 |         metrics["dh_signal_to_noise_ratio"] = math.inf if signal_norm else math.nan
00563 |         metrics["dh_signal_below_noise_floor"] = False if signal_norm else None
00564 |     return metrics
```

### `finite_difference_derivative_pair` — líneas 567–634

```py
00567 | def finite_difference_derivative_pair(
00568 |     *,
00569 |     method: str,
00570 |     delta_ang: float,
00571 |     reference_plus: sparse.spmatrix | None = None,
00572 |     reference_minus: sparse.spmatrix | None = None,
00573 |     reference_base: sparse.spmatrix | None = None,
00574 |     predicted_plus: sparse.spmatrix | None = None,
00575 |     predicted_minus: sparse.spmatrix | None = None,
00576 |     predicted_base: sparse.spmatrix | None = None,
00577 |     predicted_source: str = "graph2mat",
00578 |     reference_hashes: dict[str, str | None] | None = None,
00579 |     predicted_hashes: dict[str, str | None] | None = None,
00580 |     metadata: DerivativeMetadata | None = None,
00581 | ) -> DerivativeComparisonResult:
00582 |     """Compute paired SIESTA and ML Hamiltonian derivatives plus diagnostics."""
00583 | 
00584 |     reference = finite_difference_derivative(
00585 |         method=method,
00586 |         delta_ang=delta_ang,
00587 |         plus=reference_plus,
00588 |         minus=reference_minus,
00589 |         base=reference_base,
00590 |         source="siesta",
00591 |         matrix_hashes=reference_hashes,
00592 |         metadata=metadata,
00593 |     )
00594 |     predicted = finite_difference_derivative(
00595 |         method=method,
00596 |         delta_ang=delta_ang,
00597 |         plus=predicted_plus,
00598 |         minus=predicted_minus,
00599 |         base=predicted_base,
00600 |         source=predicted_source,
00601 |         matrix_hashes=predicted_hashes,
00602 |         metadata=metadata,
00603 |     )
00604 |     _require_matching_shapes(("reference_derivative", "predicted_derivative"), reference.matrix, predicted.matrix)
00605 |     signal_to_noise = derivative_signal_to_noise_metrics(
00606 |         method=method,
00607 |         reference_plus=reference_plus,
00608 |         reference_minus=reference_minus,
00609 |         reference_base=reference_base,
00610 |         predicted_plus=predicted_plus,
00611 |         predicted_minus=predicted_minus,
00612 |         predicted_base=predicted_base,
00613 |     )
00614 |     diagnostics = {
00615 |         **signal_to_noise,
00616 |         "dH_ref_hermiticity_defect": reference.metadata["dH_hermiticity_defect"],
00617 |         "dH_pred_hermiticity_defect": predicted.metadata["dH_hermiticity_defect"],
00618 |         "plus_minus_support_changed": bool(
00619 |             reference.metadata["plus_minus_support_changed"]
00620 |             or predicted.metadata["plus_minus_support_changed"]
00621 |         ),
00622 |         "reference_plus_minus_support_changed": reference.metadata["plus_minus_support_changed"],
00623 |         "predicted_plus_minus_support_changed": predicted.metadata["plus_minus_support_changed"],
00624 |         "derivative_nnz": int((reference.matrix != 0).maximum(predicted.matrix != 0).nnz),
00625 |         "reference_derivative_nnz": reference.metadata["derivative_nnz"],
00626 |         "predicted_derivative_nnz": predicted.metadata["derivative_nnz"],
00627 |         "derivative_density": sparse_density((reference.matrix != 0).maximum(predicted.matrix != 0).tocsr()),
00628 |         "reference_derivative_density": reference.metadata["derivative_density"],
00629 |         "predicted_derivative_density": predicted.metadata["derivative_density"],
00630 |         "finite_values": bool(reference.metadata["finite_values"] and predicted.metadata["finite_values"]),
00631 |         "reference_validation_status": reference.metadata["validation_status"],
00632 |         "predicted_validation_status": predicted.metadata["validation_status"],
00633 |     }
00634 |     return DerivativeComparisonResult(reference=reference, predicted=predicted, diagnostics=diagnostics)
```

### `direct_predicted_derivative_pair` — líneas 709–817

```py
00709 | def direct_predicted_derivative_pair(
00710 |     *,
00711 |     method: str,
00712 |     delta_ang: float,
00713 |     reference_plus: sparse.spmatrix | None = None,
00714 |     reference_minus: sparse.spmatrix | None = None,
00715 |     reference_base: sparse.spmatrix | None = None,
00716 |     predicted_matrix: sparse.spmatrix,
00717 |     predicted_source: str = "graph2mat",
00718 |     predicted_derivative_method: str = PREDICTED_DERIVATIVE_METHOD_AUTOGRAD_GRAPH2MAT,
00719 |     reference_hashes: dict[str, str | None] | None = None,
00720 |     predicted_matrix_metadata: dict[str, Any] | None = None,
00721 |     metadata: DerivativeMetadata | None = None,
00722 | ) -> DerivativeComparisonResult:
00723 |     """Pair a finite-difference SIESTA reference with a direct dH_pred/dR matrix.
00724 | 
00725 |     The reference derivative is computed with the existing finite-difference
00726 |     stencil (``reference_delta_ang = delta_ang``). The predicted derivative is
00727 |     taken directly from the model (autograd), so it has no displacement delta
00728 |     (``predicted_delta_ang = None``); it must already be in the same sparse
00729 |     layout/shape as the reference.
00730 |     """
00731 | 
00732 |     reference = finite_difference_derivative(
00733 |         method=method,
00734 |         delta_ang=delta_ang,
00735 |         plus=reference_plus,
00736 |         minus=reference_minus,
00737 |         base=reference_base,
00738 |         source="siesta",
00739 |         matrix_hashes=reference_hashes,
00740 |         metadata=metadata,
00741 |     )
00742 |     predicted_csr = predicted_matrix.tocsr(copy=True)
00743 |     predicted_csr.eliminate_zeros()
00744 |     _require_matching_shapes(
00745 |         ("reference_derivative", "predicted_derivative"), reference.matrix, predicted_csr
00746 |     )
00747 |     predicted_finite = _sparse_finite_values(predicted_csr)
00748 |     predicted_metadata = {
00749 |         "method": predicted_derivative_method,
00750 |         "reference_delta_ang": float(delta_ang),
00751 |         "predicted_delta_ang": None,
00752 |         "hamiltonian_units": EXPECTED_HAMILTONIAN_UNITS,
00753 |         "displacement_units": EXPECTED_DISPLACEMENT_UNITS,
00754 |         "derivative_units": EXPECTED_DERIVATIVE_UNITS,
00755 |         "source": str(predicted_source or "").strip().lower(),
00756 |         "matrix_hashes": {},
00757 |         "validation_status": "valid" if predicted_finite else "invalid_nonfinite_derivative",
00758 |         "operand_roles": ["direct_predicted_derivative"],
00759 |         "plus_minus_support_changed": False,
00760 |         "derivative_nnz": int(predicted_csr.nnz),
00761 |         "derivative_density": sparse_density(predicted_csr),
00762 |         "finite_values": predicted_finite,
00763 |         "dH_hermiticity_defect": sparse_hermiticity_defect(predicted_csr),
00764 |         "direct_prediction_metadata": dict(predicted_matrix_metadata or {}),
00765 |     }
00766 |     if metadata is not None:
00767 |         predicted_metadata.update(
00768 |             {
00769 |                 "sample_id": metadata.sample_id,
00770 |                 "base_sample_id": metadata.base_sample_id,
00771 |                 "atom_index_zero_based": metadata.atom_index_zero_based,
00772 |                 "axis": metadata.axis,
00773 |                 "axis_index": metadata.axis_index,
00774 |                 "claim_status": metadata.claim_status,
00775 |             }
00776 |         )
00777 |     predicted = DerivativeMatrixResult(matrix=predicted_csr, metadata=predicted_metadata)
00778 | 
00779 |     signal_to_noise = derivative_signal_to_noise_metrics(
00780 |         method=method,
00781 |         reference_plus=reference_plus,
00782 |         reference_minus=reference_minus,
00783 |         reference_base=reference_base,
00784 |         predicted_plus=None,
00785 |         predicted_minus=None,
00786 |         predicted_base=None,
00787 |     )
00788 |     if not signal_to_noise.get("dh_signal_to_noise_unavailable_reason"):
00789 |         signal_to_noise["dh_signal_to_noise_unavailable_reason"] = (
00790 |             "direct_predicted_derivative_has_no_displaced_predictions"
00791 |         )
00792 |     diagnostics = {
00793 |         **signal_to_noise,
00794 |         "dH_ref_hermiticity_defect": reference.metadata["dH_hermiticity_defect"],
00795 |         "dH_pred_hermiticity_defect": predicted.metadata["dH_hermiticity_defect"],
00796 |         "plus_minus_support_changed": bool(reference.metadata["plus_minus_support_changed"]),
00797 |         "reference_plus_minus_support_changed": reference.metadata["plus_minus_support_changed"],
00798 |         "predicted_plus_minus_support_changed": False,
00799 |         "derivative_nnz": int((reference.matrix != 0).maximum(predicted.matrix != 0).nnz),
00800 |         "reference_derivative_nnz": reference.metadata["derivative_nnz"],
00801 |         "predicted_derivative_nnz": predicted.metadata["derivative_nnz"],
00802 |         "derivative_density": sparse_density(
00803 |             ((reference.matrix != 0).maximum(predicted.matrix != 0)).tocsr()
00804 |         ),
00805 |         "reference_derivative_density": reference.metadata["derivative_density"],
00806 |         "predicted_derivative_density": predicted.metadata["derivative_density"],
00807 |         "finite_values": bool(
00808 |             reference.metadata["finite_values"] and predicted.metadata["finite_values"]
00809 |         ),
00810 |         "reference_validation_status": reference.metadata["validation_status"],
00811 |         "predicted_validation_status": predicted.metadata["validation_status"],
00812 |         "reference_derivative_method": REFERENCE_DERIVATIVE_METHOD_SIESTA,
00813 |         "predicted_derivative_method": predicted_derivative_method,
00814 |         "reference_delta_ang": float(delta_ang),
00815 |         "predicted_delta_ang": None,
00816 |     }
00817 |     return DerivativeComparisonResult(reference=reference, predicted=predicted, diagnostics=diagnostics)
```

### `derivative_sparse_metrics` — líneas 820–932

```py
00820 | def derivative_sparse_metrics(
00821 |     reference: sparse.spmatrix,
00822 |     predicted: sparse.spmatrix,
00823 |     *,
00824 |     sample: str,
00825 |     metadata: DerivativeMetadata | None = None,
00826 |     source_model: str = "",
00827 |     reference_source: str = "siesta",
00828 |     support_threshold: float = DERIVATIVE_SUPPORT_THRESHOLD,
00829 | ) -> dict[str, Any]:
00830 |     """Compare dH_pred/dR against dH_ref/dR on sparse derivative supports."""
00831 | 
00832 |     reference = reference.tocsr(copy=True)
00833 |     predicted = predicted.tocsr(copy=True)
00834 |     _require_matching_shapes(("reference_derivative", "predicted_derivative"), reference, predicted)
00835 |     ref_values = sparse_value_dict(reference, threshold=support_threshold)
00836 |     pred_values = sparse_value_dict(predicted, threshold=support_threshold)
00837 |     ref_support = set(ref_values)
00838 |     pred_support = set(pred_values)
00839 |     union_support = ref_support | pred_support
00840 |     sorted_union_support = sorted(union_support)
00841 |     intersection = ref_support & pred_support
00842 |     ref_errors = _errors_on_support(ref_values, pred_values, ref_support)
00843 |     pred_errors = _errors_on_support(ref_values, pred_values, pred_support)
00844 |     union_errors = _errors_on_support(ref_values, pred_values, sorted_union_support)
00845 |     eps = 1e-30
00846 |     denominator_epsilon_warning = 1e-20
00847 |     ref_norm = sparse_frobenius_norm(reference)
00848 |     pred_norm = sparse_frobenius_norm(predicted)
00849 |     ref_error_norm = _frobenius_from_values(ref_errors)
00850 |     union_ref_values = [ref_values.get(index, 0.0) for index in sorted_union_support]
00851 |     union_pred_values = [pred_values.get(index, 0.0) for index in sorted_union_support]
00852 |     ref_union_norm = _frobenius_from_values(union_ref_values)
00853 |     pred_union_norm = _frobenius_from_values(union_pred_values)
00854 |     union_error_norm = _frobenius_from_values(union_errors)
00855 |     ref_l1_union = _l1_from_values(union_ref_values)
00856 |     pred_l1_union = _l1_from_values(union_pred_values)
00857 |     error_l1_union = _l1_from_values(union_errors)
00858 |     zero_ref_reason = "reference_derivative_norm_zero" if ref_norm == 0.0 else ""
00859 |     residual_row = _residual_summary_union(union_errors)
00860 |     correlation_row = _correlation_summary_union(union_ref_values, union_pred_values)
00861 |     cosine, cosine_reason = _cosine_similarity_from_values(ref_values, pred_values, union_support)
00862 |     metadata_row = _derivative_metric_metadata(
00863 |         sample=sample,
00864 |         metadata=metadata,
00865 |         source_model=source_model,
00866 |         reference_source=reference_source,
00867 |     )
00868 |     row = {
00869 |         **metadata_row,
00870 |         "support_threshold": support_threshold,
00871 |         "dh_ref_nnz": len(ref_support),
00872 |         "dh_pred_nnz": len(pred_support),
00873 |         "dh_union_nnz": len(union_support),
00874 |         "dh_ref_density": sparse_density(reference),
00875 |         "dh_pred_density": sparse_density(predicted),
00876 |         "dh_union_density": len(union_support) / (reference.shape[0] * reference.shape[1])
00877 |         if reference.shape[0] and reference.shape[1]
00878 |         else math.nan,
00879 |         "dh_mae_ref_eV_per_Ang": _mean_abs(ref_errors),
00880 |         "dh_rmse_ref_eV_per_Ang": _rmse(ref_errors),
00881 |         "dh_mse_ref_eV2_per_Ang2": _mse(ref_errors),
00882 |         "dh_mae_pred_eV_per_Ang": _mean_abs(pred_errors),
00883 |         "dh_rmse_pred_eV_per_Ang": _rmse(pred_errors),
00884 |         "dh_mae_union_eV_per_Ang": _mean_abs(union_errors),
00885 |         "dh_rmse_union_eV_per_Ang": _rmse(union_errors),
00886 |         **residual_row,
00887 |         **correlation_row,
00888 |         "dh_norm_ref_fro": ref_norm,
00889 |         "dh_norm_pred_fro": pred_norm,
00890 |         "dh_norm_error_fro": union_error_norm,
00891 |         "dh_norm_ref_union_fro": ref_union_norm,
00892 |         "dh_norm_pred_union_fro": pred_union_norm,
00893 |         "dh_norm_error_union_fro": union_error_norm,
00894 |         "dh_norm_ref_l1_union": ref_l1_union,
00895 |         "dh_norm_pred_l1_union": pred_l1_union,
00896 |         "dh_norm_error_l1_union": error_l1_union,
00897 |         "dh_relative_frobenius_ref_robust": union_error_norm / (ref_norm + eps),
00898 |         "dh_relative_frobenius_union_robust": union_error_norm / (ref_union_norm + eps),
00899 |         "dh_relative_l1_union_robust": error_l1_union / (ref_l1_union + eps),
00900 |         "dh_relative_frobenius_ref_near_zero_denominator": ref_norm < denominator_epsilon_warning,
00901 |         "dh_relative_frobenius_union_near_zero_denominator": ref_union_norm < denominator_epsilon_warning,
00902 |         "dh_relative_l1_union_near_zero_denominator": ref_l1_union < denominator_epsilon_warning,
00903 |         "dh_max_abs_ref_union_eV_per_Ang": _max_abs(union_ref_values),
00904 |         "dh_max_abs_pred_union_eV_per_Ang": _max_abs(union_pred_values),
00905 |         "dh_max_abs_error_union_eV_per_Ang": _max_abs(union_errors),
00906 |         "dh_relative_frobenius_ref": ref_error_norm / ref_norm if ref_norm else math.nan,
00907 |         "dh_relative_frobenius_union": union_error_norm / ref_union_norm if ref_union_norm else math.nan,
00908 |         # Size-normalized Frobenius (audit Fase 12/16.5): comparable between
00909 |         # 2-atom and 50-atom structures, unlike the absolute norm.
00910 |         "dh_normalized_frobenius_per_element_eV_per_Ang": (
00911 |             union_error_norm / math.sqrt(len(union_support)) if union_support else math.nan
00912 |         ),
00913 |         "dh_matrix_rows": int(reference.shape[0]),
00914 |         "dh_relative_l1_union": error_l1_union / ref_l1_union if ref_l1_union else math.nan,
00915 |         "dh_cosine_similarity_union": cosine,
00916 |         "dh_support_precision": len(intersection) / len(pred_support) if pred_support else math.nan,
00917 |         "dh_support_recall": len(intersection) / len(ref_support) if ref_support else math.nan,
00918 |         "dh_support_f1": _f1(len(intersection), len(pred_support), len(ref_support)),
00919 |         "dh_false_zero_rate": len(ref_support - pred_support) / len(ref_support) if ref_support else math.nan,
00920 |         "dh_false_nonzero_rate": len(pred_support - ref_support) / len(pred_support) if pred_support else math.nan,
00921 |         "dh_hermiticity_ref": sparse_hermiticity_defect(reference),
00922 |         "dh_hermiticity_pred": sparse_hermiticity_defect(predicted),
00923 |         "dh_hermiticity_error_delta": abs(sparse_hermiticity_defect(predicted) - sparse_hermiticity_defect(reference)),
00924 |         "dh_finite_values": bool(_sparse_finite_values(reference) and _sparse_finite_values(predicted)),
00925 |         "dh_relative_unavailable_reason": zero_ref_reason,
00926 |         "dh_cosine_unavailable_reason": cosine_reason,
00927 |     }
00928 |     if ref_union_norm == 0.0 and not row["dh_relative_unavailable_reason"]:
00929 |         row["dh_relative_unavailable_reason"] = "reference_derivative_union_norm_zero"
00930 |     if ref_l1_union == 0.0 and not row["dh_relative_unavailable_reason"]:
00931 |         row["dh_relative_unavailable_reason"] = "reference_derivative_l1_norm_zero"
00932 |     return row
```

### `derivative_ref_abs_quantile_metrics` — líneas 935–1003

```py
00935 | def derivative_ref_abs_quantile_metrics(
00936 |     reference: sparse.spmatrix,
00937 |     predicted: sparse.spmatrix,
00938 |     *,
00939 |     sample: str,
00940 |     metadata: DerivativeMetadata | None = None,
00941 |     source_model: str = "",
00942 |     reference_source: str = "siesta",
00943 |     support_threshold: float = DERIVATIVE_SUPPORT_THRESHOLD,
00944 | ) -> list[dict[str, Any]]:
00945 |     reference = reference.tocsr(copy=True)
00946 |     predicted = predicted.tocsr(copy=True)
00947 |     _require_matching_shapes(("reference_derivative", "predicted_derivative"), reference, predicted)
00948 |     ref_values = sparse_value_dict(reference, threshold=support_threshold)
00949 |     pred_values = sparse_value_dict(predicted, threshold=support_threshold)
00950 |     union_support = sorted(set(ref_values) | set(pred_values))
00951 |     if not union_support:
00952 |         return []
00953 | 
00954 |     metadata_row = _derivative_metric_metadata(
00955 |         sample=sample,
00956 |         metadata=metadata,
00957 |         source_model=source_model,
00958 |         reference_source=reference_source,
00959 |     )
00960 |     entries = sorted(
00961 |         (
00962 |             (
00963 |                 abs(ref_values.get(index, 0.0)),
00964 |                 abs(pred_values.get(index, 0.0) - ref_values.get(index, 0.0)),
00965 |                 abs(pred_values.get(index, 0.0)),
00966 |             )
00967 |             for index in union_support
00968 |         ),
00969 |         key=lambda item: item[0],
00970 |     )
00971 |     rows: list[dict[str, Any]] = []
00972 |     for bin_index, bin_entries in enumerate(np.array_split(np.array(entries, dtype=float), min(4, len(entries))), start=1):
00973 |         if len(bin_entries) == 0:
00974 |             continue
00975 |         abs_ref = bin_entries[:, 0]
00976 |         abs_err = bin_entries[:, 1]
00977 |         abs_pred = bin_entries[:, 2]
00978 |         ref_zero = abs_ref == 0.0
00979 |         rows.append(
00980 |             {
00981 |                 "sample": metadata_row["sample"],
00982 |                 "source_model": metadata_row["source_model"],
00983 |                 "reference_source": metadata_row["reference_source"],
00984 |                 "base_sample_id": metadata.base_sample_id if metadata else None,
00985 |                 "atom_index_zero_based": metadata_row["atom_index_zero_based"],
00986 |                 "axis": metadata_row["axis"],
00987 |                 "delta_ang": metadata_row["delta_ang"],
00988 |                 "finite_difference_method": metadata_row["finite_difference_method"],
00989 |                 "support_threshold": support_threshold,
00990 |                 "quantile_domain": "union_support",
00991 |                 "quantile_bin": bin_index,
00992 |                 "n_entries": int(len(bin_entries)),
00993 |                 "n_ref_zero_entries": int(np.count_nonzero(ref_zero)),
00994 |                 "n_pred_nonzero_ref_zero_entries": int(np.count_nonzero(ref_zero & (abs_pred > support_threshold))),
00995 |                 "abs_ref_min_eV_per_Ang": float(np.min(abs_ref)),
00996 |                 "abs_ref_max_eV_per_Ang": float(np.max(abs_ref)),
00997 |                 "abs_ref_mean_eV_per_Ang": float(np.mean(abs_ref)),
00998 |                 "dh_error_mae_eV_per_Ang": float(np.mean(abs_err)),
00999 |                 "dh_error_rmse_eV_per_Ang": float(np.sqrt(np.mean(abs_err**2))),
01000 |                 "dh_error_relative_l1_robust": float(np.sum(abs_err) / (np.sum(abs_ref) + 1e-30)),
01001 |             }
01002 |         )
01003 |     return rows
```

### `discover_derivative_stencils` — líneas 1222–1345

```py
01222 | def discover_derivative_stencils(
01223 |     result_dir: Path | str,
01224 |     *,
01225 |     method: str | None = None,
01226 |     split: str = "all",
01227 |     finite_difference_method: str | None = None,
01228 |     require_central: bool = False,
01229 |     require_ml_predictions: bool = True,
01230 | ) -> list[DerivativeStencilDiscovery]:
01231 |     """Group existing result directories into finite-difference derivative stencils.
01232 | 
01233 |     The expected layout is the staged comparison layout:
01234 |     structures/<sample>/metadata.json, siesta_hamiltonians/<sample>/*.HSX|*.TSHS,
01235 |     and predicted_hamiltonians/<sample>/ML_prediction.HSX.
01236 | 
01237 |     With ``require_ml_predictions=False`` the discovery skips the per-sample
01238 |     ``ML_prediction.HSX`` requirement entirely: stencils then describe only the
01239 |     SIESTA reference operands. This is used when the predicted derivative comes
01240 |     from a direct (autograd) dH_pred/dR matrix instead of displaced ML
01241 |     predictions.
01242 |     """
01243 | 
01244 |     result_dir = Path(result_dir)
01245 |     source_model = _normalize_source_model(method)
01246 |     requested_method = _normalize_discovery_method(finite_difference_method)
01247 |     structures_root = result_dir / "structures"
01248 |     if not structures_root.exists():
01249 |         return [
01250 |             DerivativeStencilDiscovery(
01251 |                 status="incomplete",
01252 |                 method=requested_method,
01253 |                 group_key=("missing_structures_root", str(structures_root)),
01254 |                 stencil=None,
01255 |                 issues=(
01256 |                     DerivativeValidationIssue(
01257 |                         severity="error",
01258 |                         code="missing_structures_root",
01259 |                         message="Derivative discovery requires result-dir/structures.",
01260 |                         field="result_dir",
01261 |                     ),
01262 |                 ),
01263 |                 details={"result_dir": str(result_dir)},
01264 |             )
01265 |         ]
01266 | 
01267 |     samples = [
01268 |         sample
01269 |         for sample in (
01270 |             _discover_sample(
01271 |                 sample_dir,
01272 |                 result_dir=result_dir,
01273 |                 source_model=source_model,
01274 |                 require_ml_predictions=require_ml_predictions,
01275 |             )
01276 |             for sample_dir in sorted(structures_root.iterdir())
01277 |         )
01278 |         if sample is not None
01279 |     ]
01280 |     discoveries: list[DerivativeStencilDiscovery] = []
01281 |     selected_samples: list[_DiscoveredDerivativeSample] = []
01282 |     requested_split = str(split or "all").strip().lower()
01283 |     for sample in samples:
01284 |         split_issue = _sample_split_issue(sample, requested_split)
01285 |         if split_issue is not None:
01286 |             discoveries.append(_split_filtered_discovery_for_sample(sample, requested_split, split_issue))
01287 |             continue
01288 |         if _sample_in_split(sample, requested_split):
01289 |             selected_samples.append(sample)
01290 |     samples = selected_samples
01291 |     base_samples = [sample for sample in samples if sample.is_base]
01292 |     displaced_samples = [sample for sample in samples if not sample.is_base]
01293 |     ungroupable = [sample for sample in displaced_samples if not sample.can_group]
01294 |     for sample in ungroupable:
01295 |         discoveries.append(_incomplete_discovery_for_sample(sample, requested_method))
01296 | 
01297 |     groups: dict[tuple[Any, ...], dict[int, list[_DiscoveredDerivativeSample]]] = {}
01298 |     for sample in displaced_samples:
01299 |         if not sample.can_group:
01300 |             continue
01301 |         groups.setdefault(sample.group_key, {}).setdefault(int(sample.sign or 0), []).append(sample)
01302 | 
01303 |     for group_key, by_sign in sorted(groups.items(), key=lambda item: str(item[0])):
01304 |         plus_samples = by_sign.get(1, [])
01305 |         minus_samples = by_sign.get(-1, [])
01306 |         if len(plus_samples) > 1 or len(minus_samples) > 1:
01307 |             discoveries.append(_ambiguous_discovery(group_key, plus_samples, minus_samples, requested_method))
01308 |             continue
01309 | 
01310 |         plus = plus_samples[0] if plus_samples else None
01311 |         minus = minus_samples[0] if minus_samples else None
01312 |         base_matches = _matching_base_samples(group_key, base_samples, plus=plus, minus=minus)
01313 |         base_match = base_matches[0] if len(base_matches) == 1 else None
01314 |         if _base_ambiguity_blocks_discovery(
01315 |             requested_method=requested_method,
01316 |             require_central=require_central,
01317 |             plus=plus,
01318 |             minus=minus,
01319 |             base_matches=base_matches,
01320 |         ):
01321 |             discoveries.append(_ambiguous_base_discovery(group_key, plus, minus, base_matches, requested_method))
01322 |             continue
01323 |         methods = _methods_to_emit(
01324 |             requested_method=requested_method,
01325 |             require_central=require_central,
01326 |             plus=plus,
01327 |             minus=minus,
01328 |             base=base_match,
01329 |         )
01330 |         if not methods:
01331 |             discoveries.append(_incomplete_discovery_for_group(group_key, plus, minus, base_match, requested_method or "central"))
01332 |             continue
01333 |         for method_name in methods:
01334 |             discoveries.append(
01335 |                 _build_discovered_stencil(
01336 |                     group_key,
01337 |                     method_name,
01338 |                     plus,
01339 |                     minus,
01340 |                     base_match,
01341 |                     source_model,
01342 |                     require_predicted_operands=require_ml_predictions,
01343 |                 )
01344 |             )
01345 |     return discoveries
```

### `sparse_hermiticity_defect` — líneas 2178–2186

```py
02178 | def sparse_hermiticity_defect(matrix: sparse.spmatrix) -> float:
02179 |     matrix = matrix.tocsr()
02180 |     rows, cols = matrix.shape
02181 |     if rows != cols:
02182 |         return math.nan
02183 |     denominator = sparse_frobenius_norm(matrix)
02184 |     if denominator == 0.0:
02185 |         return math.nan
02186 |     return sparse_frobenius_norm(matrix - matrix.getH()) / denominator
```

### `sparse_blockwise_hermiticity_defect` — líneas 2189–2225

```py
02189 | def sparse_blockwise_hermiticity_defect(
02190 |     matrix: sparse.spmatrix,
02191 |     supercell_order: list[tuple[int, int, int]],
02192 | ) -> float:
02193 |     """Real-space blockwise hermiticity defect: D_ij(R) vs D_ji(-R)^dagger.
02194 | 
02195 |     ``matrix`` is the rectangular (n_orb, n_orb * n_supercells) supercell
02196 |     layout with column blocks ordered as ``supercell_order``. A naive
02197 |     ``H == H^dagger`` check is meaningless for this shape (audit Fase 4/8.3);
02198 |     hermiticity in real space pairs each R block with its -R partner. R vectors
02199 |     whose -R partner is absent from the layout are skipped.
02200 |     """
02201 |     matrix = matrix.tocsr()
02202 |     n_rows, n_cols = matrix.shape
02203 |     n_supercells = len(supercell_order)
02204 |     if n_supercells == 0 or n_cols != n_rows * n_supercells:
02205 |         return math.nan
02206 |     index_by_r = {tuple(int(x) for x in vector): i for i, vector in enumerate(supercell_order)}
02207 |     defect_sq = 0.0
02208 |     norm_sq = 0.0
02209 |     seen: set[tuple[int, int]] = set()
02210 |     for r_vector, block_index in index_by_r.items():
02211 |         minus_index = index_by_r.get((-r_vector[0], -r_vector[1], -r_vector[2]))
02212 |         if minus_index is None:
02213 |             continue
02214 |         pair = (min(block_index, minus_index), max(block_index, minus_index))
02215 |         if pair in seen:
02216 |             continue
02217 |         seen.add(pair)
02218 |         block_r = matrix[:, block_index * n_rows : (block_index + 1) * n_rows]
02219 |         block_minus = matrix[:, minus_index * n_rows : (minus_index + 1) * n_rows]
02220 |         diff = block_r - block_minus.getH()
02221 |         defect_sq += sparse_frobenius_norm(diff) ** 2
02222 |         norm_sq += sparse_frobenius_norm(block_r) ** 2 + sparse_frobenius_norm(block_minus) ** 2
02223 |     if norm_sq == 0.0:
02224 |         return math.nan
02225 |     return math.sqrt(defect_sq / norm_sq)
```

### `validate_derivative_stencil` — líneas 2247–2258

```py
02247 | def validate_derivative_stencil(
02248 |     stencil: DerivativeStencil,
02249 |     *,
02250 |     require_predicted_operands: bool = True,
02251 | ) -> list[DerivativeValidationIssue]:
02252 |     issues: list[DerivativeValidationIssue] = []
02253 |     _validate_metadata(stencil, issues)
02254 |     _validate_operands(stencil, issues, require_predicted_operands=require_predicted_operands)
02255 |     _validate_matrix_shapes(stencil, issues)
02256 |     _validate_operand_metadata(stencil, issues)
02257 |     _validate_comparability_hashes(stencil, issues)
02258 |     return issues
```

### `_validate_metadata` — líneas 2556–2622

```py
02556 | def _validate_metadata(stencil: DerivativeStencil, issues: list[DerivativeValidationIssue]) -> None:
02557 |     metadata = stencil.metadata
02558 |     if not str(metadata.sample_id or "").strip():
02559 |         _issue(issues, "error", "missing_sample_id", "Derivative stencil sample_id is required.", field="sample_id")
02560 |     method = str(metadata.method or "").strip().lower()
02561 |     if method not in VALID_METHODS:
02562 |         _issue(issues, "error", "unsupported_difference_method", f"Unsupported derivative method: {metadata.method!r}.", field="method")
02563 |     if metadata.delta_ang is None or float(metadata.delta_ang) <= 0:
02564 |         _issue(issues, "error", "invalid_delta", "Derivative stencil delta_ang must be positive.", field="delta_ang")
02565 |     if metadata.axis not in VALID_AXES:
02566 |         _issue(issues, "error", "invalid_axis", "Derivative axis must be one of x/y/z.", field="axis")
02567 |     elif metadata.axis_index != VALID_AXES[metadata.axis]:
02568 |         _issue(
02569 |             issues,
02570 |             "error",
02571 |             "axis_index_mismatch",
02572 |             "Derivative axis and axis_index disagree.",
02573 |             field="axis_index",
02574 |             axis=metadata.axis,
02575 |             axis_index=metadata.axis_index,
02576 |         )
02577 |     if metadata.atom_index_zero_based is None or int(metadata.atom_index_zero_based) < 0:
02578 |         _issue(issues, "error", "invalid_atom_index", "atom_index_zero_based must be a non-negative integer.", field="atom_index_zero_based")
02579 |     if metadata.atom_index_one_based is not None and metadata.atom_index_zero_based is not None:
02580 |         if int(metadata.atom_index_one_based) != int(metadata.atom_index_zero_based) + 1:
02581 |             _issue(issues, "error", "atom_index_mismatch", "Zero-based and one-based atom indices disagree.", field="atom_index_one_based")
02582 |     _validate_units(
02583 |         metadata.hamiltonian_units,
02584 |         metadata.displacement_units,
02585 |         metadata.derivative_units,
02586 |         issues,
02587 |         role="stencil",
02588 |         sample_id=metadata.sample_id,
02589 |     )
02590 |     claim_status = str(metadata.claim_status or "").strip().lower()
02591 |     _validate_unit_metadata_explicit(metadata, issues, claim_status=claim_status)
02592 |     if claim_status in PAPER_LEVEL_STATUSES:
02593 |         _issue(
02594 |             issues,
02595 |             "warning",
02596 |             "unsupported_paper_level_status",
02597 |             "Derivative stencil validation is internal/diagnostic; paper-level derivative status is not implemented.",
02598 |             field="claim_status",
02599 |             sample_id=metadata.sample_id,
02600 |             claim_status=metadata.claim_status,
02601 |         )
02602 |     if claim_status not in DIAGNOSTIC_STATUSES:
02603 |         for field_name in REQUIRED_NON_DIAGNOSTIC_HASHES:
02604 |             if not getattr(metadata, field_name):
02605 |                 _issue(
02606 |                     issues,
02607 |                     "error",
02608 |                     "missing_required_metadata",
02609 |                     f"{field_name} is required when derivative comparison claims more than diagnostic status.",
02610 |                     field=field_name,
02611 |                     sample_id=metadata.sample_id,
02612 |                     claim_status=metadata.claim_status,
02613 |                 )
02614 |     if method == "central" and not metadata.base_sample_id and stencil.base_structure_path is None:
02615 |         _issue(
02616 |             issues,
02617 |             "warning",
02618 |             "missing_optional_base_structure",
02619 |             "Central derivative stencil has no optional base structure/sample metadata.",
02620 |             field="base_sample_id",
02621 |             sample_id=metadata.sample_id,
02622 |         )
```

### `_validate_units` — líneas 2625–2652

```py
02625 | def _validate_units(
02626 |     hamiltonian_units: str,
02627 |     displacement_units: str,
02628 |     derivative_units: str,
02629 |     issues: list[DerivativeValidationIssue],
02630 |     *,
02631 |     role: str,
02632 |     sample_id: str | None,
02633 |     matrix_role: str | None = None,
02634 | ) -> None:
02635 |     checks = (
02636 |         ("hamiltonian_units", hamiltonian_units, EXPECTED_HAMILTONIAN_UNITS),
02637 |         ("displacement_units", displacement_units, EXPECTED_DISPLACEMENT_UNITS),
02638 |         ("derivative_units", derivative_units, EXPECTED_DERIVATIVE_UNITS),
02639 |     )
02640 |     for field_name, value, expected in checks:
02641 |         if value != expected:
02642 |             _issue(
02643 |                 issues,
02644 |                 "error",
02645 |                 "unit_mismatch",
02646 |                 f"{role} {field_name} must be {expected!r}, got {value!r}.",
02647 |                 field=field_name,
02648 |                 sample_id=sample_id,
02649 |                 matrix_role=matrix_role,
02650 |                 expected=expected,
02651 |                 actual=value,
02652 |             )
```

### `_validate_unit_metadata_explicit` — líneas 2655–2682

```py
02655 | def _validate_unit_metadata_explicit(
02656 |     metadata: DerivativeMetadata,
02657 |     issues: list[DerivativeValidationIssue],
02658 |     *,
02659 |     claim_status: str,
02660 | ) -> None:
02661 |     missing = [
02662 |         field_name
02663 |         for field_name, explicit in (
02664 |             ("hamiltonian_units", metadata.hamiltonian_units_explicit),
02665 |             ("displacement_units", metadata.displacement_units_explicit),
02666 |             ("derivative_units", metadata.derivative_units_explicit),
02667 |         )
02668 |         if not explicit
02669 |     ]
02670 |     if not missing:
02671 |         return
02672 |     non_diagnostic = claim_status not in DIAGNOSTIC_STATUSES
02673 |     _issue(
02674 |         issues,
02675 |         "error" if non_diagnostic else "warning",
02676 |         "missing_unit_metadata",
02677 |         "Derivative metadata must explicitly record hamiltonian_units, displacement_units, and derivative_units.",
02678 |         field=",".join(missing),
02679 |         sample_id=metadata.sample_id,
02680 |         claim_status=metadata.claim_status,
02681 |         missing_units=missing,
02682 |     )
```

### `_validate_operands` — líneas 2685–2743

```py
02685 | def _validate_operands(
02686 |     stencil: DerivativeStencil,
02687 |     issues: list[DerivativeValidationIssue],
02688 |     *,
02689 |     require_predicted_operands: bool = True,
02690 | ) -> None:
02691 |     method = str(stencil.metadata.method or "").strip().lower()
02692 |     required_roles = {
02693 |         "central": ("siesta_plus", "siesta_minus", "ml_plus", "ml_minus"),
02694 |         "forward": ("siesta_base", "siesta_plus", "ml_base", "ml_plus"),
02695 |         "backward": ("siesta_base", "siesta_minus", "ml_base", "ml_minus"),
02696 |     }.get(method, ())
02697 |     if not require_predicted_operands:
02698 |         # Direct predicted derivatives (autograd) replace the displaced ML
02699 |         # prediction operands; only the SIESTA reference stencil is required.
02700 |         required_roles = tuple(role for role in required_roles if not role.startswith("ml_"))
02701 |     for role in required_roles:
02702 |         if stencil.matrix_inputs()[role] is None:
02703 |             _issue(issues, "error", "missing_derivative_operand", f"Missing required {method} derivative operand: {role}.", matrix_role=role)
02704 |     for role, matrix in stencil.matrix_inputs().items():
02705 |         if matrix is None:
02706 |             continue
02707 |         source = str(matrix.source or "").strip().lower()
02708 |         if not source:
02709 |             _issue(issues, "error", "missing_source_label", "Derivative matrix source is required.", matrix_role=role, sample_id=matrix.sample_id)
02710 |         elif source not in VALID_SOURCES:
02711 |             _issue(
02712 |                 issues,
02713 |                 "error",
02714 |                 "unsupported_source_label",
02715 |                 f"Unsupported derivative matrix source: {matrix.source!r}.",
02716 |                 matrix_role=role,
02717 |                 sample_id=matrix.sample_id,
02718 |             )
02719 |         if matrix.matrix_path is None:
02720 |             _issue(issues, "error", "missing_matrix_path", "Derivative matrix path is required.", matrix_role=role, sample_id=matrix.sample_id)
02721 |         elif source == "siesta" and matrix.matrix_path.name in FORBIDDEN_SIESTA_REFERENCE_NAMES:
02722 |             _issue(
02723 |                 issues,
02724 |                 "error",
02725 |                 "forbidden_siesta_reference",
02726 |                 "ML_prediction.HSX cannot be used as a SIESTA derivative reference.",
02727 |                 matrix_role=role,
02728 |                 sample_id=matrix.sample_id,
02729 |                 matrix_path=str(matrix.matrix_path),
02730 |             )
02731 |         if matrix.matrix_sha256 is None:
02732 |             _issue(issues, "error", "missing_matrix_sha256", "Derivative matrix sha256 is required.", matrix_role=role, sample_id=matrix.sample_id)
02733 |         if matrix.matrix_shape is None:
02734 |             _issue(issues, "error", "missing_matrix_shape", "Derivative matrix shape is required.", matrix_role=role, sample_id=matrix.sample_id)
02735 |         _validate_units(
02736 |             matrix.hamiltonian_units,
02737 |             matrix.displacement_units,
02738 |             matrix.derivative_units,
02739 |             issues,
02740 |             role="matrix",
02741 |             sample_id=matrix.sample_id,
02742 |             matrix_role=role,
02743 |         )
```

### `_validate_comparability_hashes` — líneas 2837–2862

```py
02837 | def _validate_comparability_hashes(stencil: DerivativeStencil, issues: list[DerivativeValidationIssue]) -> None:
02838 |     metadata = stencil.metadata
02839 |     for field_name in OPTIONAL_COMPARABILITY_HASHES:
02840 |         values = _hash_values(stencil, field_name)
02841 |         if not values:
02842 |             _issue(
02843 |                 issues,
02844 |                 "warning",
02845 |                 f"missing_{field_name}",
02846 |                 f"{field_name} is unavailable for derivative comparability validation.",
02847 |                 field=field_name,
02848 |                 sample_id=metadata.sample_id,
02849 |             )
02850 |     for field_name in COMPARABILITY_HASH_FIELDS:
02851 |         values_by_source = _hash_values_by_source(stencil, field_name)
02852 |         unique = sorted(set(values_by_source.values()))
02853 |         if len(unique) > 1:
02854 |             _issue(
02855 |                 issues,
02856 |                 "error",
02857 |                 "metadata_hash_mismatch",
02858 |                 f"{field_name} differs across derivative stencil operands.",
02859 |                 field=field_name,
02860 |                 sample_id=metadata.sample_id,
02861 |                 values=values_by_source,
02862 |             )
```

## `Comparison/scripts/evaluate_hamiltonian_derivative_metrics.py` — extractos seleccionados

SHA-256 del archivo completo: `384fbfafc82e094d8bc808c06cce03df99f4394eba4b01a2513674cc0f61db57`

### Cabecera, imports y constantes iniciales

```py
00001 | #!/usr/bin/env python3
00002 | """Evaluate finite-difference Hamiltonian derivative metrics from archived matrices."""
00003 | 
00004 | from __future__ import annotations
00005 | 
00006 | import argparse
00007 | import csv
00008 | import json
00009 | import math
00010 | import shutil
00011 | import sys
00012 | from dataclasses import replace
00013 | from pathlib import Path
00014 | from typing import Any
00015 | from zipfile import BadZipFile
00016 | 
00017 | import numpy as np
00018 | from scipy import sparse
00019 | import sisl
00020 | 
00021 | SCRIPT_DIR = Path(__file__).resolve().parent
00022 | if str(SCRIPT_DIR) not in sys.path:
00023 |     sys.path.insert(0, str(SCRIPT_DIR))
00024 | _SHARED_DIR = SCRIPT_DIR.parents[1] / "shared"
00025 | if str(_SHARED_DIR) not in sys.path:
00026 |     sys.path.insert(0, str(_SHARED_DIR))
00027 | 
00028 | from run_inventory import collect_run_inventory  # noqa: E402
00029 | from derivative_claim_status import comparison_kind as _comparison_kind  # noqa: E402
00030 | 
00031 | from hamiltonian_derivative_stencil import (  # noqa: E402
00032 |     DERIVATIVE_SUPPORT_THRESHOLD,
00033 |     DEEPH_PREDICTION_METHOD_AUTOGRAD,
00034 |     DEEPH_PREDICTION_METHOD_FINITE_DIFFERENCE,
00035 |     EXPECTED_DERIVATIVE_UNITS,
00036 |     GRAPH2MAT_PREDICTION_METHOD_AUTOGRAD,
00037 |     GRAPH2MAT_PREDICTION_METHOD_FINITE_DIFFERENCE,
00038 |     PREDICTED_DERIVATIVE_METHOD_AUTOGRAD_DEEPH,
00039 |     PREDICTED_DERIVATIVE_METHOD_AUTOGRAD_GRAPH2MAT,
00040 |     REFERENCE_DERIVATIVE_METHOD_SIESTA,
00041 |     VALID_DEEPH_PREDICTION_METHODS,
00042 |     VALID_GRAPH2MAT_PREDICTION_METHODS,
00043 |     DerivativeMatrixInput,
00044 |     DerivativeMetadata,
00045 |     DerivativeStencil,
00046 |     DerivativeStencilDiscovery,
00047 |     derivative_ref_abs_quantile_metrics,
00048 |     derivative_sparse_metrics,
00049 |     direct_predicted_derivative_pair,
00050 |     discover_derivative_stencils,
00051 |     find_direct_derivative_prediction,
00052 |     finite_difference_derivative_pair,
00053 |     load_direct_sparse_derivative,
00054 |     validate_derivative_geometry,
00055 |     validate_derivative_stencil,
00056 |     validation_errors,
00057 | )
00058 | 
00059 | 
00060 | SCHEMA_VERSION = "hamiltonian_derivative_metrics_v1"
00061 | REFERENCE_DEFINITION = "siesta_hamiltonian_finite_difference"
00062 | SUPPORT_THRESHOLDS_SWEEP = (1e-12, 1e-10, 1e-8, 1e-6)
00063 | GROUPING_PRESERVES = [
00064 |     "source_model",
00065 |     "reference_source",
00066 |     "dataset_size",
00067 |     "seed",
00068 |     "split",
00069 |     "delta_ang",
00070 |     "finite_difference_method",
00071 |     "support_threshold",
00072 | ]
00073 | GROUP_MEAN_MEDIAN_FIELDS = [
00074 |     "dh_relative_frobenius_union_robust",
00075 |     "dh_mae_union_eV_per_Ang",
00076 |     "dh_rmse_union_eV_per_Ang",
00077 |     "dh_relative_l1_union_robust",
00078 | ]
00079 | SCALAR_DELTA_STABILITY_DEFINITION = "scalar_error_metric_pairwise_delta_change_not_matrix_delta_stability"
00080 | DELTA_STABILITY_PAIRWISE_GROUP_KEYS = [
00081 |     "source_model",
00082 |     "reference_source",
00083 |     "dataset_size",
00084 |     "seed",
00085 |     "split",
00086 |     "base_sample_id",
00087 |     "atom_index_zero_based",
00088 |     "axis",
00089 |     "finite_difference_method",
00090 |     "support_threshold",
00091 | ]
00092 | STATUS_FIELDS = [
00093 |     "sample",
00094 |     "status",
00095 |     "finite_difference_method",
00096 |     "base_sample_id",
00097 |     "plus_sample_id",
00098 |     "minus_sample_id",
00099 |     "atom_index_zero_based",
00100 |     "axis",
00101 |     "axis_index",
00102 |     "delta_ang",
00103 |     "issue_codes",
00104 |     "issue_messages",
00105 | ]
00106 | HERMITICITY_FIELDS = [
00107 |     "sample",
00108 |     "finite_difference_method",
00109 |     "source_model",
00110 |     "reference_source",
00111 |     "dH_ref_hermiticity_defect",
00112 |     "dH_pred_hermiticity_defect",
00113 |     "dH_hermiticity_error_delta",
00114 |     "finite_values",
00115 | ]
00116 | # Diagnostic-only: contextualises dh_relative_frobenius_ref by reporting whether the physical
00117 | # derivative signal ||H_plus - H_minus|| is above the model's absolute-H prediction error.
00118 | DERIVATIVE_SIGNAL_TO_NOISE_FIELDS = [
00119 |     "dh_signal_norm_fro",
00120 |     "dh_signal_over_abs_h_ref",
00121 |     "dh_abs_h_pred_error_norm_fro",
00122 |     "dh_abs_h_pred_rel_error_ref",
00123 |     "dh_signal_to_noise_ratio",
00124 |     "dh_signal_below_noise_floor",
00125 |     "dh_signal_to_noise_unavailable_reason",
00126 | ]
00127 | DEEPH_EQUIVALENCE_FIELDS = [
00128 |     "claim_status",
00129 |     "deeph_adapter_equivalence_status",
00130 |     "deeph_equivalence_status",
00131 |     "deeph_equivalence_scope",
00132 |     "deeph_equivalence_reason",
00133 |     "deeph_equivalence_evidence_paths",
00134 |     "deeph_raw_global_equivalence_proven",
00135 |     "deeph_diagnostic_only",
00136 |     "deeph_diagnostic_reason",
00137 | ]
00138 | GEOMETRY_VALIDATION_FIELDS = [
00139 |     "sample",
00140 |     "status",
00141 |     "finite_difference_method",
00142 |     "base_sample_id",
00143 |     "plus_sample_id",
00144 |     "minus_sample_id",
00145 |     "atom_index_zero_based",
00146 |     "axis",
00147 |     "axis_index",
00148 |     "delta_ang",
00149 |     "issue_codes",
00150 |     "issue_messages",
00151 | ]
00152 | DERIVATIVE_REF_ABS_QUANTILE_FIELDS = [
00153 |     "sample",
00154 |     "source_model",
00155 |     "reference_source",
00156 |     "base_sample_id",
00157 |     "atom_index_zero_based",
00158 |     "axis",
00159 |     "delta_ang",
00160 |     "finite_difference_method",
00161 |     "support_threshold",
00162 |     "quantile_domain",
00163 |     "quantile_bin",
00164 |     "n_entries",
00165 |     "n_ref_zero_entries",
00166 |     "n_pred_nonzero_ref_zero_entries",
00167 |     "abs_ref_min_eV_per_Ang",
00168 |     "abs_ref_max_eV_per_Ang",
00169 |     "abs_ref_mean_eV_per_Ang",
00170 |     "dh_error_mae_eV_per_Ang",
00171 |     "dh_error_rmse_eV_per_Ang",
00172 |     "dh_error_relative_l1_robust",
00173 | ]
00174 | DELTA_STABILITY_FIELDS = [
00175 |     "source_model",
00176 |     "base_sample_id",
00177 |     "atom_index_zero_based",
00178 |     "axis",
00179 |     "finite_difference_method",
00180 |     "delta_count",
00181 |     "delta_min_ang",
00182 |     "delta_max_ang",
00183 |     "dh_mae_union_eV_per_Ang_min",
00184 |     "dh_mae_union_eV_per_Ang_max",
00185 |     "dh_mae_union_eV_per_Ang_range",
00186 |     "dh_rmse_union_eV_per_Ang_min",
00187 |     "dh_rmse_union_eV_per_Ang_max",
00188 |     "dh_rmse_union_eV_per_Ang_range",
00189 |     "dh_relative_frobenius_ref_min",
00190 |     "dh_relative_frobenius_ref_max",
00191 |     "dh_relative_frobenius_ref_range",
00192 |     "status",
00193 | ]
00194 | 
```

### `evaluate_derivative_metrics` — líneas 270–479

```py
00270 | def evaluate_derivative_metrics(
00271 |     result_dir: Path,
00272 |     *,
00273 |     method: str,
00274 |     split: str = "all",
00275 |     require_central: bool = False,
00276 |     overwrite: bool = False,
00277 |     diagnostic_only: bool = False,
00278 |     support_threshold: float = DERIVATIVE_SUPPORT_THRESHOLD,
00279 |     max_stencils: int | None = None,
00280 |     output_dir: Path | None = None,
00281 |     source_model: str = "graph2mat",
00282 |     graph2mat_prediction_method: str = GRAPH2MAT_PREDICTION_METHOD_FINITE_DIFFERENCE,
00283 |     deeph_prediction_method: str = DEEPH_PREDICTION_METHOD_FINITE_DIFFERENCE,
00284 | ) -> dict[str, Any]:
00285 |     result_dir = Path(result_dir)
00286 |     output_dir = Path(output_dir) if output_dir is not None else result_dir / "derivative_metrics"
00287 |     source_model = str(source_model or "").strip().lower()
00288 | 
00289 |     graph2mat_prediction_method = str(
00290 |         graph2mat_prediction_method or GRAPH2MAT_PREDICTION_METHOD_FINITE_DIFFERENCE
00291 |     ).strip().lower()
00292 |     deeph_prediction_method = str(
00293 |         deeph_prediction_method or DEEPH_PREDICTION_METHOD_FINITE_DIFFERENCE
00294 |     ).strip().lower()
00295 |     if graph2mat_prediction_method not in VALID_GRAPH2MAT_PREDICTION_METHODS:
00296 |         raise DerivativeMetricEvaluationError(
00297 |             f"Unsupported graph2mat_prediction_method {graph2mat_prediction_method!r}. "
00298 |             f"Use one of: {', '.join(sorted(VALID_GRAPH2MAT_PREDICTION_METHODS))}."
00299 |         )
00300 |     if deeph_prediction_method not in VALID_DEEPH_PREDICTION_METHODS:
00301 |         raise DerivativeMetricEvaluationError(
00302 |             f"Unsupported deeph_prediction_method {deeph_prediction_method!r}. "
00303 |             f"Use one of: {', '.join(sorted(VALID_DEEPH_PREDICTION_METHODS))}."
00304 |         )
00305 |     graph2mat_direct_mode = (
00306 |         source_model == "graph2mat" and graph2mat_prediction_method == GRAPH2MAT_PREDICTION_METHOD_AUTOGRAD
00307 |     )
00308 |     deeph_direct_mode = source_model == "deeph" and deeph_prediction_method == DEEPH_PREDICTION_METHOD_AUTOGRAD
00309 |     direct_prediction_mode = graph2mat_direct_mode or deeph_direct_mode
00310 |     if graph2mat_prediction_method == GRAPH2MAT_PREDICTION_METHOD_AUTOGRAD and source_model != "graph2mat":
00311 |         raise DerivativeMetricEvaluationError(
00312 |             "graph2mat_prediction_method='autograd_vectorized' only applies to "
00313 |             f"source_model='graph2mat', got source_model={source_model!r}."
00314 |         )
00315 |     if deeph_prediction_method == DEEPH_PREDICTION_METHOD_AUTOGRAD and source_model != "deeph":
00316 |         raise DerivativeMetricEvaluationError(
00317 |             "deeph_prediction_method='autograd_vectorized' only applies to "
00318 |             f"source_model='deeph', got source_model={source_model!r}."
00319 |         )
00320 |     predicted_derivative_method = (
00321 |         PREDICTED_DERIVATIVE_METHOD_AUTOGRAD_GRAPH2MAT
00322 |         if graph2mat_direct_mode
00323 |         else PREDICTED_DERIVATIVE_METHOD_AUTOGRAD_DEEPH
00324 |         if deeph_direct_mode
00325 |         else f"finite_difference_{source_model}"
00326 |     )
00327 |     ensure_output_dir(output_dir, overwrite=overwrite)
00328 | 
00329 |     discoveries = discover_derivative_stencils(
00330 |         result_dir,
00331 |         method=source_model,
00332 |         split=split,
00333 |         finite_difference_method=method,
00334 |         require_central=require_central,
00335 |         require_ml_predictions=not direct_prediction_mode,
00336 |     )
00337 |     if max_stencils is not None:
00338 |         discoveries = discoveries[: max(0, int(max_stencils))]
00339 | 
00340 |     stencil_rows: list[dict[str, Any]] = []
00341 |     metric_rows: list[dict[str, Any]] = []
00342 |     quantile_rows: list[dict[str, Any]] = []
00343 |     sweep_rows: list[dict[str, Any]] = []
00344 |     hermiticity_rows: list[dict[str, Any]] = []
00345 |     geometry_rows: list[dict[str, Any]] = []
00346 |     warnings: list[dict[str, Any]] = []
00347 |     fatal_errors: list[dict[str, Any]] = []
00348 | 
00349 |     for discovery in discoveries:
00350 |         row, metrics, quantiles, sweep, hermiticity, geometry, warning_rows, error_rows = _evaluate_discovery(
00351 |             discovery,
00352 |             method=method,
00353 |             source_model=source_model,
00354 |             support_threshold=support_threshold,
00355 |             diagnostic_only=diagnostic_only,
00356 |             result_dir=result_dir,
00357 |             direct_prediction_mode=direct_prediction_mode,
00358 |             predicted_derivative_method=predicted_derivative_method,
00359 |             graph2mat_prediction_method=graph2mat_prediction_method,
00360 |             deeph_prediction_method=deeph_prediction_method,
00361 |         )
00362 |         stencil_rows.append(row)
00363 |         metric_rows.extend(metrics)
00364 |         quantile_rows.extend(quantiles)
00365 |         sweep_rows.extend(sweep)
00366 |         hermiticity_rows.extend(hermiticity)
00367 |         geometry_rows.append(geometry)
00368 |         warnings.extend(warning_rows)
00369 |         fatal_errors.extend(error_rows)
00370 | 
00371 |     stencils_total = len(discoveries)
00372 |     stencils_ok = len(metric_rows)
00373 |     stencils_failed = stencils_total - stencils_ok
00374 |     deeph_equivalence = _deeph_equivalence_summary(
00375 |         source_model=source_model,
00376 |         deeph_prediction_method=deeph_prediction_method,
00377 |         metric_rows=metric_rows,
00378 |     )
00379 |     scientific_status = _scientific_status(
00380 |         method=method,
00381 |         diagnostic_only=diagnostic_only,
00382 |         stencils_total=stencils_total,
00383 |         stencils_ok=stencils_ok,
00384 |         stencils_failed=stencils_failed,
00385 |         metric_rows=metric_rows,
00386 |         fatal_errors=fatal_errors,
00387 |         deeph_equivalence=deeph_equivalence,
00388 |     )
00389 |     delta_stability = _delta_stability_summary(metric_rows)
00390 |     delta_stability_convergence = _delta_stability_convergence_summary(delta_stability)
00391 |     reference_noise = _reference_noise_summary(metric_rows)
00392 |     summary = _summary(metric_rows, stencil_rows, hermiticity_rows)
00393 |     group_metrics = _derivative_group_metrics(metric_rows, split=split)
00394 |     onsite_offsite_metrics = {"available": False, "reason": "orbital_to_atom_mapping_unavailable"}
00395 |     warnings.append(
00396 |         {
00397 |             "kind": "derivative_onsite_offsite_metrics_unavailable",
00398 |             "message": "Onsite/offsite derivative metrics were not computed because orbital-to-atom mapping was unavailable.",
00399 |             "reason": "orbital_to_atom_mapping_unavailable",
00400 |         }
00401 |     )
00402 |     summary["delta_stability"] = {**delta_stability, **delta_stability_convergence}
00403 |     summary.update(delta_stability_convergence)
00404 |     summary["reference_noise"] = reference_noise
00405 |     outputs = {
00406 |         "output_dir": str(output_dir),
00407 |         "manifest": str(output_dir / "manifest.json"),
00408 |         "stencil_status": str(output_dir / "stencil_status.csv"),
00409 |         "derivative_matrix_metrics": str(output_dir / "derivative_matrix_metrics.csv"),
00410 |         "derivative_ref_abs_quantile_metrics": str(output_dir / "derivative_ref_abs_quantile_metrics.csv"),
00411 |         "derivative_support_sweep": str(output_dir / "derivative_support_sweep.csv"),
00412 |         "derivative_hermiticity": str(output_dir / "derivative_hermiticity.csv"),
00413 |         "derivative_delta_stability": str(output_dir / "derivative_delta_stability.csv"),
00414 |         "derivative_delta_stability_json": str(output_dir / "derivative_delta_stability.json"),
00415 |         "derivative_geometry_validation": str(output_dir / "derivative_geometry_validation.csv"),
00416 |         "derivative_geometry_validation_json": str(output_dir / "derivative_geometry_validation.json"),
00417 |         "derivative_summary": str(output_dir / "derivative_summary.json"),
00418 |         "derivative_group_metrics": str(output_dir / "derivative_group_metrics.json"),
00419 |         "derivative_onsite_offsite_metrics": str(output_dir / "derivative_onsite_offsite_metrics.json"),
00420 |     }
00421 |     manifest = {
00422 |         "schema_version": SCHEMA_VERSION,
00423 |         "scientific_status": scientific_status,
00424 |         "paper_level": False,
00425 |         "finite_difference_method": method,
00426 |         "force_constants_used": False,
00427 |         "reference_definition": REFERENCE_DEFINITION,
00428 |         "reference_derivative_method": REFERENCE_DERIVATIVE_METHOD_SIESTA,
00429 |         "predicted_derivative_method": predicted_derivative_method,
00430 |         # Which of the three protocol comparisons this manifest encodes
00431 |         # (audit Fase 11): B model_fd_vs_siesta_fd or C model_autograd_vs_siesta_fd.
00432 |         "comparison_kind": _comparison_kind(
00433 |             predicted_derivative_method, REFERENCE_DERIVATIVE_METHOD_SIESTA
00434 |         ),
00435 |         "graph2mat_prediction_method": graph2mat_prediction_method
00436 |         if source_model == "graph2mat"
00437 |         else None,
00438 |         "deeph_prediction_method": deeph_prediction_method if source_model == "deeph" else None,
00439 |         "predicted_delta_ang": None if direct_prediction_mode else "per_stencil_delta_ang",
00440 |         "derivative_units": EXPECTED_DERIVATIVE_UNITS,
00441 |         "result_dir": str(result_dir),
00442 |         "split": split,
00443 |         "require_central": bool(require_central),
00444 |         "diagnostic_only_requested": bool(diagnostic_only),
00445 |         "support_threshold": float(support_threshold),
00446 |         "support_threshold_sweep": list(SUPPORT_THRESHOLDS_SWEEP),
00447 |         "stencils_total": stencils_total,
00448 |         "stencils_ok": stencils_ok,
00449 |         "stencils_failed": stencils_failed,
00450 |         "geometry_validation": _geometry_validation_summary(geometry_rows),
00451 |         "delta_stability": delta_stability,
00452 |         "delta_sensitivity_study_available": delta_stability_convergence["delta_sensitivity_study_available"],
00453 |         "delta_sensitivity_study_passed": delta_stability_convergence["delta_sensitivity_study_passed"],
00454 |         "delta_stability_converged": delta_stability_convergence["delta_stability_converged"],
00455 |         "delta_stability_convergence_status": delta_stability_convergence["delta_stability_convergence_status"],
00456 |         "reference_noise": reference_noise,
00457 |         "reference_noise_status": reference_noise["status"],
00458 |         "warnings": warnings,
00459 |         "fatal_errors": fatal_errors,
00460 |         "run_inventory": collect_run_inventory(),
00461 |         "outputs": outputs,
00462 |         **deeph_equivalence,
00463 |     }
00464 | 
00465 |     write_csv(output_dir / "stencil_status.csv", STATUS_FIELDS, stencil_rows)
00466 |     write_csv(output_dir / "derivative_matrix_metrics.csv", _metric_fieldnames(metric_rows), metric_rows)
00467 |     write_csv(output_dir / "derivative_ref_abs_quantile_metrics.csv", DERIVATIVE_REF_ABS_QUANTILE_FIELDS, quantile_rows)
00468 |     write_csv(output_dir / "derivative_support_sweep.csv", _metric_fieldnames(sweep_rows), sweep_rows)
00469 |     write_csv(output_dir / "derivative_hermiticity.csv", HERMITICITY_FIELDS, hermiticity_rows)
00470 |     delta_stability_json = {**delta_stability, **delta_stability_convergence}
00471 |     write_csv(output_dir / "derivative_delta_stability.csv", DELTA_STABILITY_FIELDS, delta_stability["rows"])
00472 |     write_json(output_dir / "derivative_delta_stability.json", delta_stability_json)
00473 |     write_csv(output_dir / "derivative_geometry_validation.csv", GEOMETRY_VALIDATION_FIELDS, geometry_rows)
00474 |     write_json(output_dir / "derivative_geometry_validation.json", _geometry_validation_summary(geometry_rows, include_rows=True))
00475 |     write_json(output_dir / "derivative_summary.json", summary)
00476 |     write_json(output_dir / "derivative_group_metrics.json", group_metrics)
00477 |     write_json(output_dir / "derivative_onsite_offsite_metrics.json", onsite_offsite_metrics)
00478 |     write_json(output_dir / "manifest.json", manifest)
00479 |     return manifest
```

### `_evaluate_discovery` — líneas 482–720

```py
00482 | def _evaluate_discovery(
00483 |     discovery: DerivativeStencilDiscovery,
00484 |     *,
00485 |     method: str,
00486 |     source_model: str,
00487 |     support_threshold: float,
00488 |     diagnostic_only: bool,
00489 |     result_dir: Path | None = None,
00490 |     direct_prediction_mode: bool = False,
00491 |     predicted_derivative_method: str = PREDICTED_DERIVATIVE_METHOD_AUTOGRAD_GRAPH2MAT,
00492 |     graph2mat_prediction_method: str = GRAPH2MAT_PREDICTION_METHOD_FINITE_DIFFERENCE,
00493 |     deeph_prediction_method: str = DEEPH_PREDICTION_METHOD_FINITE_DIFFERENCE,
00494 | ) -> tuple[
00495 |     list[Any] | dict[str, Any],
00496 |     list[dict[str, Any]],
00497 |     list[dict[str, Any]],
00498 |     list[dict[str, Any]],
00499 |     dict[str, Any],
00500 |     list[dict[str, Any]],
00501 |     list[dict[str, Any]],
00502 |     list[dict[str, Any]],
00503 | ]:
00504 |     warnings: list[dict[str, Any]] = []
00505 |     fatal_errors: list[dict[str, Any]] = []
00506 |     metric_rows: list[dict[str, Any]] = []
00507 |     quantile_rows: list[dict[str, Any]] = []
00508 |     sweep_rows: list[dict[str, Any]] = []
00509 |     hermiticity_rows: list[dict[str, Any]] = []
00510 |     status_row = _stencil_status_row(discovery, status=discovery.status)
00511 |     geometry_issues = validate_derivative_geometry(discovery)
00512 |     geometry_row = _geometry_validation_row(discovery, geometry_issues)
00513 |     geometry_errors = validation_errors(geometry_issues)
00514 |     if discovery.stencil is None:
00515 |         fatal_errors.append(_discovery_error(discovery, "missing_stencil", "Discovery did not produce a complete stencil."))
00516 |         return status_row, metric_rows, quantile_rows, sweep_rows, hermiticity_rows, geometry_row, warnings, fatal_errors
00517 |     if geometry_errors and not diagnostic_only:
00518 |         status_row = _stencil_status_row(
00519 |             replace(discovery, issues=tuple([*discovery.issues, *geometry_issues])),
00520 |             status="failed",
00521 |         )
00522 |         fatal_errors.append(
00523 |             _discovery_error(
00524 |                 discovery,
00525 |                 "derivative_geometry_validation_failed",
00526 |                 "Derivative geometry validation failed before metric evaluation.",
00527 |                 issue_codes=[issue.code for issue in geometry_errors],
00528 |             )
00529 |         )
00530 |         return status_row, metric_rows, quantile_rows, sweep_rows, hermiticity_rows, geometry_row, warnings, fatal_errors
00531 |     if geometry_errors:
00532 |         warnings.append(
00533 |             _discovery_error(
00534 |                 discovery,
00535 |                 "derivative_geometry_validation_diagnostic_only",
00536 |                 "Derivative geometry validation failed, but diagnostic-only mode allowed metric evaluation to continue.",
00537 |                 issue_codes=[issue.code for issue in geometry_errors],
00538 |             )
00539 |         )
00540 | 
00541 |     try:
00542 |         loaded = _load_stencil_matrices(discovery.stencil)
00543 |         stencil = _stencil_with_loaded_shapes(discovery.stencil, loaded)
00544 |         if diagnostic_only:
00545 |             stencil = replace(stencil, metadata=replace(stencil.metadata, claim_status="diagnostic_only"))
00546 |         validation = validate_derivative_stencil(
00547 |             stencil, require_predicted_operands=not direct_prediction_mode
00548 |         )
00549 |         errors = validation_errors(validation)
00550 |         if errors:
00551 |             status_row = _stencil_status_row(
00552 |                 replace(discovery, stencil=stencil, issues=tuple([*discovery.issues, *validation])),
00553 |                 status="failed",
00554 |             )
00555 |             fatal_errors.append(
00556 |                 _discovery_error(
00557 |                     discovery,
00558 |                     "stencil_validation_failed",
00559 |                     "Derivative stencil validation failed after loading matrices.",
00560 |                     issue_codes=[issue.code for issue in errors],
00561 |                 )
00562 |             )
00563 |             return status_row, metric_rows, quantile_rows, sweep_rows, hermiticity_rows, geometry_row, warnings, fatal_errors
00564 |         metadata = _metadata_for_status(stencil.metadata, diagnostic_only=diagnostic_only)
00565 |         direct_prediction_path: Path | None = None
00566 |         direct_metadata: dict[str, Any] = {}
00567 |         if direct_prediction_mode:
00568 |             candidate_base_ids = [
00569 |                 str(metadata.base_sample_id or ""),
00570 |                 f"{_group_base_id(discovery)}_base",
00571 |                 _group_base_id(discovery),
00572 |             ]
00573 |             direct_prediction_path = find_direct_derivative_prediction(
00574 |                 result_dir if result_dir is not None else Path("."),
00575 |                 candidate_base_sample_ids=candidate_base_ids,
00576 |                 atom_index_zero_based=int(metadata.atom_index_zero_based),
00577 |                 axis_index=int(metadata.axis_index),
00578 |             )
00579 |             if direct_prediction_path is None:
00580 |                 status_row = _stencil_status_row(discovery, status="failed")
00581 |                 fatal_errors.append(
00582 |                     _discovery_error(
00583 |                         discovery,
00584 |                         "missing_direct_derivative_prediction",
00585 |                         "No direct dH_pred/dR matrix was found for this stencil; "
00586 |                         f"run the {source_model} autograd derivative prediction stage first.",
00587 |                         candidate_base_sample_ids=[c for c in candidate_base_ids if c],
00588 |                     )
00589 |                 )
00590 |                 return status_row, metric_rows, quantile_rows, sweep_rows, hermiticity_rows, geometry_row, warnings, fatal_errors
00591 |             predicted_matrix, direct_metadata = load_direct_sparse_derivative(direct_prediction_path)
00592 |             pair = direct_predicted_derivative_pair(
00593 |                 method=method,
00594 |                 delta_ang=float(metadata.delta_ang),
00595 |                 reference_plus=loaded.get("siesta_plus"),
00596 |                 reference_minus=loaded.get("siesta_minus"),
00597 |                 reference_base=loaded.get("siesta_base"),
00598 |                 predicted_matrix=predicted_matrix,
00599 |                 predicted_source=source_model,
00600 |                 predicted_derivative_method=predicted_derivative_method,
00601 |                 reference_hashes=_matrix_hashes(stencil, prefix="siesta"),
00602 |                 predicted_matrix_metadata=direct_metadata,
00603 |                 metadata=metadata,
00604 |             )
00605 |         else:
00606 |             pair = finite_difference_derivative_pair(
00607 |                 method=method,
00608 |                 delta_ang=float(metadata.delta_ang),
00609 |                 reference_plus=loaded.get("siesta_plus"),
00610 |                 reference_minus=loaded.get("siesta_minus"),
00611 |                 reference_base=loaded.get("siesta_base"),
00612 |                 predicted_plus=loaded.get("ml_plus"),
00613 |                 predicted_minus=loaded.get("ml_minus"),
00614 |                 predicted_base=loaded.get("ml_base"),
00615 |                 predicted_source=source_model,
00616 |                 reference_hashes=_matrix_hashes(stencil, prefix="siesta"),
00617 |                 predicted_hashes=_matrix_hashes(stencil, prefix="ml"),
00618 |                 metadata=metadata,
00619 |             )
00620 |         row = derivative_sparse_metrics(
00621 |             pair.reference.matrix,
00622 |             pair.predicted.matrix,
00623 |             sample=metadata.sample_id,
00624 |             metadata=metadata,
00625 |             source_model=source_model,
00626 |             reference_source="siesta",
00627 |             support_threshold=support_threshold,
00628 |         )
00629 |         row.update(
00630 |             {
00631 |                 "invalid_geometry": bool(geometry_errors),
00632 |                 "geometry_validation_failed": bool(geometry_errors),
00633 |                 "geometry_issue_codes": ";".join(issue.code for issue in geometry_errors),
00634 |                 "dh_support_changed": bool(pair.diagnostics.get("plus_minus_support_changed")),
00635 |                 "reference_plus_minus_support_changed": bool(pair.diagnostics.get("reference_plus_minus_support_changed")),
00636 |                 "predicted_plus_minus_support_changed": bool(pair.diagnostics.get("predicted_plus_minus_support_changed")),
00637 |                 "reference_derivative_method": REFERENCE_DERIVATIVE_METHOD_SIESTA,
00638 |                 "predicted_derivative_method": predicted_derivative_method,
00639 |                 "reference_delta_ang": float(metadata.delta_ang),
00640 |                 "predicted_delta_ang": None if direct_prediction_mode else float(metadata.delta_ang),
00641 |                 "graph2mat_prediction_method": (
00642 |                     graph2mat_prediction_method if source_model == "graph2mat" else None
00643 |                 ),
00644 |                 "deeph_prediction_method": deeph_prediction_method if source_model == "deeph" else None,
00645 |                 "direct_prediction_path": str(direct_prediction_path) if direct_prediction_path else "",
00646 |             }
00647 |         )
00648 |         row.update(
00649 |             _deeph_direct_equivalence_fields(
00650 |                 source_model=source_model,
00651 |                 direct_prediction_mode=direct_prediction_mode,
00652 |                 direct_metadata=direct_metadata,
00653 |             )
00654 |         )
00655 |         row.update(
00656 |             {
00657 |                 key: pair.diagnostics[key]
00658 |                 for key in DERIVATIVE_SIGNAL_TO_NOISE_FIELDS
00659 |                 if key in pair.diagnostics
00660 |             }
00661 |         )
00662 |         metric_rows.append(row)
00663 |         quantile_rows.extend(
00664 |             derivative_ref_abs_quantile_metrics(
00665 |                 pair.reference.matrix,
00666 |                 pair.predicted.matrix,
00667 |                 sample=metadata.sample_id,
00668 |                 metadata=metadata,
00669 |                 source_model=source_model,
00670 |                 reference_source="siesta",
00671 |                 support_threshold=support_threshold,
00672 |             )
00673 |         )
00674 |         if row["dh_union_nnz"] == 0:
00675 |             warnings.append(
00676 |                 _discovery_error(
00677 |                     discovery,
00678 |                     "derivative_ref_abs_quantile_metrics_empty_union_support",
00679 |                     "No derivative ref-abs quantile rows written because union support is empty.",
00680 |                 )
00681 |             )
00682 |         for threshold in SUPPORT_THRESHOLDS_SWEEP:
00683 |             sweep = derivative_sparse_metrics(
00684 |                 pair.reference.matrix,
00685 |                 pair.predicted.matrix,
00686 |                 sample=metadata.sample_id,
00687 |                 metadata=metadata,
00688 |                 source_model=source_model,
00689 |                 reference_source="siesta",
00690 |                 support_threshold=threshold,
00691 |             )
00692 |             sweep_rows.append(
00693 |                 {
00694 |                     "sample": metadata.sample_id,
00695 |                     "support_threshold": threshold,
00696 |                     "dh_union_nnz": sweep["dh_union_nnz"],
00697 |                     "dh_mae_union_eV_per_Ang": sweep["dh_mae_union_eV_per_Ang"],
00698 |                     "dh_rmse_union_eV_per_Ang": sweep["dh_rmse_union_eV_per_Ang"],
00699 |                     "dh_support_precision": sweep["dh_support_precision"],
00700 |                     "dh_support_recall": sweep["dh_support_recall"],
00701 |                     "dh_support_f1": sweep["dh_support_f1"],
00702 |                 }
00703 |             )
00704 |         hermiticity_rows.append(
00705 |             {
00706 |                 "sample": metadata.sample_id,
00707 |                 "finite_difference_method": method,
00708 |                 "source_model": source_model,
00709 |                 "reference_source": "siesta",
00710 |                 "dH_ref_hermiticity_defect": pair.diagnostics["dH_ref_hermiticity_defect"],
00711 |                 "dH_pred_hermiticity_defect": pair.diagnostics["dH_pred_hermiticity_defect"],
00712 |                 "dH_hermiticity_error_delta": row["dh_hermiticity_error_delta"],
00713 |                 "finite_values": pair.diagnostics["finite_values"],
00714 |             }
00715 |         )
00716 |         status_row = _stencil_status_row(replace(discovery, stencil=stencil, issues=tuple(validation)), status="ok")
00717 |     except Exception as exc:  # Backend-specific sisl readers raise heterogeneous exceptions.
00718 |         status_row = _stencil_status_row(discovery, status="failed")
00719 |         fatal_errors.append(_discovery_error(discovery, "derivative_metric_evaluation_failed", str(exc)))
00720 |     return status_row, metric_rows, quantile_rows, sweep_rows, hermiticity_rows, geometry_row, warnings, fatal_errors
```

### `_scientific_status` — líneas 906–932

```py
00906 | def _scientific_status(
00907 |     *,
00908 |     method: str,
00909 |     diagnostic_only: bool,
00910 |     stencils_total: int,
00911 |     stencils_ok: int,
00912 |     stencils_failed: int,
00913 |     metric_rows: list[dict[str, Any]],
00914 |     fatal_errors: list[dict[str, Any]],
00915 |     deeph_equivalence: dict[str, Any] | None = None,
00916 | ) -> str:
00917 |     if diagnostic_only or method != "central" or not metric_rows:
00918 |         return "diagnostic_only"
00919 |     if deeph_equivalence and truthy(deeph_equivalence.get("deeph_diagnostic_only")):
00920 |         return "diagnostic_only"
00921 |     if stencils_total == stencils_ok and stencils_failed == 0 and not fatal_errors:
00922 |         if all(
00923 |             row.get("comparison_status") != "diagnostic_only"
00924 |             and row.get("derivative_units") == EXPECTED_DERIVATIVE_UNITS
00925 |             and row.get("finite_difference_method") == "central"
00926 |             and row.get("dh_finite_values") is True
00927 |             and _finite_or_nan(row.get("dh_hermiticity_ref")) < 1e-8
00928 |             and _finite_or_nan(row.get("dh_hermiticity_pred")) < 1e-8
00929 |             for row in metric_rows
00930 |         ):
00931 |             return "presentation_ready"
00932 |     return "diagnostic_only"
```

### `_micro_macro_domain` — líneas 942–976

```py
00942 | def _micro_macro_domain(
00943 |     metric_rows: list[dict[str, Any]],
00944 |     value_key: str,
00945 |     *,
00946 |     weight_key: str = "dh_union_nnz",
00947 |     domain_key: str = "dh_matrix_rows",
00948 | ) -> dict[str, Any]:
00949 |     """Micro (element-weighted), macro (per-snapshot) and per-domain means.
00950 | 
00951 |     Domains are the distinct structure sizes present (matrix rows), so small
00952 |     structures are visible separately instead of being drowned by large ones
00953 |     (audit Fase 12).
00954 |     """
00955 |     pairs = []
00956 |     for row in metric_rows:
00957 |         value = row.get(value_key)
00958 |         if value is None or (isinstance(value, float) and math.isnan(value)):
00959 |             continue
00960 |         weight = row.get(weight_key) or 0
00961 |         pairs.append((float(value), float(weight), row.get(domain_key)))
00962 |     if not pairs:
00963 |         return {"micro": None, "macro_snapshot": None, "by_domain": {}, "macro_domain": None}
00964 |     total_weight = sum(w for _v, w, _d in pairs)
00965 |     micro = sum(v * w for v, w, _d in pairs) / total_weight if total_weight else None
00966 |     macro = sum(v for v, _w, _d in pairs) / len(pairs)
00967 |     by_domain: dict[str, list[float]] = {}
00968 |     for value, _weight, domain in pairs:
00969 |         by_domain.setdefault(str(domain), []).append(value)
00970 |     domain_means = {domain: sum(vals) / len(vals) for domain, vals in by_domain.items()}
00971 |     return {
00972 |         "micro": micro,
00973 |         "macro_snapshot": macro,
00974 |         "by_domain": domain_means,
00975 |         "macro_domain": sum(domain_means.values()) / len(domain_means),
00976 |     }
```

### `_summary` — líneas 979–1004

```py
00979 | def _summary(
00980 |     metric_rows: list[dict[str, Any]],
00981 |     stencil_rows: list[dict[str, Any]],
00982 |     hermiticity_rows: list[dict[str, Any]],
00983 | ) -> dict[str, Any]:
00984 |     return {
00985 |         "stencils_total": len(stencil_rows),
00986 |         "metric_rows": len(metric_rows),
00987 |         "failed_stencils": len([row for row in stencil_rows if row.get("status") != "ok"]),
00988 |         "mean_dh_mae_union_eV_per_Ang": _mean(row.get("dh_mae_union_eV_per_Ang") for row in metric_rows),
00989 |         "mean_dh_rmse_union_eV_per_Ang": _mean(row.get("dh_rmse_union_eV_per_Ang") for row in metric_rows),
00990 |         "mean_dh_relative_frobenius_ref": _mean(row.get("dh_relative_frobenius_ref") for row in metric_rows),
00991 |         "dh_mae_eV_per_Ang_reductions": _micro_macro_domain(metric_rows, "dh_mae_union_eV_per_Ang"),
00992 |         "dh_rmse_eV_per_Ang_reductions": _micro_macro_domain(metric_rows, "dh_rmse_union_eV_per_Ang"),
00993 |         "dh_relative_frobenius_reductions": _micro_macro_domain(
00994 |             metric_rows, "dh_relative_frobenius_union_robust"
00995 |         ),
00996 |         "dh_normalized_frobenius_per_element_reductions": _micro_macro_domain(
00997 |             metric_rows, "dh_normalized_frobenius_per_element_eV_per_Ang"
00998 |         ),
00999 |         "dh_cosine_reductions": _micro_macro_domain(metric_rows, "dh_cosine_similarity_union"),
01000 |         "max_dh_hermiticity_ref": _max(row.get("dH_ref_hermiticity_defect") for row in hermiticity_rows),
01001 |         "max_dh_hermiticity_pred": _max(row.get("dH_pred_hermiticity_defect") for row in hermiticity_rows),
01002 |         "force_constants_used": False,
01003 |         "reference_definition": REFERENCE_DEFINITION,
01004 |     }
```

### `_derivative_group_metrics` — líneas 1007–1015

```py
01007 | def _derivative_group_metrics(metric_rows: list[dict[str, Any]], *, split: str | None = None) -> dict[str, Any]:
01008 |     rows = [{**row, **({"split": split} if split is not None and "split" not in row else {})} for row in metric_rows]
01009 |     return {
01010 |         "schema": "hamiltonian_derivative_group_metrics_v1",
01011 |         "grouping_preserves": GROUPING_PRESERVES,
01012 |         "by_atom": _aggregate_derivative_groups(rows, ["atom_index_zero_based"]),
01013 |         "by_axis": _aggregate_derivative_groups(rows, ["axis"]),
01014 |         "by_atom_axis": _aggregate_derivative_groups(rows, ["atom_index_zero_based", "axis"]),
01015 |     }
```

### `_aggregate_derivative_group` — líneas 1029–1056

```py
01029 | def _aggregate_derivative_group(keys: list[str], key: tuple[Any, ...], rows: list[dict[str, Any]]) -> dict[str, Any]:
01030 |     payload = {name: value for name, value in zip(keys, key, strict=False)}
01031 |     payload["n_stencils"] = len(rows)
01032 |     for field in GROUP_MEAN_MEDIAN_FIELDS:
01033 |         values = _finite_values(rows, field)
01034 |         payload[f"{field}_mean"] = sum(values) / len(values) if values else None
01035 |         payload[f"{field}_median"] = _median(values)
01036 |     fro_pairs = [
01037 |         (_finite_or_nan(row.get("dh_norm_error_union_fro")), _finite_or_nan(row.get("dh_norm_ref_union_fro")))
01038 |         for row in rows
01039 |         if "dh_norm_error_union_fro" in row and "dh_norm_ref_union_fro" in row
01040 |     ]
01041 |     fro_pairs = [(err, ref) for err, ref in fro_pairs if math.isfinite(err) and math.isfinite(ref)]
01042 |     if fro_pairs:
01043 |         payload["dh_relative_frobenius_union_robust_pooled"] = math.sqrt(sum(err**2 for err, _ in fro_pairs)) / (
01044 |             math.sqrt(sum(ref**2 for _, ref in fro_pairs)) + 1e-30
01045 |         )
01046 |     l1_pairs = [
01047 |         (_finite_or_nan(row.get("dh_norm_error_l1_union")), _finite_or_nan(row.get("dh_norm_ref_l1_union")))
01048 |         for row in rows
01049 |         if "dh_norm_error_l1_union" in row and "dh_norm_ref_l1_union" in row
01050 |     ]
01051 |     l1_pairs = [(err, ref) for err, ref in l1_pairs if math.isfinite(err) and math.isfinite(ref)]
01052 |     if l1_pairs:
01053 |         payload["dh_relative_l1_union_robust_pooled"] = sum(err for err, _ in l1_pairs) / (
01054 |             sum(ref for _, ref in l1_pairs) + 1e-30
01055 |         )
01056 |     return payload
```

### `_delta_stability_summary` — líneas 1101–1150

```py
01101 | def _delta_stability_summary(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
01102 |     groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
01103 |     for row in metric_rows:
01104 |         delta = _finite_or_nan(row.get("delta_ang"))
01105 |         if not math.isfinite(delta):
01106 |             continue
01107 |         groups.setdefault(_delta_stability_group_key(row), []).append(row)
01108 | 
01109 |     rows: list[dict[str, Any]] = []
01110 |     for key, group_rows in sorted(groups.items(), key=lambda item: item[0]):
01111 |         deltas = sorted({float(row.get("delta_ang")) for row in group_rows if math.isfinite(_finite_or_nan(row.get("delta_ang")))})
01112 |         if len(deltas) < 2:
01113 |             continue
01114 |         source_model, base_sample_id, atom_index_zero_based, axis, method = key
01115 |         row = {
01116 |             "source_model": source_model,
01117 |             "base_sample_id": base_sample_id,
01118 |             "atom_index_zero_based": atom_index_zero_based,
01119 |             "axis": axis,
01120 |             "finite_difference_method": method,
01121 |             "delta_count": len(deltas),
01122 |             "delta_min_ang": min(deltas),
01123 |             "delta_max_ang": max(deltas),
01124 |             "status": "available",
01125 |         }
01126 |         row.update(_range_payload(group_rows, "dh_mae_union_eV_per_Ang", "dh_mae_union_eV_per_Ang"))
01127 |         row.update(_range_payload(group_rows, "dh_rmse_union_eV_per_Ang", "dh_rmse_union_eV_per_Ang"))
01128 |         row.update(_range_payload(group_rows, "dh_relative_frobenius_ref", "dh_relative_frobenius_ref"))
01129 |         rows.append(row)
01130 | 
01131 |     pairwise_metric_rows = _delta_stability_pairwise_metric_rows(metric_rows)
01132 |     unique_deltas = sorted({float(row.get("delta_ang")) for row in metric_rows if math.isfinite(_finite_or_nan(row.get("delta_ang")))})
01133 |     if not rows:
01134 |         status = "single_delta_only" if len(unique_deltas) < 2 else "unavailable_no_matched_delta_groups"
01135 |         reason = (
01136 |             "At least two delta_ang values for the same source_model/base_sample_id/atom/axis/method are required."
01137 |             if len(unique_deltas) >= 2
01138 |             else "Fewer than two delta_ang values were found in derivative metric rows."
01139 |         )
01140 |     else:
01141 |         status = "available"
01142 |         reason = ""
01143 |     return {
01144 |         "status": status,
01145 |         "reason": reason,
01146 |         "groups_total": len(rows),
01147 |         "unique_delta_ang": unique_deltas,
01148 |         "rows": rows,
01149 |         "pairwise_metric_rows": pairwise_metric_rows,
01150 |     }
```

### `_delta_stability_convergence_summary` — líneas 1211–1236

```py
01211 | def _delta_stability_convergence_summary(
01212 |     delta_stability: dict[str, Any],
01213 |     *,
01214 |     convergence_thresholds: dict[str, Any] | None = None,
01215 | ) -> dict[str, Any]:
01216 |     available = str(delta_stability.get("status") or "").strip().lower() == "available"
01217 |     thresholds_present = bool(convergence_thresholds)
01218 |     if not thresholds_present:
01219 |         return {
01220 |             "delta_sensitivity_study_available": available,
01221 |             "delta_sensitivity_study_passed": available,
01222 |             "delta_stability_converged": None,
01223 |             "delta_stability_convergence_status": "not_evaluated_without_thresholds",
01224 |         }
01225 |     converged = delta_stability.get("converged")
01226 |     if converged is None:
01227 |         converged = str(delta_stability.get("convergence_status") or "").strip().lower() == "converged"
01228 |     convergence_status = str(delta_stability.get("convergence_status") or "").strip() or (
01229 |         "converged" if bool(converged) else "not_converged"
01230 |     )
01231 |     return {
01232 |         "delta_sensitivity_study_available": available,
01233 |         "delta_sensitivity_study_passed": available,
01234 |         "delta_stability_converged": bool(converged) if converged is not None else None,
01235 |         "delta_stability_convergence_status": convergence_status,
01236 |     }
```

### `_reference_noise_summary` — líneas 1239–1255

```py
01239 | def _reference_noise_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
01240 |     noise_rows = [
01241 |         {key: value for key, value in row.items() if str(key).startswith("reference_noise")}
01242 |         for row in rows
01243 |         if any(str(key).startswith("reference_noise") for key in row)
01244 |     ]
01245 |     if not noise_rows:
01246 |         return {
01247 |             "status": "reference_noise_unavailable",
01248 |             "reason": "No repeated SIESTA reference/noise evidence was found in derivative metric manifests.",
01249 |             "rows": [],
01250 |         }
01251 |     return {
01252 |         "status": "available",
01253 |         "reason": "",
01254 |         "rows": noise_rows,
01255 |     }
```

## `Comparison/scripts/g2m_deeph_derivative_gate_check.py` — extractos seleccionados

SHA-256 del archivo completo: `69ba4bde7e0d5e13f734560274ae9f6f4339649ae179cb59803a019e7f6e4421`

### Cabecera, imports y constantes iniciales

```py
00001 | #!/usr/bin/env python3
00002 | """Fail-closed scientific gate checker for Hamiltonian derivative comparisons."""
00003 | 
00004 | from __future__ import annotations
00005 | 
00006 | import argparse
00007 | import csv
00008 | import json
00009 | import math
00010 | from pathlib import Path
00011 | from typing import Any
00012 | 
00013 | 
00014 | SCHEMA_VERSION = "graph2mat_deeph_derivative_gate_report_v1"
00015 | EXPECTED_REFERENCE_DEFINITION = "siesta_hamiltonian_finite_difference"
00016 | EXPECTED_DERIVATIVE_UNITS = "eV/Ang"
00017 | DEFAULT_HERMITICITY_THRESHOLD = 1e-8
00018 | DEFAULT_SUPPORT_DISCONTINUITY_THRESHOLD = 1e-12
00019 | VALID_STATUSES = ("internal_diagnostic", "technical_presentation", "paper_level_candidate", "blocked")
00020 | STATUS_RANK = {status: index for index, status in enumerate(VALID_STATUSES)}
00021 | 
```

### `central_stencil_rows` — líneas 187–196

```py
00187 | def central_stencil_rows(dataset: dict[str, Any]) -> list[dict[str, str]]:
00188 |     rows = [
00189 |         row for row in dataset["stencil_rows"]
00190 |         if str(row.get("finite_difference_method") or dataset["manifest"].get("finite_difference_method") or "").strip().lower() == "central"
00191 |     ]
00192 |     if rows:
00193 |         return rows
00194 |     if str(dataset["manifest"].get("finite_difference_method") or "").strip().lower() == "central":
00195 |         return dataset["stencil_rows"]
00196 |     return []
```

### `central_metric_rows` — líneas 199–208

```py
00199 | def central_metric_rows(dataset: dict[str, Any]) -> list[dict[str, str]]:
00200 |     rows = [
00201 |         row for row in dataset["matrix_rows"]
00202 |         if str(row.get("finite_difference_method") or dataset["manifest"].get("finite_difference_method") or "").strip().lower() == "central"
00203 |     ]
00204 |     if rows:
00205 |         return rows
00206 |     if str(dataset["manifest"].get("finite_difference_method") or "").strip().lower() == "central":
00207 |         return dataset["matrix_rows"]
00208 |     return []
```

### `max_hermiticity_defect` — líneas 211–223

```py
00211 | def max_hermiticity_defect(dataset: dict[str, Any]) -> float:
00212 |     values: list[float] = []
00213 |     for row in dataset["hermiticity_rows"]:
00214 |         for key in ("dH_ref_hermiticity_defect", "dH_pred_hermiticity_defect", "dH_hermiticity_error_delta"):
00215 |             value = number(row.get(key))
00216 |             if value is not None:
00217 |                 values.append(value)
00218 |     for row in dataset["matrix_rows"]:
00219 |         for key in ("dh_hermiticity_ref", "dh_hermiticity_pred", "dh_hermiticity_error_delta"):
00220 |             value = number(row.get(key))
00221 |             if value is not None:
00222 |                 values.append(value)
00223 |     return max(values) if values else 0.0
```

### `support_discontinuity_detected` — líneas 226–236

```py
00226 | def support_discontinuity_detected(dataset: dict[str, Any], threshold: float) -> bool:
00227 |     for row in dataset["matrix_rows"]:
00228 |         if truthy(row.get("dh_support_changed")):
00229 |             return True
00230 |         if (number(row.get("dh_false_zero_rate")) or 0.0) > threshold:
00231 |             return True
00232 |         if (number(row.get("dh_false_nonzero_rate")) or 0.0) > threshold:
00233 |             return True
00234 |         if truthy(row.get("reference_plus_minus_support_changed")) or truthy(row.get("predicted_plus_minus_support_changed")):
00235 |             return True
00236 |     return False
```

### `has_consistent_required_hashes` — líneas 262–273

```py
00262 | def has_consistent_required_hashes(dataset: dict[str, Any]) -> bool:
00263 |     required = ("basis_hash", "pseudopotential_hash", "orbital_ordering_hash", "material_compatibility_hash")
00264 |     rows = rows_for_hash_checks(dataset)
00265 |     for field in required:
00266 |         if any(not str(row.get(field) or "").strip() for row in rows):
00267 |             return False
00268 |         values = unique_nonempty([row.get(field) for row in rows])
00269 |         if not values:
00270 |             return False
00271 |         if len(values) > 1:
00272 |             return False
00273 |     return True
```

### `geometry_validation_passed` — líneas 276–284

```py
00276 | def geometry_validation_passed(dataset: dict[str, Any]) -> bool:
00277 |     summary = dataset["manifest"].get("geometry_validation")
00278 |     if isinstance(summary, dict) and int_or_none(summary.get("errors")) not in (None, 0):
00279 |         return False
00280 |     if any(str(row.get("status") or "").strip().lower() == "error" for row in dataset["geometry_rows"]):
00281 |         return False
00282 |     if any(truthy(row.get("invalid_geometry")) or truthy(row.get("geometry_validation_failed")) for row in dataset["matrix_rows"]):
00283 |         return False
00284 |     return bool(summary or dataset["geometry_rows"] or dataset["matrix_rows"])
```

### `unit_metadata_explicit` — líneas 287–294

```py
00287 | def unit_metadata_explicit(dataset: dict[str, Any]) -> bool:
00288 |     if explicit_issue(dataset, ("missing_unit_metadata",)):
00289 |         return False
00290 |     for row in dataset["matrix_rows"]:
00291 |         value = str(row.get("unit_metadata_explicit") or "").strip()
00292 |         if value and not truthy(value):
00293 |             return False
00294 |     return True
```

### `delta_sensitivity_has_two_deltas` — líneas 308–321

```py
00308 | def delta_sensitivity_has_two_deltas(dataset: dict[str, Any]) -> bool:
00309 |     delta_stability = dataset.get("delta_stability") if isinstance(dataset.get("delta_stability"), dict) else {}
00310 |     unique_deltas = [
00311 |         value
00312 |         for value in (delta_stability.get("unique_delta_ang") or [])
00313 |         if number(value) is not None
00314 |     ]
00315 |     if len(set(float(value) for value in unique_deltas)) >= 2:
00316 |         return True
00317 |     for row in delta_stability.get("rows") or []:
00318 |         if int_or_none(row.get("delta_count")) and int(row.get("delta_count")) >= 2:
00319 |             return True
00320 |     metric_deltas = [number(row.get("delta_ang")) for row in central_metric_rows(dataset)]
00321 |     return len({float(value) for value in metric_deltas if value is not None}) >= 2
```

### `split_consistency_proven` — líneas 324–330

```py
00324 | def split_consistency_proven(dataset: dict[str, Any]) -> bool:
00325 |     manifest = dataset["manifest"]
00326 |     return (
00327 |         truthy(manifest.get("split_consistency_proven"))
00328 |         or truthy(manifest.get("split_metadata_verified"))
00329 |         or truthy(manifest.get("dataset_split_evidence"))
00330 |     )
```

### `deeph_autograd_equivalence_proven` — líneas 345–360

```py
00345 | def deeph_autograd_equivalence_proven(dataset: dict[str, Any]) -> bool:
00346 |     if str(dataset["model"]).strip().lower() != "deeph":
00347 |         return True
00348 |     manifest = dataset["manifest"]
00349 |     rows = dataset["matrix_rows"]
00350 |     deeph_autograd = str(manifest.get("deeph_prediction_method") or "").strip().lower() == "autograd_vectorized" or any(
00351 |         str(row.get("deeph_prediction_method") or "").strip().lower() == "autograd_vectorized"
00352 |         for row in rows
00353 |     )
00354 |     if not deeph_autograd:
00355 |         return True
00356 |     if truthy(manifest.get("deeph_diagnostic_only")):
00357 |         return False
00358 |     if "deeph_all_raw_global_equivalence_proven" in manifest:
00359 |         return truthy(manifest.get("deeph_all_raw_global_equivalence_proven"))
00360 |     return bool(rows) and all(truthy(row.get("deeph_raw_global_equivalence_proven")) for row in rows)
```

### `dataset_paper_evidence` — líneas 363–413

```py
00363 | def dataset_paper_evidence(dataset: dict[str, Any]) -> dict[str, bool]:
00364 |     manifest = dataset["manifest"]
00365 |     matrix_rows = dataset["matrix_rows"]
00366 |     delta_stability = dataset.get("delta_stability") if isinstance(dataset.get("delta_stability"), dict) else {}
00367 |     manifest_delta_stability = manifest.get("delta_stability") if isinstance(manifest.get("delta_stability"), dict) else {}
00368 |     delta_status = str(delta_stability.get("status") or manifest_delta_stability.get("status") or "").strip().lower()
00369 |     delta_available = truthy(manifest.get("delta_sensitivity_study_available")) or truthy(manifest.get("delta_sensitivity_study_passed"))
00370 |     if not delta_available:
00371 |         delta_available = delta_status == "available"
00372 |     delta_passed = truthy(manifest.get("delta_sensitivity_study_passed"))
00373 |     if "delta_sensitivity_study_passed" not in manifest:
00374 |         delta_passed = delta_available
00375 |     convergence_status = str(
00376 |         manifest.get("delta_stability_convergence_status")
00377 |         or delta_stability.get("delta_stability_convergence_status")
00378 |         or ""
00379 |     ).strip().lower()
00380 |     converged_value = manifest.get("delta_stability_converged")
00381 |     if converged_value is None:
00382 |         converged_value = delta_stability.get("delta_stability_converged")
00383 |     delta_converged = truthy(converged_value)
00384 |     reference_noise = manifest.get("reference_noise") if isinstance(manifest.get("reference_noise"), dict) else {}
00385 |     reference_noise_status = str(reference_noise.get("status") or manifest.get("reference_noise_status") or "").strip().lower()
00386 |     comparison_statuses = unique_nonempty([row.get("comparison_status") for row in matrix_rows])
00387 |     return {
00388 |         "has_non_diagnostic_comparison_rows": bool(comparison_statuses) and all(status != "diagnostic_only" for status in comparison_statuses),
00389 |         "central_only": central_only(dataset),
00390 |         "geometry_validation_passed": geometry_validation_passed(dataset),
00391 |         "unit_metadata_explicit": unit_metadata_explicit(dataset),
00392 |         "required_hashes_present_and_consistent": has_consistent_required_hashes(dataset),
00393 |         "basis_gauge_verified": truthy(manifest.get("basis_gauge_verified")) or truthy(manifest.get("basis_gauge_evidence")),
00394 |         "orbital_ordering_verified": truthy(manifest.get("orbital_ordering_verified")) or truthy(manifest.get("orbital_ordering_evidence")),
00395 |         "delta_sensitivity_study_available": delta_available,
00396 |         "delta_sensitivity_study_passed": delta_passed,
00397 |         "delta_sensitivity_has_two_deltas": delta_sensitivity_has_two_deltas(dataset),
00398 |         "delta_stability_converged": delta_converged,
00399 |         "delta_stability_convergence_status": convergence_status or "not_evaluated_without_thresholds",
00400 |         "reference_noise_evidence": truthy(manifest.get("reference_noise_verified"))
00401 |         or truthy(manifest.get("reference_noise_evidence"))
00402 |         or reference_noise_status == "available",
00403 |         "independent_dataset_metadata": truthy(manifest.get("independent_dataset_metadata"))
00404 |         or truthy(manifest.get("dataset_split_evidence"))
00405 |         or (str(manifest.get("split") or "").strip().lower() == "test" and truthy(manifest.get("split_metadata_verified"))),
00406 |         "split_consistency_proven": split_consistency_proven(dataset),
00407 |         "cross_model_equivalence_proven": truthy(manifest.get("cross_model_equivalence_proven"))
00408 |         or str(manifest.get("cross_model_equivalence_status") or "").strip().lower() == "proven",
00409 |         "deeph_autograd_equivalence_proven": deeph_autograd_equivalence_proven(dataset),
00410 |         "paper_level_candidate_requested": truthy(manifest.get("paper_level_candidate_requested"))
00411 |         or truthy(manifest.get("paper_level_evidence_complete")),
00412 |         "winner_claim_paired_gate_ok": winner_claim_requires_paired_gate(dataset),
00413 |     }
```

### `evaluate_dataset` — líneas 416–645

```py
00416 | def evaluate_dataset(
00417 |     dataset: dict[str, Any],
00418 |     *,
00419 |     hermiticity_threshold: float,
00420 |     support_discontinuity_threshold: float,
00421 | ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, bool]]:
00422 |     manifest = dataset["manifest"]
00423 |     model = dataset["model"]
00424 |     evidence_paths = list(dataset["paths"].values())
00425 |     blockers: list[dict[str, Any]] = []
00426 |     warnings: list[dict[str, Any]] = []
00427 |     central_stencils = central_stencil_rows(dataset)
00428 |     central_metrics = central_metric_rows(dataset)
00429 | 
00430 |     if truthy(manifest.get("force_constants_used")):
00431 |         blockers.append(
00432 |             gate_row(
00433 |                 "force_constants_used",
00434 |                 model=model,
00435 |                 severity="blocker",
00436 |                 status="fail",
00437 |                 blocks_status="blocked",
00438 |                 message="SIESTA force constants must not be used as the dH/dR reference.",
00439 |                 evidence_paths=evidence_paths,
00440 |             )
00441 |         )
00442 |     if str(manifest.get("reference_definition") or "").strip() != EXPECTED_REFERENCE_DEFINITION:
00443 |         blockers.append(
00444 |             gate_row(
00445 |                 "reference_definition_invalid",
00446 |                 model=model,
00447 |                 severity="blocker",
00448 |                 status="fail",
00449 |                 blocks_status="blocked",
00450 |                 message="reference_definition must be siesta_hamiltonian_finite_difference.",
00451 |                 evidence_paths=evidence_paths,
00452 |             )
00453 |         )
00454 |     if not central_stencils and not central_metrics:
00455 |         blockers.append(
00456 |             gate_row(
00457 |                 "missing_central_stencil",
00458 |                 model=model,
00459 |                 severity="blocker",
00460 |                 status="fail",
00461 |                 blocks_status="blocked",
00462 |                 message="No central finite-difference stencil is available for derivative gating.",
00463 |                 evidence_paths=evidence_paths,
00464 |             )
00465 |         )
00466 |     else:
00467 |         for row in central_stencils:
00468 |             if not str(row.get("plus_sample_id") or "").strip() or not str(row.get("minus_sample_id") or "").strip():
00469 |                 blockers.append(
00470 |                     gate_row(
00471 |                         "missing_plus_minus_pairing",
00472 |                         model=model,
00473 |                         severity="blocker",
00474 |                         status="fail",
00475 |                         blocks_status="blocked",
00476 |                         message="Central stencils require both plus and minus samples.",
00477 |                         evidence_paths=evidence_paths,
00478 |                     )
00479 |                 )
00480 |                 break
00481 |     derivative_units = unique_nonempty([manifest.get("derivative_units"), *(row.get("derivative_units") for row in dataset["matrix_rows"])])
00482 |     if any(unit != EXPECTED_DERIVATIVE_UNITS for unit in derivative_units):
00483 |         blockers.append(
00484 |             gate_row(
00485 |                 "mismatched_units",
00486 |                 model=model,
00487 |                 severity="blocker",
00488 |                 status="fail",
00489 |                 blocks_status="blocked",
00490 |                 message=f"Derivative units must be {EXPECTED_DERIVATIVE_UNITS}.",
00491 |                 evidence_paths=evidence_paths,
00492 |             )
00493 |         )
00494 |     elif explicit_issue(dataset, ("unit_mismatch", "units mismatch", "hamiltonian_units", "displacement_units", "derivative_units")):
00495 |         blockers.append(
00496 |             gate_row(
00497 |                 "mismatched_units",
00498 |                 model=model,
00499 |                 severity="blocker",
00500 |                 status="fail",
00501 |                 blocks_status="blocked",
00502 |                 message="Unit mismatch evidence was found in derivative validation diagnostics.",
00503 |                 evidence_paths=evidence_paths,
00504 |             )
00505 |         )
00506 |     delta_values = [number(row.get("delta_ang")) for row in central_metrics or central_stencils]
00507 |     if not delta_values or any(value is None or value <= 0 for value in delta_values):
00508 |         blockers.append(
00509 |             gate_row(
00510 |                 "mismatched_delta",
00511 |                 model=model,
00512 |                 severity="blocker",
00513 |                 status="fail",
00514 |                 blocks_status="blocked",
00515 |                 message="Central derivative rows must carry a positive delta_ang.",
00516 |                 evidence_paths=evidence_paths,
00517 |             )
00518 |         )
00519 |     elif explicit_issue(dataset, ("delta_mismatch", "invalid_delta", "delta_ang")):
00520 |         blockers.append(
00521 |             gate_row(
00522 |                 "mismatched_delta",
00523 |                 model=model,
00524 |                 severity="blocker",
00525 |                 status="fail",
00526 |                 blocks_status="blocked",
00527 |                 message="Derivative validation reported mismatched or invalid delta metadata.",
00528 |                 evidence_paths=evidence_paths,
00529 |             )
00530 |         )
00531 |     atom_indices = [int_or_none(row.get("atom_index_zero_based")) for row in central_metrics or central_stencils]
00532 |     if not atom_indices or any(index is None or index < 0 for index in atom_indices):
00533 |         blockers.append(
00534 |             gate_row(
00535 |                 "atom_indexing_missing_or_inconsistent",
00536 |                 model=model,
00537 |                 severity="blocker",
00538 |                 status="fail",
00539 |                 blocks_status="blocked",
00540 |                 message="atom_index_zero_based is missing or inconsistent in derivative rows.",
00541 |                 evidence_paths=evidence_paths,
00542 |             )
00543 |         )
00544 |     elif explicit_issue(dataset, ("invalid_atom_index", "atom_index_mismatch", "atom_index_zero_based")):
00545 |         blockers.append(
00546 |             gate_row(
00547 |                 "atom_indexing_missing_or_inconsistent",
00548 |                 model=model,
00549 |                 severity="blocker",
00550 |                 status="fail",
00551 |                 blocks_status="blocked",
00552 |                 message="Derivative validation reported atom indexing inconsistencies.",
00553 |                 evidence_paths=evidence_paths,
00554 |             )
00555 |         )
00556 |     if explicit_issue(dataset, ("shape_mismatch", "matrix shape", "shapes disagree", "matching shapes")):
00557 |         blockers.append(
00558 |             gate_row(
00559 |                 "mismatched_shapes",
00560 |                 model=model,
00561 |                 severity="blocker",
00562 |                 status="fail",
00563 |                 blocks_status="blocked",
00564 |                 message="Derivative validation reported mismatched matrix shapes.",
00565 |                 evidence_paths=evidence_paths,
00566 |             )
00567 |         )
00568 |     if explicit_issue(dataset, ("orbital_ordering", "missing_required_metadata"), require_row_failure=True):
00569 |         blockers.append(
00570 |             gate_row(
00571 |                 "orbital_ordering_metadata_missing_or_inconsistent",
00572 |                 model=model,
00573 |                 severity="blocker",
00574 |                 status="fail",
00575 |                 blocks_status="blocked",
00576 |                 message="Derivative validation reported missing or inconsistent orbital ordering metadata.",
00577 |                 evidence_paths=evidence_paths,
00578 |             )
00579 |         )
00580 |     max_herm = max_hermiticity_defect(dataset)
00581 |     if max_herm > hermiticity_threshold:
00582 |         blockers.append(
00583 |             gate_row(
00584 |                 "high_hermiticity_defect",
00585 |                 model=model,
00586 |                 severity="blocker",
00587 |                 status="fail",
00588 |                 blocks_status="blocked",
00589 |                 message=f"Hermiticity defect {max_herm:.3e} exceeds the threshold {hermiticity_threshold:.3e}.",
00590 |                 evidence_paths=evidence_paths,
00591 |             )
00592 |         )
00593 |     if support_discontinuity_detected(dataset, support_discontinuity_threshold):
00594 |         blockers.append(
00595 |             gate_row(
00596 |                 "support_pattern_discontinuity",
00597 |                 model=model,
00598 |                 severity="blocker",
00599 |                 status="fail",
00600 |                 blocks_status="blocked",
00601 |                 message="Support pattern discontinuity or false-zero/false-nonzero activity exceeded the configured threshold.",
00602 |                 evidence_paths=evidence_paths,
00603 |             )
00604 |         )
00605 |     if explicit_issue(dataset, ("neighbor_list_hash", "neighbor list", "sparsity_pattern_hash", "sparsity pattern")):
00606 |         warnings.append(
00607 |             warning_row(
00608 |                 "neighbor_or_sparsity_warning",
00609 |                 model=model,
00610 |                 severity="severe",
00611 |                 message="Neighbor-list or sparsity diagnostics were reported; interpret derivative comparisons conservatively.",
00612 |                 evidence_paths=evidence_paths,
00613 |             )
00614 |         )
00615 |     if truthy(manifest.get("diagnostic_only_requested")) or str(manifest.get("scientific_status") or "").strip().lower() == "diagnostic_only":
00616 |         warnings.append(
00617 |             warning_row(
00618 |                 "diagnostic_only_requested",
00619 |                 model=model,
00620 |                 message="This derivative evaluation was marked diagnostic-only in its manifest.",
00621 |                 evidence_paths=evidence_paths,
00622 |             )
00623 |         )
00624 |     if str(model).strip().lower() == "deeph" and not deeph_autograd_equivalence_proven(dataset):
00625 |         warnings.append(
00626 |             warning_row(
00627 |                 "deeph_autograd_equivalence_not_proven",
00628 |                 model=model,
00629 |                 severity="severe",
00630 |                 message="DeepH autograd derivative claims remain diagnostic-only until raw/global equivalence is proven.",
00631 |                 evidence_paths=evidence_paths,
00632 |             )
00633 |         )
00634 |     for fatal in manifest.get("fatal_errors") or []:
00635 |         if isinstance(fatal, dict):
00636 |             warnings.append(
00637 |                 warning_row(
00638 |                     f"fatal_error_{fatal.get('kind') or 'reported'}",
00639 |                     model=model,
00640 |                     severity="severe",
00641 |                     message=str(fatal.get("message") or fatal.get("kind") or "Derivative fatal error reported."),
00642 |                     evidence_paths=evidence_paths,
00643 |                 )
00644 |             )
00645 |     return blockers, warnings, dataset_paper_evidence(dataset)
```

### `overall_status` — líneas 648–688

```py
00648 | def overall_status(
00649 |     datasets: list[dict[str, Any]],
00650 |     blockers: list[dict[str, Any]],
00651 |     warnings: list[dict[str, Any]],
00652 |     paper_evidence: dict[str, bool],
00653 | ) -> str:
00654 |     if any(row.get("blocks_status") == "blocked" for row in blockers):
00655 |         return "blocked"
00656 |     if any(
00657 |         truthy(dataset["manifest"].get("diagnostic_only_requested"))
00658 |         or str(dataset["manifest"].get("scientific_status") or "").strip().lower() == "diagnostic_only"
00659 |         for dataset in datasets
00660 |     ):
00661 |         return "internal_diagnostic"
00662 |     if any(not evidence["has_non_diagnostic_comparison_rows"] for evidence in paper_evidence.values()):
00663 |         return "internal_diagnostic"
00664 |     if any(not evidence.get("deeph_autograd_equivalence_proven", True) for evidence in paper_evidence.values()):
00665 |         return "internal_diagnostic"
00666 |     paper_blocked = [
00667 |         not info["central_only"]
00668 |         or not info["geometry_validation_passed"]
00669 |         or not info["unit_metadata_explicit"]
00670 |         or not info["required_hashes_present_and_consistent"]
00671 |         or not info["basis_gauge_verified"]
00672 |         or not info["orbital_ordering_verified"]
00673 |         or not info["delta_sensitivity_has_two_deltas"]
00674 |         or not info["delta_sensitivity_study_passed"]
00675 |         or not info["delta_stability_converged"]
00676 |         or not info["reference_noise_evidence"]
00677 |         or not info["independent_dataset_metadata"]
00678 |         or not info["split_consistency_proven"]
00679 |         or not info["winner_claim_paired_gate_ok"]
00680 |         for info in paper_evidence.values()
00681 |     ]
00682 |     if len(datasets) > 1 and not all(info["cross_model_equivalence_proven"] for info in paper_evidence.values()):
00683 |         paper_blocked.append(True)
00684 |     if not all(info["paper_level_candidate_requested"] for info in paper_evidence.values()):
00685 |         paper_blocked.append(True)
00686 |     if any(paper_blocked):
00687 |         return "technical_presentation"
00688 |     return "paper_level_candidate"
```

### `allowed_claims_for_status` — líneas 691–706

```py
00691 | def allowed_claims_for_status(status: str) -> list[str]:
00692 |     if status == "blocked":
00693 |         return ["No scientific derivative comparison claim is allowed; only blocker diagnostics may be discussed."]
00694 |     claims = [
00695 |         "dH/dR refers to derivatives of Hamiltonian matrix elements with respect to Cartesian atomic displacement.",
00696 |         "The reference is finite differences of SIESTA Hamiltonians, not force constants.",
00697 |     ]
00698 |     if status == "internal_diagnostic":
00699 |         claims.append("Derivative errors may be shown as internal diagnostic-only evidence without ranking or winner claims.")
00700 |         return claims
00701 |     claims.append("Derivative metrics may be presented as technical finite-difference diagnostics against SIESTA Hamiltonians.")
00702 |     if status == "paper_level_candidate":
00703 |         claims.append("Paper-level candidate wording is allowed only as a candidate status pending independent review.")
00704 |     else:
00705 |         claims.append("No paper-level or winner claim is allowed from this report.")
00706 |     return claims
```

### `blocked_claims_for_status` — líneas 709–720

```py
00709 | def blocked_claims_for_status(status: str, *, multiple_models: bool, paper_evidence: dict[str, bool]) -> list[str]:
00710 |     claims = [
00711 |         "Do not state or imply that SIESTA force constants, dynamical matrices, or phonons were used as the dH/dR reference.",
00712 |         "Do not declare a derivative winner by default.",
00713 |     ]
00714 |     if status in {"blocked", "internal_diagnostic", "technical_presentation"}:
00715 |         claims.append("Do not claim paper-level validation of Hamiltonian derivatives.")
00716 |     if status in {"blocked", "internal_diagnostic"}:
00717 |         claims.append("Do not frame the derivative comparison as more than diagnostic evidence.")
00718 |     if multiple_models and not all(info["cross_model_equivalence_proven"] for info in paper_evidence.values()):
00719 |         claims.append("Do not claim Graph2Mat vs DeepH paper-level comparability; cross-model equivalence remains diagnostic-only.")
00720 |     return claims
```

### `build_derivative_gate_report` — líneas 791–1033

```py
00791 | def build_derivative_gate_report(
00792 |     *,
00793 |     derivative_roots: list[Path],
00794 |     run_root: Path | None = None,
00795 |     hermiticity_threshold: float = DEFAULT_HERMITICITY_THRESHOLD,
00796 |     support_discontinuity_threshold: float = DEFAULT_SUPPORT_DISCONTINUITY_THRESHOLD,
00797 | ) -> dict[str, Any]:
00798 |     datasets = [load_dataset(root) for root in derivative_roots]
00799 |     blockers: list[dict[str, Any]] = []
00800 |     warnings: list[dict[str, Any]] = []
00801 |     paper_evidence: dict[str, bool] = {}
00802 |     for dataset in datasets:
00803 |         dataset_blockers, dataset_warnings, evidence = evaluate_dataset(
00804 |             dataset,
00805 |             hermiticity_threshold=hermiticity_threshold,
00806 |             support_discontinuity_threshold=support_discontinuity_threshold,
00807 |         )
00808 |         blockers.extend(dataset_blockers)
00809 |         warnings.extend(dataset_warnings)
00810 |         paper_evidence[dataset["model"]] = evidence
00811 | 
00812 |     status = overall_status(datasets, blockers, warnings, paper_evidence)
00813 |     if len(datasets) > 1 and not all(info["cross_model_equivalence_proven"] for info in paper_evidence.values()):
00814 |         blockers.append(
00815 |             gate_row(
00816 |                 "cross_model_equivalence_diagnostic_only",
00817 |                 severity="blocker",
00818 |                 status="fail",
00819 |                 blocks_status="paper_level_candidate",
00820 |                 claim_scope="paper_level_only",
00821 |                 message="Graph2Mat/DeepH cross-model equivalence remains diagnostic-only without explicit proof.",
00822 |                 evidence_paths=[dataset["paths"]["manifest"] for dataset in datasets],
00823 |             )
00824 |         )
00825 |     for model, info in paper_evidence.items():
00826 |         dataset = next(dataset for dataset in datasets if dataset["model"] == model)
00827 |         manifest_path = dataset["paths"]["manifest"]
00828 |         if not info["central_only"]:
00829 |             blockers.append(
00830 |                 gate_row(
00831 |                     "paper_level_central_only_required",
00832 |                     model=model,
00833 |                     severity="blocker",
00834 |                     status="fail",
00835 |                     blocks_status="paper_level_candidate",
00836 |                     claim_scope="paper_level_only",
00837 |                     message="Paper-level derivative candidate status requires central finite-difference stencils only.",
00838 |                     evidence_paths=[manifest_path],
00839 |                 )
00840 |             )
00841 |         if not info["geometry_validation_passed"]:
00842 |             blockers.append(
00843 |                 gate_row(
00844 |                     "paper_level_geometry_validation_failed",
00845 |                     model=model,
00846 |                     severity="blocker",
00847 |                     status="fail",
00848 |                     blocks_status="paper_level_candidate",
00849 |                     claim_scope="paper_level_only",
00850 |                     message="Paper-level derivative candidate status requires geometry validation to pass.",
00851 |                     evidence_paths=[dataset["paths"]["geometry_validation"], manifest_path],
00852 |                 )
00853 |             )
00854 |         if not info["unit_metadata_explicit"]:
00855 |             blockers.append(
00856 |                 gate_row(
00857 |                     "paper_level_unit_metadata_missing",
00858 |                     model=model,
00859 |                     severity="blocker",
00860 |                     status="fail",
00861 |                     blocks_status="paper_level_candidate",
00862 |                     claim_scope="paper_level_only",
00863 |                     message="Paper-level derivative candidate status requires explicit Hamiltonian, displacement, and derivative units.",
00864 |                     evidence_paths=[dataset["paths"]["matrix_metrics"], dataset["paths"]["stencil_status"]],
00865 |                 )
00866 |             )
00867 |         if not info["required_hashes_present_and_consistent"]:
00868 |             blockers.append(
00869 |                 gate_row(
00870 |                     "paper_level_required_hashes_missing_or_inconsistent",
00871 |                     model=model,
00872 |                     severity="blocker",
00873 |                     status="fail",
00874 |                     blocks_status="paper_level_candidate",
00875 |                     claim_scope="paper_level_only",
00876 |                     message="Paper-level derivative candidate status requires consistent basis, pseudopotential, orbital-ordering, and material-compatibility hashes.",
00877 |                     evidence_paths=[dataset["paths"]["matrix_metrics"], dataset["paths"]["stencil_status"]],
00878 |                 )
00879 |             )
00880 |         if not info["delta_sensitivity_study_available"]:
00881 |             blockers.append(
00882 |                 gate_row(
00883 |                     "paper_level_delta_sweep_missing",
00884 |                     model=model,
00885 |                     severity="blocker",
00886 |                     status="fail",
00887 |                     blocks_status="paper_level_candidate",
00888 |                     claim_scope="paper_level_only",
00889 |                     message="Paper-level candidate status requires a delta sensitivity study.",
00890 |                     evidence_paths=[manifest_path],
00891 |                 )
00892 |             )
00893 |         elif not info["delta_sensitivity_study_passed"]:
00894 |             blockers.append(
00895 |                 gate_row(
00896 |                     "paper_level_delta_sweep_failed",
00897 |                     model=model,
00898 |                     severity="blocker",
00899 |                     status="fail",
00900 |                     blocks_status="paper_level_candidate",
00901 |                     claim_scope="paper_level_only",
00902 |                     message="Paper-level derivative candidate status requires the configured delta sensitivity criterion to pass.",
00903 |                     evidence_paths=[dataset["paths"]["delta_stability"], manifest_path],
00904 |                 )
00905 |             )
00906 |         elif not info["delta_sensitivity_has_two_deltas"]:
00907 |             blockers.append(
00908 |                 gate_row(
00909 |                     "paper_level_delta_sweep_needs_two_deltas",
00910 |                     model=model,
00911 |                     severity="blocker",
00912 |                     status="fail",
00913 |                     blocks_status="paper_level_candidate",
00914 |                     claim_scope="paper_level_only",
00915 |                     message="Paper-level derivative candidate status requires a delta sensitivity study with at least two delta values.",
00916 |                     evidence_paths=[dataset["paths"]["delta_stability"], manifest_path],
00917 |                 )
00918 |             )
00919 |         elif not info["delta_stability_converged"]:
00920 |             blockers.append(
00921 |                 gate_row(
00922 |                     "paper_level_delta_stability_not_converged",
00923 |                     model=model,
00924 |                     severity="blocker",
00925 |                     status="fail",
00926 |                     blocks_status="paper_level_candidate",
00927 |                     claim_scope="paper_level_only",
00928 |                     message="Paper-level candidate status requires documented delta stability convergence thresholds and convergence evidence.",
00929 |                     evidence_paths=[manifest_path],
00930 |                 )
00931 |             )
00932 |         if not info["basis_gauge_verified"] or not info["orbital_ordering_verified"]:
00933 |             blockers.append(
00934 |                 gate_row(
00935 |                     "paper_level_ordering_or_gauge_evidence_missing",
00936 |                     model=model,
00937 |                     severity="blocker",
00938 |                     status="fail",
00939 |                     blocks_status="paper_level_candidate",
00940 |                     claim_scope="paper_level_only",
00941 |                     message="Paper-level candidate status requires explicit orbital-ordering and basis/gauge evidence.",
00942 |                     evidence_paths=[manifest_path],
00943 |                 )
00944 |             )
00945 |         if not info["reference_noise_evidence"]:
00946 |             blockers.append(
00947 |                 gate_row(
00948 |                     "paper_level_reference_noise_missing",
00949 |                     model=model,
00950 |                     severity="blocker",
00951 |                     status="fail",
00952 |                     blocks_status="paper_level_candidate",
00953 |                     claim_scope="paper_level_only",
00954 |                     message="Paper-level candidate status requires repeated-reference/noise evidence.",
00955 |                     evidence_paths=[manifest_path],
00956 |                 )
00957 |             )
00958 |         if not info["independent_dataset_metadata"]:
00959 |             blockers.append(
00960 |                 gate_row(
00961 |                     "paper_level_independent_dataset_metadata_missing",
00962 |                     model=model,
00963 |                     severity="blocker",
00964 |                     status="fail",
00965 |                     blocks_status="paper_level_candidate",
00966 |                     claim_scope="paper_level_only",
00967 |                     message="Paper-level candidate status requires independent dataset/split metadata.",
00968 |                     evidence_paths=[manifest_path],
00969 |                 )
00970 |             )
00971 |         if not info["split_consistency_proven"]:
00972 |             blockers.append(
00973 |                 gate_row(
00974 |                     "paper_level_split_consistency_missing",
00975 |                     model=model,
00976 |                     severity="blocker",
00977 |                     status="fail",
00978 |                     blocks_status="paper_level_candidate",
00979 |                     claim_scope="paper_level_only",
00980 |                     message="Paper-level derivative candidate status requires proven split consistency.",
00981 |                     evidence_paths=[manifest_path],
00982 |                 )
00983 |             )
00984 |         if not info["winner_claim_paired_gate_ok"]:
00985 |             blockers.append(
00986 |                 gate_row(
00987 |                     "paper_level_winner_claim_requires_paired_gate",
00988 |                     model=model,
00989 |                     severity="blocker",
00990 |                     status="fail",
00991 |                     blocks_status="paper_level_candidate",
00992 |                     claim_scope="paper_level_only",
00993 |                     message="Derivative winner claims require an explicit paired-comparison gate.",
00994 |                     evidence_paths=[manifest_path],
00995 |                 )
00996 |             )
00997 | 
00998 |     report = {
00999 |         "schema_version": SCHEMA_VERSION,
01000 |         "scientific_status": status,
01001 |         "allowed_claims": allowed_claims_for_status(status),
01002 |         "blocked_claims": blocked_claims_for_status(status, multiple_models=len(datasets) > 1, paper_evidence=paper_evidence),
01003 |         "blockers": blockers,
01004 |         "warnings": warnings,
01005 |         "recommended_next_steps": recommended_next_steps(
01006 |             status,
01007 |             blockers,
01008 |             paper_evidence,
01009 |             multiple_models=len(datasets) > 1,
01010 |         ),
01011 |         "evidence_paths": {
01012 |             "run_root": str(run_root) if run_root is not None else "",
01013 |             "derivative_roots": [str(dataset["root"]) for dataset in datasets],
01014 |             "manifests": [str(dataset["paths"]["manifest"]) for dataset in datasets],
01015 |             "matrix_metrics": [str(dataset["paths"]["matrix_metrics"]) for dataset in datasets],
01016 |             "stencil_status": [str(dataset["paths"]["stencil_status"]) for dataset in datasets],
01017 |             "hermiticity": [str(dataset["paths"]["hermiticity"]) for dataset in datasets],
01018 |             "delta_stability": [str(dataset["paths"]["delta_stability"]) for dataset in datasets],
01019 |         },
01020 |         "datasets": [
01021 |             {
01022 |                 "model": dataset["model"],
01023 |                 "root": str(dataset["root"]),
01024 |                 "manifest_scientific_status": dataset["manifest"].get("scientific_status"),
01025 |                 "stencils_total": dataset["manifest"].get("stencils_total"),
01026 |                 "stencils_ok": dataset["manifest"].get("stencils_ok"),
01027 |                 "stencils_failed": dataset["manifest"].get("stencils_failed"),
01028 |                 "paper_evidence": paper_evidence[dataset["model"]],
01029 |             }
01030 |             for dataset in datasets
01031 |         ],
01032 |     }
01033 |     return report
```
