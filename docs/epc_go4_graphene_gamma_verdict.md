# C21 / GO-4 - veredicto graphene Gamma

C21 publica el veredicto terminal ya decidido por C19/C20; no recalcula fisica.
El productor es `Comparison/scripts/evaluate_epc_metrics.py` y el artifact
canonico es `Comparison/results/epc/gamma_epc/graphene/go4_verdict.json`.

| nivel | veredicto |
| --- | --- |
| derivative-level | PASS |
| finite-PAO EPC | **NO_GO:model_accuracy** |
| full-KS EPC | **BLOCKED: Delta_out** |

Los gates de referencia read-only, FC-vs-FD E2g, JVP, frozen, respuesta de base
v6 y GO-6 pasan. El unico gate que falla es `tau_model_adequacy`, atribuido a la
capa `model`: `tau_model=1.794210486335622` frente al umbral `0.05`.
La cota sí transfiere al split de validación, pero no cubre el split de
resultado (`max(error)=3.0730359706 > tau_model`). Por ello
`quantitative_for_g=false`.

`NO_GO:model_accuracy` termina la validacion finite-PAO. No autoriza retraining
ni fine tuning; ese trabajo queda sólo recomendado. Tampoco promociona el
resultado a full-KS. La ausencia de `Delta_out` se conserva
como limitacion explicita, no como fallo de C19/C20.

```bash
.venv/bin/python Comparison/scripts/evaluate_epc_metrics.py --verify
.venv/bin/python -m pytest tests/test_evaluate_epc_metrics.py
```
