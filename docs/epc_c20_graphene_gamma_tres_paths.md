# C19/C20 - graphene Gamma read-only

`Comparison/scripts/run_graphene_gamma_epc_paths.py` ejecuta C19/C20 como consumidor
read-only. No llama a `run_campaign`, `execute_run`, `siesta_e2g_campaign` ni genera
geometrias o calculos SIESTA. Si el preflight no encuentra un artifact exacto, termina
con `BLOCKED: missing_artifact`.

## Preflight

El artifact `Comparison/results/epc/gamma_epc/graphene/read_only_preflight.json`
registra:

| contrato | resultado |
| --- | --- |
| central H/S, geometria, celda, especies, base, ORB_INDX, gauge y runtime | PASS |
| `dHSdR.nc`: I=1,2; alpha=1,2,3 para delta=0.005, 0.01 y 0.02 Ang | completo |
| E2g FD | 12/12 TSHS originales |
| `A_I` v6 | 5/5 PASS |
| C18 / GO-2 / GO-6 / C14C-v2 | PASS / PASS / PASS / PASS |
| `missing_artifacts` | `[]` |

Los hashes centrales fijados son:

- base FDF: `c8b67d7572ec18a18a99bf0c73de65f96ba49e4411001bf4ad9224f90a43c958`;
- equilibrium TSHS: `72793773bd138da3e9a35ad9623dcb3f0cab30d4222c95bbd69086f2837a02de`;
- ORB_INDX: `dacebf0f6182b2a3d653bb906cdb99614dbc1f1a0928f98f320936674d31d2c5`;
- C.ion.xml: `6740d9f56df9f2d42ff27e6a7abd9b7b0224a49cbda58b52b7f38136bcfc8b6f`.

El error maximo al reconstruir la geometria central desde los desplazamientos es
`0.0 Ang`. La referencia FC E2g se ensambla linealmente desde el `dHSdR.nc`
atom-resolved certificado y se contrasta con los 12 TSHS FD: 4/4 comparaciones
`D_H`/`D_S` pasan y son `noise_limited`.

## C19

La respuesta completa usa, para cada direccion,

```text
S_R_full(R) = S_R_inter(R) + A_I(R)
S_L_full(R) = S_L_inter(R) - A_I(R)
Delta_PAO_cov = D_H - S_L S^-1 H - H S^-1 S_R
```

Todos los paths comparten el mismo `A_I` v6, los mismos eigenspaces y la
normalizacion C18. El observable se publica como **finite-PAO covariant
electron-phonon matrix element** (`g_PAO`), nunca como full-KS EPC.

## C20

Norma de Frobenius de `g_PAO` en eV para el bloque preregistrado:

| k | direccion | SIESTA | Graph2Mat JVP | Graph2Mat frozen |
| --- | --- | ---: | ---: | ---: |
| `k_generic_1` | longitudinal | 0.595780 | 1.559154 | 1.547142 |
| `k_generic_1` | transverse | 0.355312 | 0.490046 | 0.490955 |
| `K` | longitudinal | 0.494479 | 1.546636 | 1.552938 |
| `K` | transverse | 0.630004 | 2.479190 | 2.465772 |

Los valores singulares y los bloques `G/g` completos estan en
`graphene_gamma_epc_g_blocks.json`. `tau_num` y `tau_backend` pasan. La cota
calibrada transfiere a validacion, pero no cubre el split de resultado:
`max(error)=3.0730359706 > tau_model=1.794210486335622`. Además, `tau_model`
supera el umbral de adecuacion `0.05`; por ambas razones el checkpoint no es
cuantitativo para `g`. El detalle FC-vs-FD queda persistido en
`embedded_read_only_split_fc_fd_v2.json`.

Resultado terminal:

```text
GO4_finite_PAO=NO_GO:model_accuracy
full_KS=BLOCKED: Delta_out
```

Los checks congelados `uniform_translation_null` y
`symmetry_of_the_doublet_at_K` permanecen registrados como hallazgos de alcance
no bloqueantes; no cambian el veredicto read-only.

## Reproducir

```bash
.venv/bin/python Comparison/scripts/run_graphene_gamma_epc_paths.py --preflight-only
.venv/bin/python Comparison/scripts/run_graphene_gamma_epc_paths.py
.venv/bin/python -m pytest tests/test_run_graphene_gamma_epc_paths.py
```
