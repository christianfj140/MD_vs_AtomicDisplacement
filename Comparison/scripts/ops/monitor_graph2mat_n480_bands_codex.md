Supervisa y recupera exclusivamente la campaña proyectada N=480:

- Repositorio: `/home/christian/repositorios/MD_vs_AtomicDisplacement`
- Configuración: `Comparison/config/graphene_hbn_magic_angle_spectral_campaign.json`
- Objetivo único: Graph2Mat, N=480, seed 0, 256 estados con proyecciones
  Mulliken compatibles con S.
- Flujo secuencial existente: smoke Γ–K–M–Γ (31 k), K′–Γ–K (3 k),
  producción Γ–K–M–Γ (151 k) y DOS/PDOS 6×6 (36 k).
- Servicios: `moire-projected-n480-smoke.service` y
  `moire-projected-n480-followup.service`.
- Estado final:
  `Comparison/results/graphene_hbn_magic_angle_spectral/spectra/projected_followup_status.json`.
- Auditoría final:
  `Comparison/results/graphene_hbn_magic_angle_spectral/summary/projected_acceptance_report.json`.

Actúa, no te limites a informar, pero respeta estas reglas:

1. Inspecciona primero procesos, `status.json`, manifiesto, logs, RAM, disco y
   temperatura. No interrumpas ni reinicies un solver sano.
2. No lances DeepH, otros tamaños, otras seeds ni el DOS 24×24. Respeta el
   orden escalonado y nunca ejecutes dos solvers simultáneamente.
3. Conserva `gpu_cudss`, sin fallback denso, con `OPENBLAS_NUM_THREADS=8`,
   `OMP_NUM_THREADS=8`, `MKL_NUM_THREADS=8`, `CUDSS_HOST_THREADS=1` y límite
   de VRAM de 28 GiB.
4. Guardrails obligatorios:
   - nunca continúes si el disco libre es menor de 12% (el requisito del usuario
     es no bajar del 10%);
   - no reinicies hasta tener al menos 20 GiB de RAM disponible;
   - no reinicies con CPU >=80 °C ni GPU >=75 °C; el solver aborta antes de
     85 °C/80 °C respectivamente;
   - conserva el monitor de ejecución que exige al menos 20 GiB disponibles.
5. Si el proceso está sano, verifica que los guardrails siguen activos y termina.
6. Si una etapa terminó correctamente, deja que el servicio followup avance.
   Si todo terminó, verifica el informe de aceptación y la publicación en la UI.
7. Si se detuvo:
   - determina la causa exacta usando logs y manifiestos;
   - si faltan recursos, espera a la siguiente ejecución del cron sin reiniciar;
   - si es un fallo transitorio o un bug reproducible, aplica la corrección mínima,
     ejecuta las pruebas relevantes y reinicia únicamente el servicio de la etapa
     pendiente; usa su `ExecStart` existente como fuente del comando;
   - confirma que el nuevo proceso sigue activo y sano antes de terminar.
8. Preserva todos los cambios preexistentes del worktree. No borres resultados,
   no uses comandos destructivos y no hagas commits, pushes ni cambios externos.
9. No falsifiques progreso ni interpoles bandas. Cada etapa del solver no tiene
   checkpoint reanudable: no reinicies un proceso sano y reutiliza únicamente
   manifests completos y validados.
10. No esperes a que termine el cálculo largo. Sal tras verificar salud o
    recuperación. No generes subagentes.

Deja una conclusión breve con estado, diagnóstico, acción tomada y recursos.
