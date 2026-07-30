Supervisa y recupera exclusivamente la campaña de bandas:

- Repositorio: `/home/christian/repositorios/MD_vs_AtomicDisplacement`
- Configuración: `Comparison/config/graphene_hbn_magic_angle_spectral_campaign.json`
- Objetivo único: Graph2Mat, N=480, seed 0, Tier B, 16 bandas y 8 puntos por
  segmento (22 evaluaciones Γ–K–M–Γ).
- Resultado:
  `Comparison/results/graphene_hbn_magic_angle_spectral/spectra/graph2mat/n480/seed0/tier_b/solver_manifest.json`
- Comando de recuperación:
  `.venv/bin/python Comparison/scripts/run_graphene_hbn_moire_spectral_campaign.py solve-bands --config Comparison/config/graphene_hbn_magic_angle_spectral_campaign.json`

Actúa, no te limites a informar, pero respeta estas reglas:

1. Inspecciona primero procesos, `status.json`, manifiesto, logs, RAM, disco y
   temperatura. No interrumpas ni reinicies un solver sano.
2. No lances DeepH, otros tamaños, otras seeds ni DOS.
3. Mantén PARDISO out-of-core. No uses un fallback denso.
   Mantén `OMP_NUM_THREADS=8` y `MKL_NUM_THREADS=8`.
4. Guardrails obligatorios:
   - nunca continúes si el disco libre es menor de 12% (el requisito del usuario
     es no bajar del 10%);
   - no reinicies hasta tener al menos 20 GiB de RAM disponible;
   - no reinicies con CPU >=80 °C; el solver debe abortar a 90 °C;
   - conserva el monitor de ejecución que aborta con menos de 8 GiB de RAM.
5. Si el proceso está sano, verifica que los guardrails siguen activos y termina.
6. Si terminó correctamente, ejecuta `aggregate` sólo si hace falta publicar el
   resultado en la UI y termina.
7. Si se detuvo:
   - determina la causa exacta usando logs y manifiestos;
   - si faltan recursos, espera a la siguiente ejecución del cron sin reiniciar;
   - si es un fallo transitorio o un bug reproducible, aplica la corrección mínima,
     ejecuta las pruebas relevantes y relanza el comando de recuperación;
   - confirma que el nuevo proceso sigue activo y sano antes de terminar.
8. Preserva todos los cambios preexistentes del worktree. No borres resultados,
   no uses comandos destructivos y no hagas commits, pushes ni cambios externos.
9. No falsifiques progreso ni interpoles bandas. El solver no tiene checkpoint
   por punto k: un relanzamiento empieza el Tier B desde el principio.
10. No esperes a que termine el cálculo largo. Sal tras verificar salud o
    recuperación. No generes subagentes.

Deja una conclusión breve con estado, diagnóstico, acción tomada y recursos.
