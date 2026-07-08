# Goal: corregir la derivada autograd dH/dR de DeepH (Fase 1 rota) y cerrar los huecos de testing

## Contexto verificado (no re-derivar, ya lo comprobé empíricamente)

Un agente anterior implementó una ruta opt-in `derivative.deeph_prediction_method =
"finite_difference" | "autograd_vectorized"` en dos repos:

- `/home/christian/repositorios/DeepH-pack` (repo git independiente, NO submódulo):
  cambios sin commitear en `deeph/inference/pred_ham.py` y `deeph/scripts/inference.py`.
  Añadieron forward-mode AD (`torch.autograd.forward_ad`, alias `fwAD`) dentro de
  `predict_with_grad()` para rellenar `hamiltonians_grad_pred.h5`, que antes se
  llenaba con `np.nan` (stub sin terminar).
- `/home/christian/repositorios/MD_vs_AtomicDisplacement` (repo principal, cambios
  sin commitear): nuevo `Comparison/scripts/run_deeph_autograd_derivative_predictions.py`,
  constantes en `hamiltonian_derivative_stencil.py`, wiring en `g2m_deeph_runner.py`,
  integración en `evaluate_hamiltonian_derivative_metrics.py` y gate honesto en
  `g2m_deeph_derivative_gate_check.py`.

**El wiring del lado MD_vs_AtomicDisplacement (constantes, runner, evaluador, gate,
197 tests) está bien hecho y correctamente testeado. NO lo toques salvo lo indicado
en la Fase 3 de este prompt.**

**El problema real está en DeepH-pack: la derivada calculada NO es la derivada
real.** Lo verifiqué así:

1. `predict_with_grad` exige `assert kernel.config.getboolean('graph', 'new_sp',
   fallback=False)` — un assert preexistente, no tocado por el parche. NINGÚN
   modelo entrenado en este entorno usa `new_sp=True` (grep en todos los
   `config.ini` de checkpoints reales: cero coincidencias). Sin esto, la ruta
   autograd no puede correr contra ningún checkpoint de producción.
2. Entrené un modelo mínimo compatible clonando un config existente:
   ```bash
   cd /home/christian/repositorios/DeepH-pack
   # config base: cualquier train/config.ini real bajo
   # Comparison/results/.../DH-DERIV-PLOT-SMOKE-N10/deeph/train/config.ini
   # (8 muestras procesadas, epochs=1, create_from_dft=True) — clónalo y en la
   # sección [graph] cambia new_sp=False -> new_sp=True, apunta graph_dir/save_dir
   # a un directorio nuevo, device=cpu, disable_cuda=True, tb_writer=False.
   .venv/bin/deeph-train --config <config_clonado_con_new_sp_true.ini>
Tarda segundos (8 muestras, 1 época). Esto SÍ produce un best_model.pt
compatible con predict_with_grad.
3. Con ese modelo, ejecuté predict_with_grad sobre una estructura base real
(.../DH-DERIV-PLOT-SMOKE-N10/deeph/inference/000006_md_9, copiada a un
directorio de trabajo con lat.dat, site_positions.dat, orbital_types.dat,
overlaps.h5, rc.h5, element.dat) pidiendo solo atom_indices=[0],    axis_indices=[0]. Comparé el resultado (hamiltonians_grad_pred.h5) contra
finite-difference de predict_with_grad llamado con atom_indices=[],    axis_indices=[] (que solo evalúa el valor, ya rotado a base global — mismo
camino de rotación que el gradiente, para comparar manzanas con manzanas)
sobre copias con site_positions.dat perturbado en ±δ en (átomo 0, eje x).
4. Resultado: error relativo (norma de Frobenius sobre los 74 bloques) ≈1.0
(100%) y ESTABLE en δ = 1e-2, 1e-3, 1e-4, 1e-5, 1e-6. Esto descarta error
de truncamiento de FD o no-suavidad del modelo (ambos darían una tendencia
monótona con δ; aquí no la hay). Es un bug real de la implementación.
5. Aislé un único bloque (edge 0, salida cruda del modelo antes de rotar, y
después de rotate_kernel.rotate_openmx_H) reconstruyendo la misma lógica
a mano con fwAD.dual_level()/make_dual/unpack_dual: error relativo
elemento a elemento ~30%, pero la proyección de suma sale ~156x más pequeña
en autograd que en FD (-0.00026 vs -0.0405). Esto apunta a pérdida
PARCIAL de gradiente en algún tramo, no a un error trivial de signo o de
indexado.

Candidatos a revisar para el root cause (ninguno confirmado como LA causa,
son pistas, no conclusiones — no asumas cuál es sin verificar)
deeph/preprocess/get_rc.py:124 (aprox.): neighbours_i.dists = torch.tensor( neighbours_i.dists, dtype=cart_coords.dtype) reconstruye un tensor nuevo a partir de una lista de tensores — esto rompe el grafo diferenciable/dual de esa variable. Verifica con cuidado si dists (una vez roto) se usa en algo más que el orden de torch.sort (uso puramente discreto, inofensivo) o si se filtra a algún cálculo de VALOR que sí debería llevar gradiente.
El patrón edge_specs en pred_ham.py: se construye UNA VEZ desde la geometría base (real_index/imag_index/key_str por edge) y se reutiliza dentro de forward_hamiltonian_blocks(coords), que a su vez recalcula get_rc/get_graph desde cero en cada llamada (incluida cada evaluación fwAD). Verifica que el ORDEN de edges producido por get_graph sea IDÉNTICO entre la geometría base y la geometría dual/perturbada (debería serlo, ya que depende solo de las claves fijas de overlaps.h5, no de los valores de posición) — pero confírmalo explícitamente, no lo asumas. Instrumenta comparando batch.edge_index/batch.edge_attr[:, 4:10] (los R-vectors, fijos) entre la llamada base y una llamada con coords dual.
Cobertura de forward-mode AD en operaciones de agregación: scatter_add (usado en el forward del modelo, deeph/model.py) y las operaciones de deeph/rotate.py (Rotate.rotate_openmx_H, que usa e3nn.o3.matrix_to_angles y matrices de Wigner-D). Algunas operaciones de PyTorch tienen fórmulas de forward-mode AD incompletas o silenciosamente incorrectas para casos poco comunes; comprueba aislando la rotación (rotate_openmx_H) con un input sintético simple bajo fwAD.dual_level() y comparando contra FD directamente, sin pasar por el modelo ni por get_rc/get_graph.
Revisa también si torch.autograd.forward_ad interactúa mal con .to(device) /.clone() esparcidos por get_rc/get_graph (llamadas que podrían desprender el tangente silenciosamente en vez de fallar con un error claro).
Regla de metodología (aplícala en cualquier test que escribas)
Nunca valides "autograd vs FD" con un solo δ. Usa al menos 3 valores de δ que
difieran en órdenes de magnitud (p. ej. 1e-3, 1e-4, 1e-5). Si autograd es
correcto, el error relativo debe ser pequeño y estable (o mejorar) al bajar δ
hasta el punto donde domina el ruido de float32; si autograd tiene un bug,
verás lo que yo vi: error ~100% sin tendencia. Un test que solo prueba un δ
fijo puede pasar por casualidad y no detectar este tipo de bug.

Reglas de alcance (no te salgas de esto)
Solo DeepH clásico. No toques DeepH-E3, no añadas JAX.
No toques Graph2Mat ni su ruta autograd (graph2mat_autograd_derivatives.py, run_graph2mat_autograd_derivative_predictions.py) — ya está cerrada y correcta.
No toques SIESTA ni el finite-difference legacy de DeepH (predict() sin grad, el flujo deeph_prediction_method="finite_difference" actual) — debe seguir funcionando exactamente igual.
No toques el wiring ya correcto en evaluate_hamiltonian_derivative_metrics.py, g2m_deeph_derivative_gate_check.py, g2m_deeph_runner.py, hamiltonian_derivative_stencil.py salvo que la Fase 3/4 de abajo lo requiera explícitamente.
Ladder de simplicidad: el fix debe ser el cambio mínimo dentro de pred_ham.py/get_rc.py/graph.py/rotate.py que restaure gradiente correcto — no reescribas la ruta autograd desde cero, no introduzcas una abstracción nueva si arreglar la línea que corta el gradiente basta.
DeepH-pack no tiene infraestructura de test (no hay tests/, no hay pytest instalado en su .venv, no hay CI). El test de regresión que añadas ahí debe ser un script standalone basado en assert con un bloque if __name__ == "__main__":, no un test de pytest — no instales pytest solo para esto.
Fase 0 — Reproducir el bug como test que falla (antes de tocar nada)
En /home/christian/repositorios/DeepH-pack, clona un config de entrenamiento real pequeño (busca uno con pocas muestras procesadas y epochs bajo, p. ej. bajo Comparison/results/*/deeph/train/config.ini en el repo MD, o usa work/result/graphene_smoke/*/config.ini dentro de DeepH-pack), cambia new_sp=False a new_sp=True en [graph], ajusta save_dir/graph_dir a un directorio nuevo, device=cpu, disable_cuda=True, tb_writer=False. Entrena con deeph-train --config ... (debería tardar segundos).
Escribe un script deeph/inference/_sanity_autograd_vs_fd.py (o ubicación equivalente, standalone, sin pytest) que:
copie una estructura de inferencia real completa (necesita lat.dat, site_positions.dat, orbital_types.dat, element.dat, overlaps.h5, rc.h5) a tres directorios de trabajo (base, +δ, -δ) para un (átomo, eje) elegido,
llame a predict_with_grad(..., atom_indices=[atom], axis_indices=[axis]) sobre el directorio base,
llame a predict_with_grad(..., atom_indices=[], axis_indices=[]) sobre +δ y -δ (esto solo evalúa el valor ya rotado a base global, sin gastar tiempo en direcciones que no necesitas, y usa el MISMO camino de rotación que el gradiente),
calcule FD = (H(+δ) - H(-δ)) / (2δ) por bloque desde hamiltonians_pred.h5, compare contra hamiltonians_grad_pred.h5[...,atom,axis],
repita para al menos 3 valores de δ (regla de metodología de arriba) y falle (assert) si el error relativo no es pequeño y estable.
Confirma que este script FALLA hoy (reproduce mi hallazgo). Si por algún motivo no falla en tu entorno, documenta la discrepancia con mi hallazgo antes de continuar — no asumas que ya está arreglado.
Fase 1 — Root-cause y fix mínimo (en DeepH-pack)
Usa el script de la Fase 0 y los candidatos listados arriba para localizar el punto exacto donde el tangente/gradiente se pierde o se corrompe. Instrumenta por etapas (igual que hice yo): distancia cruda -> edge_attr -> features angulares (sub_edge_ang) -> salida cruda del modelo -> bloque rotado. Compara cada etapa contra FD de esa misma etapa hasta encontrar dónde diverge.
Aplica el fix mínimo. Candidatos de arreglo típicos para este tipo de bug: sustituir una reconstrucción de tensor que rompe el grafo (torch.tensor( lista_de_tensores)) por torch.stack(lista) cuando la variable en cuestión sí necesita llevar gradiente; asegurar que ninguna llamada .to(device) o .clone() producida dentro de la closure diferenciable desprenda el tangente silenciosamente; si el problema está en cobertura de forward-mode AD de alguna operación de rotate.py/scatter_add, considera si conviene usar reverse-mode clásico (torch.autograd.grad, ya usado y validado en la ruta Graph2Mat) en vez de fwAD si resulta más robusto aquí — no hay obligación de mantener forward-mode si reverse-mode es más fiable y suficientemente rápido para el tamaño de problema real (n_atoms pequeño por dirección, pero puede haber muchas direcciones pedidas: si reverse-mode por dirección resulta caro, evalúa banda por chunks de direcciones en vez de una por una, sin sobre-diseñar).
Corre el script de la Fase 0 hasta que pase con error relativo pequeño y convergente (no solo "menor que antes" — debe demostrar la tendencia de convergencia al bajar δ).
Verifica que el finite-difference legacy (predict() sin grad) y el resto de predict_with_grad para modelos new_sp=False (debe seguir fallando el assert exactamente igual que antes, no lo cambies) permanecen intactos.
Fase 2 — Precondición fail-closed para new_sp=True
Ahora mismo, si alguien intenta autograd_vectorized con un modelo
new_sp=False, el fallo es un AssertionError desnudo sin contexto. Mejora
esto mínimamente en predict_with_grad (o en el punto de entrada de
deeph/scripts/inference.py cuando with_grad=True): si new_sp no es
True, lanza un error claro tipo "predict_with_grad requiere un modelo
entrenado con graph.new_sp=True; este checkpoint fue entrenado con
new_sp=False, reentrena o usa finite_difference". No conviertas esto en un
sistema de validación general — es una comprobación puntual.

Fase 3 — Test de regresión permanente en DeepH-pack
Convierte el script de la Fase 0 (ya arreglado y en verde) en un test de
regresión permanente bajo DeepH-pack/tests/test_autograd_vs_finite_difference.py
(o ubicación que sigas, standalone con assert + if __name__ == "__main__",
sin pytest). Debe:

entrenar (o reutilizar si ya existe) el modelo smoke new_sp=True mínimo,
correr la comparación autograd vs FD con barrido de δ,
fallar con un mensaje claro si el error relativo no converge. Este es exactamente el test que el prompt original pedía y que faltaba — por eso el bug pasó desapercibido. No lo omitas.
Fase 4 — Tests que faltan en MD_vs_AtomicDisplacement
Comparison/scripts/run_deeph_autograd_derivative_predictions.py (nuevo, 727
líneas) no tiene NINGÚN test dedicado — solo se testea indirectamente el
wiring del runner/evaluador con fixtures fabricadas a mano. Añade
tests/test_run_deeph_autograd_derivative_predictions.py cubriendo, con un
DeepH CLI fake (subprocess stub, sin DeepH real, siguiendo el patrón de
tests/test_run_hamiltonian_derivative_predictions.py si ya fabrica algo
similar para Graph2Mat/DeepH finite-difference):

collect_derivative_requests: agrupa correctamente pares (átomo, eje) por estructura base, falla si falta la estructura base, respeta filtros --atoms/--axes/--base-sample-id.
_select_gradient_block: valida shape (rows, cols, atoms, axes), falla con índices fuera de rango.
Flujo completo con un hamiltonians_grad_pred.h5 fake (bloques conocidos, no NaN) verificando que el .npz/.json de salida tiene los metadatos correctos (predicted_derivative_method, deeph_prediction_method, predicted_delta_ang=None, campos de equivalencia de _deeph_base_equivalence_fields).
Fallo cerrado (error en el CSV de status, no excepción sin capturar) si deeph-preprocess o deeph-inference fake devuelven returncode != 0, o si falta hamiltonians_grad_pred.h5.
Criterio de éxito global
El script/test de la Fase 0 pasa con evidencia de convergencia real (no solo un número menor).
predict() legacy (finite-difference) y SIESTA no cambian de comportamiento.
Un modelo new_sp=False sigue fallando con autograd_vectorized, pero ahora con un mensaje claro en vez de un AssertionError desnudo.
Existe un test de regresión permanente en DeepH-pack (standalone, sin pytest) que habría atrapado el bug original.
Existe un test dedicado para run_deeph_autograd_derivative_predictions.py en MD_vs_AtomicDisplacement.
Los 197 tests existentes en MD_vs_AtomicDisplacement (tests/ test_hamiltonian_derivative_direct_prediction.py, tests/test_g2m_deeph_runner.py, tests/test_g2m_deeph_derivative_gate_check.py, tests/test_run_hamiltonian_derivative_predictions.py) siguen pasando.
No se ha tocado Graph2Mat, SIESTA, DeepH-E3, ni el wiring MD ya correcto (constantes, evaluador, gate) salvo lo estrictamente necesario para pasar el mensaje de error de la Fase 2.
Trabaja por fases, corre el test más pequeño que pruebe cada fase antes de
avanzar a la siguiente. Si en la Fase 1 no logras converger tras un esfuerzo
razonable de root-cause, documenta exactamente dónde se corta el gradiente
(con evidencia numérica, no especulación) y deja la ruta autograd_vectorized
bloqueada explícitamente (error claro, no resultados silenciosamente
incorrectos) en vez de forzar un "arreglo" que solo oculte el síntoma.