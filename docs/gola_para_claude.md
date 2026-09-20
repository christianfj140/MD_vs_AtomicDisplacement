Quiero completar y mejorar los resultaods de mi reposiotorio, para eso te dejo un plan que quiero que implementes.

Este primer plan es la version general: """Las dos respuestas convergen en una campaña bastante simple. No ampliaría las 40 mejores combinaciones ni añadiría N=12 todavía.

## Objetivo

Obtener tres resultados defendibles:

1. Curvas de aprendizaje limpias para `N={4,8,16,32,64}`.
2. Frontera coste–precisión.
3. Comparación final con MD sobre exactamente las mismas estructuras.

## Fase 0 — Hacer comparables los resultados actuales

Antes de entrenar nada:

- Separar claramente el H-MAE antiguo del pilot (~279–290 meV) y el de producción.
- Tratar las métricas actuales de `6x6` como validación, no como test final.
- Construir un único conjunto común de desarrollo para `6x6`, idealmente con 48 estructuras ya etiquetadas y sin solapamiento con entrenamiento.
- Reevaluar en él los 80 checkpoints existentes.
- Confirmar que se usa el checkpoint con mejor `val_loss`, no simplemente el último.
- Inventariar los frames MD disponibles para `w90` y `6x6`.

Resultado: ranking corregido sin nuevos entrenamientos ni SIESTA, salvo que no existan suficientes estructuras independientes para el desarrollo común.

## Fase 1 — Curvas de aprendizaje

### Selección

Elegir seis recetas por sistema:

- Tres `sobol_sparse`.
- Tres `random_cartesian`.

Por cada familia:

- Dos de las mejores según el desarrollo común, con distinta dimensión o dominio.
- Un control que cubra una región diferente, por ejemplo 3D/dominio alto o 1D/dominio bajo.

No seleccionar únicamente los ganadores absolutos.

### Entrenamiento

Para cada receta:

- Un único pool maestro N=64.
- Misma semilla de muestreo.
- Prefijos exactos `N={4,8,16,32,64}`.
- Misma validación, arquitectura y política de checkpoint.
- Una semilla de entrenamiento durante el screening.

Esto supone:

- 6 recetas × 5 tamaños × 2 sistemas = **60 entrenamientos**.
- **Cero SIESTA nuevo** si los pools N=64 existentes pueden reutilizarse.

Reentrenaría también N=16 y N=64 para que toda la curva tenga exactamente la misma procedencia.

### Latin hypercube

No construiría una curva LHS porque no es anidada.

Conservaría únicamente:

- La mejor LHS N=64 de `w90`.
- La mejor LHS N=64 de `6x6`.
- Tres semillas de entrenamiento para cada una.

Total: **6 entrenamientos adicionales**.

## Fase 2 — Confirmación de finalistas

Seleccionar tres finalistas por sistema usando solamente desarrollo.

Para cada finalista conservar:

- `N=64`.
- El menor `N*` cuyo error esté dentro del 5% del resultado N=64.

Añadir dos semillas de entrenamiento a las ya existentes:

- 3 finalistas × 2 tamaños × 2 semillas nuevas × 2 sistemas
- **24 entrenamientos**.

Total hasta aquí: **90 entrenamientos nuevos**, casi todos reutilizando etiquetas existentes.

Reglas de parada:

- No añadir N=12 salvo que aparezca una caída clara entre N=8 y N=16.
- Detener el crecimiento si la mejora 32→64 es menor que la variación entre semillas y menor del 5%.
- Abandonar la rama LHS si sus tres semillas quedan claramente dominadas.
- No incorporar todavía active learning, FPS, fonones ni DP-GEN.

## Fase 3 — Comparación final y MD

Congelar los modelos antes de crear o abrir el test final.

Por sistema:

- 48 estructuras sintéticas comunes:
  - amplitudes 0.02, 0.04, 0.06, 0.08, 0.10 y 0.12 Å;
  - cobertura de las dimensionalidades;
  - zonas in-domain, frontera y OOD.
- 16 frames MD independientes y decorrelacionados.

Máximo: **64 etiquetas finales por sistema**, 128 cálculos SIESTA nuevos.

Si no existen frames MD `6x6`, se debe:

- hacer la comparación diseñada-versus-MD solamente para `w90`;
- presentar `6x6` como estudio de escalado y construcción del dataset;
- no afirmar todavía equivalencia con MD para `6x6`.

El baseline MD debe usar:

- misma arquitectura;
- mismos hiperparámetros;
- tres semillas;
- misma base orbital y Hamiltoniano;
- mismo test final;
- misma métrica y política de checkpoint.

El rango histórico 62.6–69.1 meV puede mostrarse como referencia previa, pero no como prueba de equivalencia.

## Métricas

Primaria:

- H-MAE calculado por estructura y después agregado.

Secundarias:

- H-RMSE.
- Relative Frobenius.
- H-MAE por tipo de bloque orbital.
- Hermiticidad como control de consistencia.
- Bandas/DOS solo para los finalistas.

No compararía directamente el error espectral Γ de `w90` y `6x6` debido al plegamiento de bandas.

## Coste–precisión

Mantener dos fronteras:

- Coste incremental: lo que cuesta ampliar ahora el repositorio.
- Coste reproducible: lo que costaría generar el resultado desde cero.

Registrar para cada punto:

- estructuras SIESTA únicas;
- CPU·h de SIESTA;
- GPU·h de entrenamiento;
- época del mejor checkpoint;
- memoria máxima.

No usar únicamente N como coste, porque una estructura de 72 átomos no cuesta ni aporta lo mismo que una de dos.

## Resultados para presentar al jefe

Prepararía cuatro figuras principales:

1. H-MAE frente a N, en escala log₂, separando `w90` y `6x6`.
2. Frontera de Pareto: CPU·h SIESTA frente a H-MAE.
3. Comparación emparejada del mejor dataset diseñado frente a MD.
4. Heatmap de error según dominio de entrenamiento y amplitud de test.

Y una tabla final:

| Sistema | Estrategia | N mínimo | H-MAE | Rel. Frobenius | CPU·h SIESTA | GPU·h | Diferencia vs MD |
|---|---|---:|---:|---:|---:|---:|---:|

## Orden práctico

1. Reevaluar los checkpoints existentes sobre desarrollo común.
2. Ejecutar las 60 curvas Sobol/random.
3. Ejecutar las 6 réplicas LHS.
4. Elegir seis finalistas totales.
5. Ejecutar las 24 réplicas de confirmación.
6. Entrenar o reevaluar el baseline MD.
7. Crear el test final.
8. Evaluar una sola vez y producir las cuatro figuras.

Cambios científicos mínimos: un test común, pools anidados y tres semillas solo para finalistas. Todo lo demás puede esperar."""


mientras que esta que te voy a a`psar es la version mas detalalda del mismo plan: """Entiendo que quieres el **plan anterior mucho más detallado y ejecutable paso a paso**, manteniendo el foco científico y evitando ingeniería innecesaria.

# Plan científico detallado

## Resultado final buscado

Al terminar deberías poder responder, por separado para `w90` y `6x6`:

1. ¿Cómo disminuye el error al aumentar `N_train`?
2. ¿Cuál es el menor dataset que obtiene un resultado próximo al de N=64?
3. ¿Qué familia, Sobol o random, ofrece mejor coste–precisión?
4. ¿El resultado se mantiene con distintas semillas de entrenamiento?
5. ¿El dataset diseñado se acerca a uno construido con MD sobre el mismo test?
6. ¿Cuánto cuesta generar y entrenar cada alternativa?
7. ¿Qué configuraciones forman la frontera de Pareto?

No intentaría todavía demostrar:

- que una familia es universalmente mejor;
- que `w90` y `6x6` tienen errores directamente comparables;
- que el método generaliza a otros materiales;
- que LHS tiene una curva de aprendizaje anidada;
- que active learning o fonones son necesarios.

---

# Fase 0 — Congelar las preguntas científicas

Antes de ejecutar nada, escribir una especificación de una página.

## 0.1. Pregunta primaria

Para cada sistema por separado:

> ¿Cuál es el menor número de estructuras de entrenamiento, dentro de N={4,8,16,32,64}, que alcanza una precisión próxima a la obtenida con N=64?

## 0.2. Pregunta secundaria

> ¿Sobol o random_cartesian ofrece mejor H-MAE al mismo coste SIESTA?

## 0.3. Pregunta MD

> ¿El mejor dataset diseñado es no inferior a un dataset MD de igual tamaño y evaluado sobre exactamente las mismas estructuras?

## 0.4. Métrica primaria

Usar:

\[
E = \frac{1}{N_\mathrm{test}}
\sum_{i=1}^{N_\mathrm{test}}
\operatorname{MAE}\left(\hat H_i,H_i^\mathrm{SIESTA}\right)
\]

Es importante:

- calcular primero el MAE dentro de cada estructura;
- después promediar entre estructuras;
- no tratar cada elemento de la matriz como una observación independiente.

Unidad: meV por elemento de Hamiltoniano.

## 0.5. Métricas secundarias

Mantener únicamente:

- H-RMSE;
- relative Frobenius;
- hermiticidad;
- tiempo de entrenamiento;
- mejor época;
- coste SIESTA.

Bandas y DOS se calcularán únicamente para finalistas.

## 0.6. Regla provisional de saturación

Durante desarrollo, definir:

\[
N^* =
\min \left\{
N:
E(N)\leq1.05\,E(64)
\right\}
\]

Es decir, el menor N cuyo error no sea más de un 5% peor que N=64.

Este 5% sirve para localizar la saturación de la curva. No es todavía el margen físico de equivalencia con MD.

## Salida de esta fase

Un pequeño documento con:

- preguntas;
- métrica primaria;
- tamaños N;
- regla de selección;
- tests;
- semillas;
- criterios de parada.

Una vez escrito, no debe cambiarse mirando los resultados.

---

# Fase 1 — Ordenar los resultados existentes

El objetivo es corregir la interpretación antes de gastar GPU o SIESTA.

## 1.1. Separar pilot y producción

Construir una tabla con una fila por receta y columnas diferentes:

| Campo | Significado |
|---|---|
| `pilot_H_MAE_meV` | Resultado antiguo usado para ordenar las 80 recetas |
| `production_H_MAE_meV` | Resultado del entrenamiento con arquitectura grande |
| `system` | `w90` o `6x6` |
| `family` | Sobol, random o LHS |
| `dim` | 1D_in, 1D_z, 2D_in o 3D |
| `R_train` | 0.03, 0.05, 0.08 o 0.12 Å |
| `N_train` | 16 o 64 |
| `sampler_seed` | Semilla que generó las estructuras |
| `training_seed` | Semilla del modelo |
| `evaluation_role` | pilot, development o final |
| `evaluation_manifest` | Test concreto utilizado |

El valor cercano a 279–290 meV no debe aparecer como resultado del modelo de producción.

## 1.2. Revisar los pools de entrenamiento

Para cada receta Sobol/random N=64:

1. Obtener los hashes de sus 64 estructuras.
2. Confirmar que no hay duplicados.
3. Generar conceptualmente los prefijos:
   - primeros 4;
   - primeros 8;
   - primeros 16;
   - primeros 32;
   - los 64 completos.
4. Verificar mediante hashes:

\[
D_4\subset D_8\subset D_{16}\subset D_{32}\subset D_{64}
\]

5. Comprobar que todos esos puntos tienen etiqueta SIESTA disponible.

Si esto se cumple, las nuevas curvas no necesitan cálculos SIESTA de entrenamiento.

## 1.3. Ignorar los N=16 incompatibles

Si una receta N=16 existente:

- usa otra `sampler_seed`;
- no es prefijo del N=64;
- utiliza otro pool;

se conserva como resultado histórico, pero no se usa en la nueva curva causal.

El N=16 se vuelve a entrenar usando el prefijo exacto del pool N=64.

## 1.4. Verificar el checkpoint

Para cada entrenamiento, distinguir:

- último checkpoint;
- mejor checkpoint según `val_loss`;
- época del mejor checkpoint;
- época de parada.

La predicción debe usar el mejor checkpoint de validación.

No hace falta refactorizar el entrenamiento: solo asegurar que el checkpoint seleccionado es realmente el mejor.

## Salida de esta fase

- Una tabla limpia de los 160 resultados actuales.
- Lista de pools N=64 reutilizables.
- Lista de recetas descartadas por procedencia incompatible.
- Confirmación de que los prefijos no requieren nuevo SIESTA.

---

# Fase 2 — Construir un desarrollo común

Esta es la corrección científica más importante para `6x6`.

## 2.1. Desarrollo común de `w90`

Mantener el conjunto común existente de 24 estructuras si:

- es idéntico para todos los modelos;
- no pertenece a ningún entrenamiento;
- no se utilizó para producir los pools;
- su procedencia SIESTA está completa.

No hace falta crear otro si estas condiciones se cumplen.

## 2.2. Desarrollo común de `6x6`

Las ocho estructuras actuales de cada receta no sirven para comparar recetas porque:

- cambian con familia, dimensión y dominio;
- se usan como validación para early stopping;
- después se vuelven a medir como si fueran test.

Construir `common_dev_6x6_v1` con 48 estructuras.

### Estratificación sugerida

Cuatro dimensionalidades:

- 1D_in;
- 1D_z;
- 2D_in;
- 3D.

Cuatro regiones de amplitud real:

- `0 < R ≤ 0.03 Å`;
- `0.03 < R ≤ 0.05 Å`;
- `0.05 < R ≤ 0.08 Å`;
- `0.08 < R ≤ 0.12 Å`.

Tres estructuras por combinación:

\[
4\text{ dimensiones}
\times
4\text{ regiones}
\times
3\text{ réplicas}
=
48
\]

## 2.3. Reutilizar etiquetas existentes

Antes de lanzar SIESTA:

1. Unir todas las estructuras `6x6` ya etiquetadas.
2. Deduplicarlas mediante hash geométrico.
3. Eliminar cualquier estructura usada para entrenar cualquiera de las recetas candidatas.
4. Seleccionar 48 por estratos, sin usar el error del modelo como criterio.
5. Lanzar SIESTA únicamente para los estratos que no puedan completarse.

## 2.4. Uso correcto

El desarrollo común sirve para:

- ordenar las recetas;
- seleccionar celdas;
- elegir N*;
- aplicar early stopping si se decide reutilizarlo como validación.

No sirve para el claim final.

## 2.5. Reevaluación

Ejecutar predicción sobre:

- 80 checkpoints `w90`;
- 80 checkpoints `6x6`.

No hay nuevos entrenamientos.

## Salida de esta fase

Dos tablas comparables internamente:

- ranking `w90` sobre un desarrollo común;
- ranking `6x6` sobre un desarrollo común.

No se mezclan sus valores absolutos en un ranking único.

---

# Fase 3 — Seleccionar las recetas de las curvas

Se seleccionan seis recetas por sistema.

Total:

- seis para `w90`;
- seis para `6x6`.

Pueden coincidir algunas, pero no es obligatorio.

## 3.1. Selección Sobol

Elegir tres recetas:

### Sobol A — mejor resultado

La receta Sobol con menor H-MAE en desarrollo común.

### Sobol B — segundo candidato independiente

La mejor receta Sobol que tenga:

- una dimensionalidad diferente de Sobol A; o
- un dominio diferente de Sobol A.

Se debe evitar que A y B sean prácticamente la misma receta.

### Sobol C — control de cobertura

Seleccionar antes de mirar más métricas una receta que cubra una región diferente.

Por ejemplo:

- si A y B son 2D/3D, elegir un control 1D;
- si A y B usan R=0.03–0.05, elegir un control R=0.08;
- si A y B son dominios altos, elegir dominio bajo.

El control no debe ser necesariamente el peor; debe representar una región que de otro modo quedaría sin estudiar.

## 3.2. Selección random

Aplicar exactamente la misma regla:

- Random A: mejor;
- Random B: mejor con distinta dimensión o R;
- Random C: control de cobertura.

## 3.3. Bloquear selección

Guardar un manifiesto con:

- las seis recetas;
- motivo de selección;
- pool N=64;
- sampler seed;
- dimensiones;
- dominio;
- ranking previo;
- hashes del desarrollo común.

Después no cambiar estas seis recetas según los nuevos resultados.

## Salida de esta fase

Una matriz como:

| Sistema | Receta | Familia | Dim | R | Función |
|---|---|---|---|---:|---|
| w90 | S-W-A | Sobol | … | … | mejor |
| w90 | S-W-B | Sobol | … | … | alternativa |
| w90 | S-W-C | Sobol | … | … | control |
| w90 | R-W-A | Random | … | … | mejor |
| … | … | … | … | … | … |
| 6x6 | R-6-C | Random | … | … | control |

---

# Fase 4 — Generar las curvas de aprendizaje

## 4.1. Tamaños

Usar:

\[
N=\{4,8,16,32,64\}
\]

No usar N=12 inicialmente.

Motivos:

- Sobol funciona naturalmente bien con potencias de dos;
- cubre uniformemente el eje log₂;
- reduce entrenamientos;
- permite localizar saturación;
- N=12 solo es útil si aparece una rodilla entre 8 y 16.

## 4.2. Pool maestro

Para cada receta:

1. Cargar un único pool N=64.
2. Mantener la misma `sampler_seed`.
3. Crear los datasets como prefijos:

```text
N4  = pool[0:4]
N8  = pool[0:8]
N16 = pool[0:16]
N32 = pool[0:32]
N64 = pool[0:64]
```

No regenerar cada N por separado.

## 4.3. Configuración de entrenamiento

Mantener fija:

- arquitectura Graph2Mat;
- loss;
- learning rate;
- batch size;
- precisión;
- máximo de 600 épocas;
- early stopping con patience 80;
- validación común;
- semilla de entrenamiento 0;
- política de checkpoint.

La única variable debe ser N.

## 4.4. Número de entrenamientos

\[
6\text{ recetas}
\times
5\text{ tamaños}
\times
2\text{ sistemas}
=
60
\]

Aunque existan checkpoints N=16/N=64, conviene repetirlos bajo la misma procedencia y selección de checkpoint.

## 4.5. Datos registrados por ejecución

Guardar:

- sistema;
- receta;
- familia;
- dimensión;
- R;
- N;
- sampler seed;
- training seed;
- pool hash;
- hashes de los índices usados;
- checkpoint;
- mejor época;
- época final;
- H-MAE de desarrollo;
- H-RMSE;
- relative Frobenius;
- hermiticidad;
- tiempo de entrenamiento;
- GPU·h;
- memoria máxima.

## 4.6. Análisis inicial

Por cada receta, dibujar:

\[
E(N)
\quad\text{frente a}\quad
N
\]

con eje N en log₂.

Calcular:

\[
g_{N\rightarrow2N}
=
\frac{E(N)-E(2N)}{E(N)}
\]

para:

- 4→8;
- 8→16;
- 16→32;
- 32→64.

## 4.7. Decisión sobre N=12

Añadir N=12 solamente si:

- la caída 8→16 es mucho mayor que 4→8 y 16→32;
- o N=8 es claramente insuficiente pero N=16 ya está saturado;
- o existe un límite práctico alrededor de 12 cálculos.

En ese caso se añade N=12 únicamente a las recetas que muestran esa rodilla.

## Salida de esta fase

- 12 curvas de aprendizaje.
- N* provisional de cada receta.
- Ganancia marginal por duplicación.
- Primera frontera coste–precisión.

---

# Fase 5 — Tratar Latin Hypercube por separado

LHS no entra en las curvas anteriores.

## 5.1. Selección

Por sistema:

- elegir la mejor receta LHS N=64 según desarrollo común.

## 5.2. Repeticiones

Entrenar con:

\[
\text{training seed}=\{0,1,2\}
\]

manteniendo exactamente el mismo dataset N=64.

Total máximo:

\[
1\text{ LHS}
\times
3\text{ seeds}
\times
2\text{ sistemas}
=
6
\]

Si un checkpoint existente cumple la nueva política, puede reutilizarse.

## 5.3. Pregunta

No preguntar “¿cómo escala LHS con N?”.

Preguntar:

> ¿La aparente ventaja de LHS N=64 se mantiene frente a la variabilidad de entrenamiento?

## 5.4. Criterio

LHS pasa a confirmación si:

- sus tres entrenamientos son válidos;
- su H-MAE medio queda dentro del 5% del mejor Sobol/random N=64;
- o mejora claramente su coste–precisión.

Si no, la rama LHS termina aquí.

---

# Fase 6 — Elegir finalistas

Seleccionar tres finalistas por sistema.

## 6.1. Finalista de precisión

La receta con menor H-MAE de desarrollo en N=64.

## 6.2. Finalista eficiente

La receta con menor coste que cumpla:

\[
E(N^*)\le1.05\,E(64)
\]

## 6.3. Finalista alternativo

Una receta de una familia o dominio distinto para evitar que los tres finalistas representen el mismo tipo de dataset.

Puede ser:

- Sobol si los otros son random;
- random si los otros son Sobol;
- LHS N=64 si fue estable;
- un control con mejor generalización OOD.

## 6.4. Evitar selección redundante

Si precisión y eficiencia corresponden a la misma receta:

- se conserva como un solo finalista;
- se elige el siguiente candidato Pareto.

## Salida de esta fase

Tres finalistas por sistema, cada uno con:

- receta;
- N*;
- N=64;
- motivo de selección;
- coste;
- error;
- dominio;
- familia.

---

# Fase 7 — Confirmación con semillas

El screening tenía una sola semilla. Ahora se mide variabilidad.

## 7.1. Configuraciones repetidas

Para cada finalista entrenar:

- N*;
- N=64;
- training seeds 0, 1 y 2.

La seed 0 ya existe. Se añaden seeds 1 y 2.

## 7.2. Número de entrenamientos nuevos

\[
3\text{ finalistas}
\times
2\text{ tamaños}
\times
2\text{ seeds nuevas}
\times
2\text{ sistemas}
=
24
\]

## 7.3. Agregación

Para cada configuración reportar:

- media;
- mediana;
- desviación entre seeds;
- mínimo y máximo;
- intervalo bootstrap descriptivo.

No escoger la mejor seed como resultado final.

El resultado es la distribución de las tres seeds.

## 7.4. Confirmación de N*

N* queda confirmado si:

1. la media cumple:

\[
\bar E(N^*)\le1.05\,\bar E(64)
\]

2. la diferencia no está dominada por una seed anómala;
3. la dispersión entre seeds es comparable o menor que la diferencia N*→64;
4. las tres ejecuciones son físicamente válidas.

Si no se cumple, subir al siguiente tamaño:

```text
4 → 8 → 16 → 32 → 64
```

## Salida de esta fase

- N mínimo confirmado por finalista.
- Variabilidad entre semillas.
- Pareto preliminar con incertidumbre.
- Uno o dos candidatos finales por sistema.

---

# Fase 8 — Construir el baseline MD comparable

Esta fase es obligatoria para afirmar “similar a MD”.

## 8.1. Inventario MD

Para cada sistema comprobar:

- número de frames;
- trayectoria de origen;
- temperatura;
- timestep;
- correlación temporal;
- base orbital;
- configuración SIESTA;
- etiquetas disponibles;
- qué frames se usaron anteriormente;
- cuáles están completamente sin utilizar.

## 8.2. Separación temporal

Los frames MD deben dividirse por bloques de trayectoria, no aleatoriamente frame a frame.

Ejemplo:

```text
bloque inicial       → entrenamiento
bloque intermedio    → validación
bloque final         → test
```

Debe evitarse que frames consecutivos aparezcan en train y test.

## 8.3. Modelo MD

Usar exactamente:

- misma arquitectura;
- misma loss;
- mismo optimizador;
- misma política de early stopping;
- mismas training seeds;
- mismo test final.

## 8.4. Tamaños comparables

Por sistema entrenar MD en:

- N=N* del mejor dataset diseñado;
- N=64.

Con tres semillas:

\[
2\text{ tamaños}
\times
3\text{ seeds}
=
6
\]

Por dos sistemas serían 12 entrenamientos.

## 8.5. Si no hay 64 frames MD

No inventar ni duplicar frames.

Usar el mayor tamaño común disponible:

\[
N_\mathrm{matched}
=
\min(N_\mathrm{designed},N_\mathrm{MD\,available})
\]

La comparación se limita a ese N.

## 8.6. Si no existe MD `6x6`

Entonces:

- `w90`: comparación dataset diseñado frente a MD.
- `6x6`: curvas de aprendizaje, generalización y Pareto interno.
- No afirmar equivalencia con MD para `6x6`.

---

# Fase 9 — Crear el test final congelado

Solo se crea después de fijar:

- recetas;
- N*;
- semillas;
- baseline MD;
- métricas.

## 9.1. Test sintético `w90`

Propuesta de 48 estructuras:

\[
k=\{1,2\}
\times
4\text{ dimensiones}
\times
6\text{ amplitudes}
=
48
\]

Amplitudes:

```text
0.02, 0.04, 0.06, 0.08, 0.10, 0.12 Å
```

Esto cubre:

- interior del dominio;
- frontera;
- OOD moderado.

## 9.2. Test sintético `6x6`

Propuesta de 48 estructuras:

\[
4\text{ dimensiones}
\times
6\text{ amplitudes}
\times
2\text{ réplicas}
=
48
\]

Las estructuras deben ser comunes para todos los modelos `6x6`.

## 9.3. Componente MD

Añadir 16 frames MD independientes por sistema.

Deben proceder de:

- bloques no usados en entrenamiento;
- bloques no usados en validación;
- una región temporal independiente;
- temperaturas documentadas.

## 9.4. Tamaño final

Por sistema:

- 48 sintéticas;
- 16 MD;
- total 64.

Para dos sistemas:

\[
64\times2=128
\]

Máximo: 128 nuevos cálculos SIESTA.

Si algunas estructuras ya tienen etiquetas independientes y no fueron usadas para tomar decisiones, pueden reutilizarse.

## 9.5. Congelación

Guardar:

- IDs;
- geometrías;
- hashes;
- referencia SIESTA;
- estrato;
- amplitud;
- dimensión;
- origen MD/sintético.

Después de abrir este test no se cambian:

- recetas;
- hiperparámetros;
- N;
- semillas;
- margen;
- métrica primaria.

---

# Fase 10 — Evaluación final

Evaluar todos los checkpoints de las tres seeds, no solo la mejor seed.

## 10.1. Comparaciones

Por sistema:

1. Dataset diseñado N* frente a diseñado N=64.
2. Dataset diseñado N* frente a MD N*.
3. Dataset diseñado N=64 frente a MD N=64.
4. Sobol frente a random.
5. Error in-domain frente a OOD.
6. Error sintético frente a frames MD.

## 10.2. Diferencias emparejadas

Como todos los modelos usan las mismas estructuras:

\[
d_i =
E^\mathrm{designed}_i
-
E^\mathrm{MD}_i
\]

Calcular el intervalo sobre estas diferencias emparejadas.

Esto es más informativo que comparar dos medias independientes.

## 10.3. No inferioridad

Definir:

\[
\Delta =
E^\mathrm{designed}
-
E^\mathrm{MD}
\]

Se podrá afirmar no inferioridad solamente si:

\[
U_{95\%}(\Delta)<\delta_\mathrm{NI}
\]

donde:

- \(U_{95\%}\) es el límite superior unilateral;
- \(\delta_\mathrm{NI}\) es el margen fijado antes de abrir el test.

## 10.4. Cómo elegir el margen

No usar automáticamente 5 o 10 meV.

Primero relacionar en desarrollo:

- H-MAE;
- error de bandas;
- error de DOS en la ventana energética relevante.

Después definir qué degradación en bandas/DOS considera aceptable el objetivo físico.

Si no puede justificarse un margen:

- reportar diferencia e intervalo;
- no usar la palabra “no inferior”.

---

# Fase 11 — Construir la frontera coste–precisión

## 11.1. Coste reproducible

Coste de repetir el experimento desde cero:

\[
C_\mathrm{reproducible}
=
C_\mathrm{SIESTA}
+
C_\mathrm{training}
\]

## 11.2. Coste incremental

Coste desde el estado actual del repositorio:

\[
C_\mathrm{incremental}
=
C_\mathrm{SIESTA\,nuevo}
+
C_\mathrm{training\,nuevo}
\]

## 11.3. Variables registradas

Para cada modelo:

- número de estructuras SIESTA únicas;
- CPU·h de SIESTA;
- GPU·h;
- mejor época;
- tiempo hasta mejor época;
- memoria máxima;
- número de átomos por estructura;
- H-MAE final.

## 11.4. Pareto

Un modelo A domina a B si:

\[
E_A\le E_B
\]

y:

\[
C_A\le C_B
\]

con al menos una desigualdad estricta.

Construir dos gráficos:

- Pareto reproducible.
- Pareto incremental.

No usar N como único coste.

---

# Fase 12 — Resultados para mostrar al supervisor

## Figura 1 — Curvas de aprendizaje

- Panel izquierdo: `w90`.
- Panel derecho: `6x6`.
- Eje X: N en escala log₂.
- Eje Y: H-MAE.
- Líneas Sobol/random.
- Bandas entre seeds solo para finalistas.
- LHS N=64 como punto aislado.

Mensaje:

> Cuántas estructuras necesita cada estrategia y dónde deja de compensar añadir datos.

## Figura 2 — Pareto coste–precisión

- Eje X: CPU·h SIESTA.
- Eje Y: H-MAE final.
- Color: familia.
- Símbolo: sistema.
- Línea: frontera de Pareto.
- MD evaluado sobre el mismo test.

Mensaje:

> Qué dataset proporciona la mejor precisión para cada presupuesto.

## Figura 3 — Dataset diseñado frente a MD

Para cada sistema:

- diferencia emparejada por estructura;
- media;
- intervalo;
- margen de no inferioridad si existe.

Mensaje:

> Si el dataset diseñado reproduce el rendimiento de MD con menor coste.

## Figura 4 — Generalización

Heatmap:

- filas: dominio de entrenamiento;
- columnas: amplitud del test;
- color: H-MAE.

Mensaje:

> Hasta qué desplazamientos generaliza cada dataset.

## Figura 5 — Estabilidad entre semillas

- Tres puntos por modelo.
- Media e intervalo.
- N* y N=64.

Mensaje:

> Si las mejoras superan la variabilidad del entrenamiento.

## Tabla final

| Sistema | Dataset | Familia | N | H-MAE | IC | Rel. Frob | CPU·h SIESTA | GPU·h | Diferencia MD |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|

---

# Presupuesto completo

## Predicciones iniciales

- 160 checkpoints existentes.
- Coste: solo inferencia.

## Screening principal

- 60 entrenamientos.

## LHS

- Hasta 6 entrenamientos.

## Confirmación

- 24 entrenamientos nuevos.

## Baseline MD

- 6 entrenamientos si solo puede hacerse en `w90`.
- 12 si existe MD comparable para ambos sistemas.

## Total

Sin MD `6x6`:

\[
60+6+24+6=96
\]

Con MD para ambos:

\[
60+6+24+12=102
\]

## SIESTA nuevo

Para las curvas:

\[
0
\]

Para test final:

- hasta 64 por sistema;
- máximo total: 128.

---

# Criterios de parada globales

Detener la campaña y no añadir métodos nuevos cuando ocurra cualquiera:

1. N=32 queda dentro del 5% de N=64 y su variabilidad se solapa.
2. La mejora 32→64 es menor que la variación entre seeds.
3. Sobol y random son estadísticamente indistinguibles y uno es más barato.
4. Un único diseño ocupa claramente la frontera de Pareto.
5. El dataset diseñado ya cumple el margen frente a MD.
6. El error restante parece provenir de la arquitectura y no de cobertura del dataset.

Solo si las curvas se saturan claramente por encima de MD se consideraría después:

1. mezcla designed + MD;
2. farthest-point sampling;
3. committee/uncertainty;
4. modos normales/fonones;
5. active learning completo.

---

# Implementación mínima

No tocaría la UI ni reorganizaría el pipeline.

Solo hacen falta:

1. Un pequeño script para seleccionar prefijos N=4/8/16/32/64 desde pools N=64 existentes.
2. Asegurar que el entrenamiento devuelve el checkpoint de menor `val_loss`.
3. Usar un manifiesto común de desarrollo para `6x6`.
4. Un CSV/JSON agregado para curvas, costes y figuras.

No hacen falta nuevas dependencias, nuevas arquitecturas ni una infraestructura de experimentos general."""



recuerda: no hagas demasiada ingenieria de software, eso es analisis de datos cientifico para fisica, con tener los resutlados y tenerlos bien vale, la ingenieria de software solo lo necesario