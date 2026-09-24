# Barrido de átomos activos en grafeno 6×6 — diseño fijado

## Pregunta

¿Cuántos átomos deben desplazarse en las estructuras de entrenamiento para que Graph2Mat mantenga la precisión sobre configuraciones colectivas de una celda 6×6?

## Diseño

- `k = {2, 12, 22, 32, 42, 52, 62, 72}`.
- Sobol 3D, `R = 0.12 Å`, Train64.
- Validation48 común y Test32 Sobol 3D `R = 0.12 Å`, ambos con `k=72`.
- Dos semillas de entrenamiento: 0 y 1.
- `elementwise_mse`, batch 16, 8000 actualizaciones, 2000 validaciones, scheduler coseno y mejor checkpoint por `val_loss`.
- Métricas: H-MAE y Frobenius relativo por estructura.

Los conjuntos activos son prefijos compactos, conectados y anidados del grafo periódico de enlaces. Se genera una única secuencia Sobol de 72 átomos y, para cada `k`, se enmascaran los restantes; los desplazamientos de cualquier átomo compartido son idénticos entre valores de `k`.

## Comparación principal

Todos los modelos se evalúan sobre exactamente el mismo Test32 con `k=72`. Así, un `k` pequeño no obtiene artificialmente un error menor por contener más átomos sin desplazar.

La referencia es `k=72`. El mínimo exploratorio será el menor `k` cuya media de dos semillas cumpla simultáneamente:

- H-MAE ≤ 1.05 × referencia.
- Frobenius relativo ≤ 1.10 × referencia.

Con dos semillas, la variabilidad se interpretará únicamente de forma descriptiva.

## Reutilización y coste esperado

Se reutilizan el Train64, Test32 y los dos checkpoints actuales de `k=72`. Se generarán como máximo 448 etiquetas SIESTA y 14 entrenamientos nuevos. Tiempo esperado: 10–11 horas en la máquina actual. Cada valor adicional de `k` costaría aproximadamente 1.4 horas.

El diseño no se cambiará después de observar los resultados.
