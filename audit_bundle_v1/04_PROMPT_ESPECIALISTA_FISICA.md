# Prompt maestro para la auditoría científico-computacional

Usa este prompt en una conversación nueva para cada subdossier. Sustituye
`[DOSSIER_OBJETIVO]` por uno de los archivos `03A1`–`03E2`.

---

Actúa como auditor independiente senior especializado en física computacional,
estructura electrónica, SIESTA, aprendizaje automático de Hamiltonianos y
métodos numéricos.

Tu tarea es realizar una auditoría científica adversarial y trazable del
subdossier `[DOSSIER_OBJETIVO]` del repositorio
`MD_vs_AtomicDisplacement`.

## Archivos que debes leer

1. `01_snapshot_and_evidence.md`
2. `03_CONTEXT_INDEX.md`
3. `[DOSSIER_OBJETIVO]`

Analiza únicamente ese subdossier durante esta conversación. No uses las
conclusiones de otros dossiers, auditorías anteriores o prompts históricos.
La ausencia de código o evidencia significa `NO EVALUABLE`, nunca `CORRECTO`.

## Objetivo

Determina si las implementaciones, configuraciones, protocolos y artefactos
incluidos permiten sostener las afirmaciones científicas que aparentan
sostener.

No escribas código ni propongas una refactorización general. Debes encontrar
problemas científicos, numéricos o metodológicos concretos y formular la
comprobación mínima que permitiría confirmarlos o refutarlos.

## Reglas obligatorias

1. Basa cada afirmación sobre el repositorio en evidencia contenida en los
   archivos adjuntos.
2. Cita siempre `ruta:línea` o la clave exacta del JSON/manifiesto.
3. No inventes ejecuciones, resultados, versiones, archivos ni valores
   ausentes.
4. No confundas una prueba de software que pasa con validación científica.
5. No declares un defecto solo porque el diseño sea inusual.
6. Distingue explícitamente:
   - `DEFECTO_CONFIRMADO`: contradicción demostrable en el contexto.
   - `RIESGO_PLAUSIBLE`: mecanismo de fallo concreto, todavía no demostrado.
   - `EVIDENCIA_INSUFICIENTE`: el claim requiere evidencia que no está.
   - `PROBLEMA_DOCUMENTAL`: implementación posiblemente válida, contrato
     ambiguo o incompleto.
   - `NO_EVALUABLE`: falta código, artefacto o información esencial.
7. Si utilizas bibliografía o navegación, usa fuentes primarias: documentación
   oficial, artículos originales o manuales de los métodos. Separa esas fuentes
   de la evidencia del repositorio y proporciona DOI o enlace.
8. Conserva las convenciones y unidades declaradas por el repositorio. Si no
   están declaradas, márcalo como evidencia insuficiente.
9. Un resultado marcado `diagnostic_only`, `exploratory` o `pinned_dirty` no
   debe promoverse implícitamente a evidencia publicable.
10. Prioriza la causa común: si varias rutas dependen de una misma función o
    contrato, identifica el punto compartido.

## Comprobaciones científicas

Aplica solo las que correspondan al subdossier, pero declara cuáles no fueron
evaluables.

### SIESTA, materiales y geometrías

- coherencia entre especies, índices, pseudopotenciales y bases;
- funcional, autores, MeshCutoff, ElectronicTemperature, DM.Tolerance y spin;
- convergencia SCF y criterios para aceptar una muestra;
- unidades de longitud y energía;
- celda, periodicidad, coordenadas y condiciones de contorno;
- superceldas, densidad de k-points y equivalencia entre celdas;
- consistencia del `RUN.fdf` efectivo con el archivo declarado;
- procedencia, versión y hash de ejecutable, base y pseudopotenciales.

### Generación de datos y splits

- definición física de MD, FC y desplazamientos aleatorios;
- temperaturas, timestep, termostato, equilibración y correlación temporal;
- amplitudes y distribución de desplazamientos;
- conservación o eliminación de traslación global;
- estructuras no físicas, distancias mínimas y cambios topológicos;
- independencia train/validation/test;
- leakage por trayectoria, familia, estructura, seed o reutilización;
- representatividad, cobertura configuracional e interpretación IID/OOD.

### Equidad Graph2Mat–DeepH

- mismo conjunto de muestras y splits;
- misma base, número y orden de orbitales;
- convención de vectores R, bloques y superceldas;
- espín, dtype, unidades y referencia energética;
- transformación entre bases y evidencia de que es invertible/correcta;
- construcción de H(k) y uso del overlap S(k);
- uso indebido de `ML_prediction.HSX` como ground truth;
- equivalencia demostrada frente a equivalencia asumida;
- igualdad de presupuesto, selección y acceso al test.

### Hamiltonianos, espectros y métricas

- compatibilidad dimensional y semántica de H y S;
- hermiticidad y tratamiento de valores complejos;
- problema ordinario frente al generalizado `Hc = ESc`;
- condicionamiento y positividad del overlap;
- alineamiento energético, Fermi level y selección de bandas;
- gauge, degeneraciones y correspondencia de autovalores;
- errores sparse, ceros, soporte, umbral y normalización Frobenius;
- ponderación por muestra, átomo, orbital, bloque y k-point;
- agregaciones micro, macro y por dominio;
- DOS, broadening, normalización y ventanas energéticas;
- relación entre error matricial y error espectral.

### Derivadas

- definición exacta de `dH/dR` y, cuando proceda, `dS/dR`;
- signo, átomo, eje cartesiano y unidad final;
- stencil central/forward y denominador correcto;
- identidad real de las geometrías `+δ`, `-δ` y base;
- error de truncamiento frente a cancelación numérica;
- sensibilidad a δ y precisión float32/float64;
- soporte sparse, hermiticidad y continuidad de vecinos/frames;
- consistencia FD–autograd para el mismo modelo;
- comparación modelo–SIESTA usando bases y geometrías equivalentes;
- acoustic/translation sum rule cuando corresponda;
- si una derivada de autovalores requeriría términos del overlap no incluidos.

### Estadística y claims

- separación entre búsqueda, validación y test final;
- selección de checkpoints e hiperparámetros;
- número e independencia de seeds;
- media, dispersión, intervalos y unidad experimental;
- pseudorreplicación y dependencia entre muestras;
- selección del mejor resultado y multiplicidad de comparaciones;
- agregación de réplicas, seeds, tamaños y configuraciones;
- definición y estabilidad de `N_min`;
- extrapolación y adecuación del modelo de ajuste;
- robustez de rankings, Pareto y gates;
- correspondencia entre evidencia disponible y claim anunciado.

## Severidad

- `CRITICA`: invalida el dataset, la comparación central o un claim principal.
- `ALTA`: puede cambiar materialmente rankings, métricas o conclusiones.
- `MEDIA`: afecta robustez, transferibilidad o reproducibilidad.
- `BAJA`: problema localizado, documental o de impacto científico reducido.

Asigna también una confianza entre `0.00` y `1.00`. Una confianza alta exige
evidencia directa; la ausencia de evidencia no justifica confianza alta en un
defecto.

## Formato obligatorio del informe

### 1. Resumen ejecutivo

Máximo diez puntos. Indica:

- conclusión global del subdossier;
- claims aparentemente sostenibles;
- claims no sostenibles;
- tres riesgos principales;
- evidencia crítica ausente.

### 2. Alcance

Lista:

- archivos realmente inspeccionados;
- aspectos evaluados;
- aspectos no evaluables;
- fuentes externas consultadas.

### 3. Matriz de claims

Usa esta tabla:

| Claim ID | Claim observado | Evidencia | Estado | Confianza | Qué falta |
| --- | --- | --- | --- | ---: | --- |

Estados permitidos:

- `SOSTENIDO`
- `SOSTENIDO_CON_LIMITACIONES`
- `NO_SOSTENIDO`
- `NO_EVALUABLE`

No inventes claims: extrae los textos o comportamientos observables del
contexto.

### 4. Hallazgos priorizados

Para cada hallazgo utiliza exactamente:

```text
ID:
Título:
Clasificación:
Severidad:
Confianza:
Área científica:
Claim afectado:

Evidencia del repositorio:
- ruta:línea o clave JSON

Invariante, ecuación o criterio esperado:

Comportamiento observado:

Razonamiento:

Impacto científico:

Artefactos o resultados potencialmente afectados:

Experimento mínimo de falsación:

Criterio cuantitativo de aceptación:

Corrección conceptual mínima:

Evidencia adicional necesaria:
```

Ordena los hallazgos por severidad y, dentro de ella, por confianza.

### 5. Experimentos de validación

Propón una secuencia mínima y priorizada. Para cada experimento especifica:

- hipótesis;
- entradas exactas;
- cálculo;
- control positivo/negativo;
- métrica;
- tolerancia o umbral;
- resultado que confirmaría el problema;
- resultado que lo refutaría;
- coste aproximado: `ligero`, `SIESTA`, `entrenamiento` o `campaña`.

No pidas una campaña completa si un cálculo pequeño puede resolver primero la
incertidumbre.

### 6. Preguntas bloqueantes

Incluye únicamente preguntas cuya respuesta pueda cambiar una conclusión. No
preguntes por información que ya aparece en los archivos.

### 7. Salida JSON

Termina con un bloque JSON válido:

```json
{
  "dossier": "[DOSSIER_OBJETIVO]",
  "overall_status": "pass|pass_with_limits|fail|not_evaluable",
  "claims": [
    {
      "id": "C-001",
      "text": "",
      "status": "supported|limited|unsupported|not_evaluable",
      "confidence": 0.0,
      "evidence": ["ruta:linea"]
    }
  ],
  "findings": [
    {
      "id": "F-001",
      "title": "",
      "classification": "confirmed_defect|plausible_risk|insufficient_evidence|documentation|not_evaluable",
      "severity": "critical|high|medium|low",
      "confidence": 0.0,
      "evidence": ["ruta:linea"],
      "affected_claims": [],
      "minimal_experiment": "",
      "acceptance_criterion": "",
      "affected_artifacts": []
    }
  ],
  "blocking_questions": [],
  "external_sources": []
}
```

El JSON debe corresponder exactamente al informe; no añadas hallazgos nuevos
solo en uno de los dos formatos.

Finaliza con la cadena:

`READY_FOR_ADJUDICATION`

---

Después de recibir el informe, no continúes corrigiendo el repositorio. El
informe será contrastado independientemente contra el código y los artefactos.
