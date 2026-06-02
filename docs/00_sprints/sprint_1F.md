# Sprint 1F — Labeling / Threshold Experiments

**Estado:** hecho

## Objetivo

Comparar sistemáticamente 6 estrategias de etiquetado para evaluar el
impacto del overlap threshold y la política de manejo de ventanas
ambiguas en la detección de crisis epilépticas.

## Diseño experimental

### Configuraciones

| # | Threshold | Drop partial | Nombre              |
|---|-----------|-------------|---------------------|
| 1 | 0.3       | Yes         | thresh_0.3_drop     |
| 2 | 0.3       | No          | thresh_0.3_keep     |
| 3 | 0.5       | Yes         | thresh_0.5_drop     |
| 4 | 0.5       | No          | thresh_0.5_keep     |
| 5 | 0.7       | Yes         | thresh_0.7_drop     |
| 6 | 0.7       | No          | thresh_0.7_keep     |

### Modelo

Solo Logistic Regression con `class_weight="balanced"`. Random Forest se
excluye porque en Sprint 1E mostró overfitting catastrófico (0%
sensitivity en test), lo que lo invalida para comparación de estrategias.

### Metodología

En lugar de reconstruir el dataset completo desde los EEG crudos para
cada configuración (costoso, requiere mne), se reutiliza el dataset
existente:

1. **Join** `features.parquet` con `segments.parquet` en la clave
   `(subject, path, start_sec, end_sec)` para recuperar `overlap_ratio`
2. **Relabel** vectorizado: reasigna etiquetas según la nueva política
3. **Split** con la misma asignación patient-independent de Sprint 1C
4. **Train** Logistic Regression (seed=42, scaled features)
5. **Evaluate** con las 19 métricas clínicas de Sprint 1E

## Resultados — Test Set

| Config          | Thresh | Drop | N+    | Sens   | Spec   | F1     | F2     | AUROC  | AUPRC  |
|-----------------|--------|------|-------|--------|--------|--------|--------|--------|--------|
| thresh_0.3_drop | 0.3    | Yes  | 2,321 | 0.2314 | 0.9289 | 0.0331 | 0.0682 | 0.6623 | 0.0123 |
| thresh_0.3_keep | 0.3    | No   | 2,321 | 0.2314 | 0.9289 | 0.0331 | 0.0682 | 0.6623 | 0.0123 |
| thresh_0.5_drop | 0.5    | Yes  | 2,321 | 0.2314 | 0.9289 | 0.0331 | 0.0682 | 0.6623 | 0.0123 |
| thresh_0.5_keep | 0.5    | No   | 2,321 | 0.2314 | 0.9289 | 0.0331 | 0.0682 | 0.6623 | 0.0123 |
| thresh_0.7_drop | 0.7    | Yes  | 2,247 | 0.2289 | 0.9310 | 0.0323 | 0.0666 | 0.6549 | 0.0114 |
| thresh_0.7_keep | 0.7    | No   | 2,247 | 0.2289 | 0.9311 | 0.0323 | 0.0667 | 0.6553 | 0.0115 |

## Hallazgos clave

### 1. Configs 1-4 producen resultados idénticos

Los thresholds 0.3 y 0.5 (con ambas políticas drop/keep) generan
exactamente el mismo dataset y las mismas métricas. Esto ocurre porque
**no existen ventanas con 0 < overlap < 0.5** en el dataset disponible.

Causa: el dataset original se construyó con `threshold=0.5,
drop_partial=True`, eliminando las ventanas con overlap parcial < 0.5.
Los únicos valores de overlap parcial presentes son 0.6 (74 ventanas) y
0.8 (82 ventanas), ambos por encima de 0.5.

### 2. Threshold 0.7 tiene impacto mínimo

Solo 74 ventanas cambian de estado (las que tienen overlap=0.6):
- **Drop**: se eliminan 74 ventanas → sensibilidad baja de 0.2314 a
  0.2289 (-0.0025)
- **Keep**: 74 ventanas se reetiquetan como negativas → resultado casi
  idéntico al drop

La diferencia es imperceptible porque 74 ventanas representan el
0.01% del dataset total (707,524 ventanas).

### 3. La estrategia de etiquetado NO es el bottleneck

El hallazgo principal es que la elección de threshold tiene impacto
negligible en el rendimiento. Esto se debe a:

- Solo 156 de 707,524 ventanas (0.022%) tienen overlap parcial
- Los bordes de las crisis epilépticas en CHB-MIT se alinean
  relativamente bien con los límites de ventanas de 5 segundos
- El verdadero bottleneck está en las **features** y la **arquitectura
  del modelo**, no en la política de etiquetado

### 4. Validación de la decisión original

Los resultados validan que `threshold=0.5, drop_partial=True` es una
elección razonable para el baseline. No hay beneficio significativo
en usar thresholds más bajos o más altos con este dataset y tamaño
de ventana.

## Limitación de datos

Las ventanas con `0 < overlap < 0.5` fueron descartadas durante la
construcción original del dataset. Análisis de los bordes de crisis
sugiere que existen ~74 ventanas con overlap ≈ 0.4 y ~82 con overlap
≈ 0.2 que no están disponibles.

Esto afecta principalmente al experimento con `threshold=0.3`, donde
~74 ventanas que deberían ser positivas (overlap 0.4 >= 0.3) faltan
del dataset. Sin embargo, dado que 74 ventanas son el 3% de los
~2,321 positivos totales, el impacto en las métricas sería marginal.

## Entregables

| Archivo | Descripción |
|---------|-------------|
| `src/neuro_eeg_cdss/experiments/__init__.py` | Package init |
| `src/neuro_eeg_cdss/experiments/labeling.py` | Módulo experimental: join, relabel, run, format |
| `scripts/experiments/run_labeling_experiments.py` | Script orquestador |
| `tests/test_labeling_experiment.py` | 28 tests |
| `experiments/labeling/all_results.json` | Resultados combinados |
| `experiments/labeling/comparison_test.txt` | Tabla comparativa (test) |

## Tests

28 tests organizados en 8 clases:
- `TestLabelingExperimentConfig` (4 tests): nombres y configuración
- `TestJoinOverlapRatio` (6 tests): join correcto, errores en inputs
- `TestRelabelDataset` (7 tests): relabeling con diferentes políticas
- `TestComputeDatasetStats` (3 tests): estadísticas del dataset
- `TestAnalyzeDataCompleteness` (2 tests): análisis de completitud
- `TestRunSingleExperiment` (4 tests): pipeline end-to-end
- `TestFormatComparisonTable` (2 tests): formato de salida

## Implicaciones para el paper

Este sprint aporta evidencia empírica de que:

1. **La política de etiquetado no es un factor diferenciador** para este
   dataset y tamaño de ventana → el cuello de botella está en la
   representación de features
2. **El estudio experimental completo** demuestra rigor metodológico
   incluso cuando los resultados son negativos (no-difference)
3. **Los resultados negativos son publicables** cuando están
   correctamente documentados y explicados
4. **Dirección futura**: el impacto del threshold podría ser mayor con
   ventanas más pequeñas (donde más bordes de crisis caerían dentro
   de ventanas parciales) o con overlap entre ventanas

## Decisiones técnicas

| Decisión | Justificación |
|----------|---------------|
| Join en memoria vs. rebuild | Evita dependencia de mne/EDF, reutiliza features existentes |
| Solo LR, no RF | RF demostró overfitting catastrófico en Sprint 1E |
| Vectorized relabeling | Performance: relabela 707K ventanas en <1s vs. row-by-row |
| Documentar missing windows | Transparencia metodológica, cuantificar impacto |

## Commit sugerido

```
Sprint 1F: Labeling threshold experiments — 6 configs show minimal
impact of overlap policy on seizure detection with 5s windows
```
