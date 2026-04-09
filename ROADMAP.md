# Roadmap completo — EEG Seizure Detection System

## Visión general

El proyecto se construirá en 6 bloques:

1. Fundaciones de datos
2. Baseline clásico sólido
3. Evaluación clínica y sistema real
4. Deep Learning y modelos avanzados
5. Investigación orientada a paper
6. Extensiones futuras (LLM/NLP y sistema ampliado)

---

# BLOQUE 0 — Fundaciones del proyecto [HECHO]

## Sprint 0A — Environment & Project Setup

**Estado:** hecho

### Objetivo

Dejar una base reproducible y profesional para todo el proyecto.

### Entregables

* Estructura del repo
* Devcontainer
* Python 3.11
* pre-commit con ruff
* tests básicos
* README inicial

### Documentación

* `docs/00_sprints/sprint_0A.md`

---

## Sprint 0B — Data Ingestion & Validation

**Estado:** hecho

### Objetivo

Descargar, validar y entender el dataset BIDS.

### Entregables

* Script de descarga
* Validación estructura BIDS
* Primera lectura con `mne-bids`
* Inspección de anotaciones

### Documentación

* `docs/00_sprints/sprint_0B.md`

---

## Sprint 0C — Dataset Indexing

**Estado:** hecho

### Objetivo

Crear el índice estructurado del dataset.

### Entregables

* `manifest.parquet`
* Script reproducible de indexado
* Validación del manifest

### Documentación

* `docs/00_sprints/sprint_0C.md`

---

# BLOQUE 1 — Baseline clásico sólido [ONGOING]

## Sprint 1A — Dataset Preparation

**Estado:** hecho

### Objetivo

Transformar registros EEG crudos en segmentos etiquetados.

### Entregables

* `events.py`
* `segmentation.py`
* `labeling.py`
* `dataset_builder.py`
* `segments.parquet`
* tests y sanity checks

### Documentación

* `docs/00_sprints/sprint_1A.md`
* `docs/10_dataset.md`
* `docs/60_project_overview.md`

---

## Sprint 1B — Feature Engineering (baseline clásico)

**Estado:** hecho

### Objetivo

Extraer features clásicas por ventana para entrenar los primeros modelos.

### Entregables

* Dataset de features tabulares
* Primer pipeline reproducible de extracción

### Tareas

1. Diseñar schema de `features.parquet`
2. Implementar carga de segmento EEG desde `segments.parquet`
3. Implementar features de dominio temporal:

   * mean
   * std
   * RMS
   * line length
4. Implementar features de frecuencia:

   * bandpower delta
   * theta
   * alpha
   * beta
5. Decidir agregación por canal:

   * conservar por canal
   * o estadísticas agregadas
6. Guardar:

   * `data/processed/features.parquet`
7. Añadir tests para features
8. Añadir script de sanity checks

### Validaciones

* No NaNs
* Misma dimensionalidad en todas las filas
* Features razonables
* Correspondencia exacta con segmentos

### Documentación

* `docs/00_sprints/sprint_1B.md`
* empezar `docs/20_pipeline.md`
* empezar `docs/30_modeling.md`
* añadir decisiones clave a `docs/50_decisions.md`

### Contribución al sistema final

Esta fase conecta la señal cruda con ML interpretable y defendible.

---

[AQUÍ ESTAMOS]
## Sprint 1C — Patient-Independent Split

### Objetivo

Crear una estrategia de entrenamiento y evaluación sin leakage por paciente.

### Entregables

* Split train/val/test por sujeto
* Config reproducible de splits

### Tareas

1. Definir estrategia:

   * train / validation / test
   * o cross-validation por sujeto
2. Implementar módulo `splits.py`
3. Garantizar que no hay mezcla de pacientes
4. Guardar archivos de split:

   * `train_subjects.json`
   * `val_subjects.json`
   * `test_subjects.json`
5. Añadir tests

### Validaciones

* Intersección de sujetos vacía
* Distribución de positivos razonable por split

### Documentación

* `docs/00_sprints/sprint_1C.md`
* actualizar `docs/30_modeling.md`
* actualizar `docs/50_decisions.md`

### Contribución al paper

Esto es crucial para credibilidad metodológica.

---

## Sprint 1D — Baseline Models

### Objetivo

Entrenar un baseline simple pero correcto.

### Entregables

* Logistic Regression
* Random Forest
* scripts reproducibles de entrenamiento

### Tareas

1. Preparar matrices X, y
2. Implementar pipeline de entrenamiento
3. Entrenar:

   * Logistic Regression con `class_weight="balanced"`
   * RandomForest con `class_weight="balanced"`
4. Guardar:

   * modelos
   * configs
   * resultados iniciales
5. Añadir control de seeds
6. Registrar hyperparameters

### Validaciones

* Entrenamiento estable
* Sin errores por desbalanceo
* Guardado correcto de artefactos

### Documentación

* `docs/00_sprints/sprint_1D.md`
* actualizar `docs/30_modeling.md`

### Contribución al sistema final

Te da un primer detector serio y defendible.

---

## Sprint 1E — Clinical Evaluation (window-level)

### Objetivo

Evaluar los modelos con métricas relevantes clínicamente.

### Entregables

* métricas por modelo
* tablas y gráficos

### Tareas

1. Implementar métricas:

   * recall / sensitivity
   * specificity
   * F1
   * AUROC
   * confusion matrix
2. Evaluar por split
3. Analizar falsos positivos y falsos negativos
4. Generar reportes básicos

### Validaciones

* Resultados coherentes con dataset muy desbalanceado
* Interpretación clínica inicial

### Documentación

* `docs/00_sprints/sprint_1E.md`
* empezar `docs/40_results.md`

### Contribución al paper

Primera sección de resultados.

---

## Sprint 1F — Labeling/Threshold Experiments

### Objetivo

Iniciar la línea experimental más paper-worthy.

### Entregables

* comparación sistemática de estrategias de etiquetado

### Tareas

1. Repetir dataset generation con distintos thresholds:

   * 0.3
   * 0.5
   * 0.7
2. Comparar:

   * positivos generados
   * recall
   * specificity
   * F1
   * AUROC
3. Comparar también:

   * drop ambiguous
   * keep ambiguous as negative
4. Analizar impacto en falsos positivos/negativos

### Documentación

* `docs/00_sprints/sprint_1F.md`
* actualizar `docs/40_results.md`
* actualizar `docs/50_decisions.md`

### Contribución al paper

Este sprint puede convertirse en el núcleo experimental del paper.

---

# BLOQUE 2 — Del baseline a un sistema clínicamente más real

## Sprint 2A — Temporal Post-processing

### Objetivo

Mejorar la coherencia temporal de las predicciones.

### Entregables

* módulo de smoothing/post-processing

### Tareas

1. Diseñar reglas temporales:

   * majority voting en ventanas consecutivas
   * smoothing
   * mínimo número de ventanas para declarar evento
2. Implementar módulo reusable
3. Comparar antes y después

### Documentación

* `docs/00_sprints/sprint_2A.md`
* actualizar `docs/20_pipeline.md`

### Contribución al sistema real

Reduce predicciones aisladas poco realistas.

---

## Sprint 2B — Event-level Evaluation

### Objetivo

Pasar de evaluación por ventana a evaluación por evento clínico.

### Entregables

* métricas a nivel evento

### Tareas

1. Definir qué es un evento detectado
2. Implementar matching entre eventos predichos y reales
3. Calcular:

   * event recall
   * false alarms per hour
   * detection delay
4. Comparar modelos y post-processing

### Documentación

* `docs/00_sprints/sprint_2B.md`
* actualizar `docs/40_results.md`

### Contribución al paper

Esto eleva mucho el rigor clínico del proyecto.

---

## Sprint 2C — Probability Calibration

### Objetivo

Hacer que las probabilidades sean interpretables.

### Entregables

* calibration module
* reliability diagrams

### Tareas

1. Medir calibración inicial
2. Implementar:

   * Platt scaling o temperature scaling
3. Generar:

   * reliability diagrams
   * calibration metrics
4. Comparar calibrated vs uncalibrated

### Documentación

* `docs/00_sprints/sprint_2C.md`
* actualizar `docs/30_modeling.md`
* actualizar `docs/40_results.md`

### Contribución al sistema real

Necesario si luego quieres decisiones basadas en confianza.

---

## Sprint 2D — API básica de inferencia

### Objetivo

Convertir el modelo en un servicio usable.

### Entregables

* FastAPI mínima operativa

### Tareas

1. Diseñar contrato de inferencia
2. Crear endpoint:

   * `POST /predict`
3. Input:

   * EEG chunk o archivo
4. Output:

   * label
   * probability
5. Añadir validación de inputs
6. Documentar API

### Documentación

* `docs/00_sprints/sprint_2D.md`
* empezar `docs/system.md` o ampliar `docs/20_pipeline.md`

### Contribución al sistema final

Aquí el proyecto deja de ser solo modelo y empieza a ser sistema.

---

## Sprint 2E — End-to-End Inference Pipeline

### Objetivo

Conectar preprocessing + features + model + API.

### Entregables

* pipeline completo de inferencia

### Tareas

1. Segmentation en inferencia
2. Feature extraction en inferencia
3. Predicción por ventana
4. Post-processing
5. Output agregado
6. Tests end-to-end

### Documentación

* `docs/00_sprints/sprint_2E.md`
* actualizar `docs/20_pipeline.md`

### Contribución al sistema final

Primera versión real del sistema completo.

---

# BLOQUE 3 — Deep Learning

## Sprint 3A — Dataset preparation for DL

### Objetivo

Preparar ventanas de señal cruda para modelos profundos.

### Entregables

* dataset listo para PyTorch

### Tareas

1. Definir tensor shape
2. Crear Dataset/Dataloader
3. Decidir normalización
4. Implementar caching si hace falta
5. Añadir tests

### Documentación

* `docs/00_sprints/sprint_3A.md`
* actualizar `docs/30_modeling.md`

---

## Sprint 3B — CNN 1D baseline

### Objetivo

Entrenar un baseline profundo multicanal.

### Entregables

* CNN 1D funcional
* pipeline de training DL

### Tareas

1. Definir arquitectura
2. Entrenar con early stopping
3. Guardar checkpoints
4. Evaluar con mismas métricas que baseline clásico

### Documentación

* `docs/00_sprints/sprint_3B.md`
* actualizar `docs/30_modeling.md`
* actualizar `docs/40_results.md`

### Contribución

Subida fuerte de nivel técnico.

---

## Sprint 3C — Sequence models

### Objetivo

Modelar contexto temporal más largo.

### Entregables

* LSTM/GRU baseline

### Tareas

1. Diseñar input secuencial
2. Implementar training
3. Comparar con CNN y clásico

### Documentación

* `docs/00_sprints/sprint_3C.md`

---

## Sprint 3D — Transformer-based modeling

### Objetivo

Explorar un modelo más avanzado y con proyección.

### Entregables

* primer Transformer temporal

### Tareas

1. Definir granularidad temporal
2. Diseñar encoder temporal
3. Entrenar versión inicial
4. Comparar con CNN/LSTM

### Documentación

* `docs/00_sprints/sprint_3D.md`
* actualizar `docs/30_modeling.md`
* actualizar `docs/40_results.md`

### Contribución

Este sprint es muy valioso para máster y visión a futuro.

---

## Sprint 3E — Comparative Study

### Objetivo

Comparar clásicas vs DL vs Transformers.

### Entregables

* tabla comparativa sólida

### Tareas

1. Comparar:

   * performance
   * coste computacional
   * interpretabilidad
   * robustez
2. Redactar conclusiones técnicas

### Documentación

* `docs/00_sprints/sprint_3E.md`
* actualizar `docs/40_results.md`

### Contribución al paper

Sección central de comparación.

---

# BLOQUE 4 — Uncertainty + Explainability

## Sprint 4A — Uncertainty Estimation

### Objetivo

Añadir estimación de incertidumbre.

### Entregables

* salida con prediction + uncertainty

### Tareas

1. Elegir método:

   * MC Dropout
   * ensemble ligero
2. Implementar score de incertidumbre
3. Analizar relación:

   * error vs uncertainty
4. Diseñar regla:

   * “refer to specialist if uncertainty > threshold”

### Documentación

* `docs/00_sprints/sprint_4A.md`
* actualizar `docs/30_modeling.md`
* actualizar `docs/40_results.md`

### Contribución clínica

Muy fuerte: introduces abstention / decision support.

---

## Sprint 4B — Decision under uncertainty

### Objetivo

Convertir la incertidumbre en una política de uso.

### Entregables

* estrategia de triage o referral

### Tareas

1. Simular política:

   * alta confianza → automática
   * baja confianza → revisar por especialista
2. Medir trade-offs
3. Analizar utilidad clínica

### Documentación

* `docs/00_sprints/sprint_4B.md`
* actualizar `docs/50_decisions.md`

### Contribución al paper

Muy interesante y diferencial.

---

## Sprint 4C — Explainability for classical/DL models

### Objetivo

Visualizar qué partes de la señal son relevantes.

### Entregables

* visualizaciones interpretables

### Tareas

1. Para clásico:

   * feature importance
2. Para CNN:

   * saliency temporal / Grad-CAM adaptado
3. Para Transformer:

   * attention maps
4. Análisis cualitativo de casos

### Documentación

* `docs/00_sprints/sprint_4C.md`
* actualizar `docs/40_results.md`

### Contribución

Sube mucho el valor clínico y académico.

---

# BLOQUE 5 — Research / Paper

## Sprint 5A — Define paper question

### Objetivo

Cerrar la pregunta principal del paper.

### Opciones prioritarias

1. Impact of labeling strategy on seizure detection
2. Threshold trade-offs and clinical utility
3. Modular seizure detection pipeline with uncertainty-aware decision support

### Tareas

1. Elegir pregunta principal
2. Elegir hipótesis
3. Elegir experimentos mínimos obligatorios

### Documentación

* `docs/00_sprints/sprint_5A.md`
* crear `paper/outline.md`

---

## Sprint 5B — Core experiments

### Objetivo

Ejecutar los experimentos centrales del paper.

### Tareas

1. Threshold studies
2. Ambiguous window policy
3. Class imbalance handling
4. Calibration impact
5. Optional uncertainty impact

### Entregables

* tablas
* figuras
* conclusiones

### Documentación

* `docs/00_sprints/sprint_5B.md`
* actualizar `docs/40_results.md`

---

## Sprint 5C — Ablation Study

### Objetivo

Entender qué componentes aportan valor.

### Tareas

1. Sin bandpower
2. Sin line length
3. Ventanas 5s vs 10s
4. Con y sin smoothing
5. Con y sin calibration

### Documentación

* `docs/00_sprints/sprint_5C.md`

### Contribución al paper

Aporta rigor experimental.

---

## Sprint 5D — Robustness & Error Analysis

### Objetivo

Analizar fallos de forma profunda.

### Tareas

1. Resultados por sujeto
2. Casos difíciles
3. Sesgos del modelo
4. Sensibilidad al ruido
5. Error analysis cualitativo

### Documentación

* `docs/00_sprints/sprint_5D.md`
* actualizar `docs/40_results.md`

---

## Sprint 5E — Write research-style manuscript

### Objetivo

Convertir toda la documentación acumulada en paper.

### Estructura

* Abstract
* Introduction
* Related Work
* Dataset
* Methods
* Experiments
* Results
* Discussion
* Limitations
* Future Work

### Tareas

1. Reutilizar `project_overview.md`
2. Reutilizar `dataset.md`
3. Reutilizar `pipeline.md`
4. Reutilizar `results.md`
5. Redactar discusión final

### Entregables

* `paper/paper.md` o LaTeX
* figuras finales

---

# BLOQUE 6 — Sistema final y extensiones futuras

## Sprint 6A — Final API with advanced outputs

### Objetivo

Mejorar la API final.

### Output esperado

* prediction
* calibrated probability
* uncertainty
* optional explanation

### Tareas

1. Extender contrato REST
2. Añadir logging
3. Añadir validación robusta
4. Tests de integración

---

## Sprint 6B — System packaging and demo

### Objetivo

Dejar el proyecto enseñable.

### Tareas

1. Demo reproducible
2. README fuerte
3. example requests
4. architecture figure
5. model card

### Documentación

* `docs/final_system.md`
* `model_card.md`

---

## Sprint 6C — NLP/LLM extension: report generation

### Objetivo

Añadir una capa de explicación textual.

### Ejemplos

* resumen automático de predicciones
* generación de pseudo-informe clínico

### Tareas

1. Diseñar salida estructurada del sistema
2. Crear generador textual simple
3. Opcionalmente integrar LLM

### Nota

Esto debe venir al final, no antes.

---

## Sprint 6D — NLP/LLM extension: RAG or guideline grounding

### Objetivo

Explorar una capa híbrida ML + LLM.

### Tareas

1. Recuperar guías o conocimiento médico
2. Generar explicaciones grounding-aware
3. Evaluar utilidad

### Nota

Esto es extra avanzado; muy valioso, pero no imprescindible para el núcleo.

---
