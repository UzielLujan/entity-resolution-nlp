# CLAUDE.md — entity-resolution-nlp

Índice de contexto del proyecto. Leer este archivo primero, luego navegar al documento correspondiente según la tarea de la sesión.

---

## Qué es este proyecto

Sistema de Deep Learning para record linkage aplicado a bases de datos clínicas. Implementa una arquitectura Retrieve & Rerank compuesta por un Bi-Encoder entrenado mediante Multiple Negative Ranking Loss (MNRL) y un Cross-Encoder entrenado mediante Binary Cross Entropy (BCE).

Consume registros tabulares previamente etiquetados y serializados; produce modelos, embeddings y evaluaciones para resolución de entidades. No procesa datos crudos ni construye el ground truth.

El dataset de entrada es producido por el repositorio `~/Projects/consultoria-iner`; ambos repos son independientes y se comunican mediante artefactos Parquet.

> **Estado (2026-09-17):** pipeline neuronal implementado y evaluado; seis capítulos del manuscrito redactados, pendientes de revisión. Foco actual: cerrar la primera versión neuronal de la tesis. Detalles y pendientes en `MEMORY.md`.

---

## Datos y artefactos

`docs/data_contract.md` define el contrato entre `consultoria-iner` y este repositorio:
schema de entrada, rutas, propiedad de artefactos, invariantes entre variantes y reglas de
regeneración. Consultarlo antes de tareas que lean o produzcan artefactos de datos.

Este repositorio consume Parquet preparados y produce artefactos bajo `INER_DATA_ROOT/modeling/`.
Nunca lee ni modifica `INER_DATA_ROOT/raw/`.

---

## Índice de documentación y rutas externas

| Si la tarea involucra... | Leer... |
|---|---|
| Contexto global, relación con el repo de consultoría | `~/Projects/consultoria-iner/docs/Contexto_Maestro_Proyecto.md` |
| Relación técnica entre el pipeline de datos y este repositorio | `~/Projects/consultoria-iner/docs/interfaz_consultoria_tesis.md` |
| Bitácora, métricas confirmadas, estado actual y preguntas abiertas | `MEMORY.md` |
| Estructura del repo, rutas y decisiones técnicas | `docs/design_decisions.md` |
| Metodología neuronal: SBERT, DITTO, MNRL, serialización y aumentación | `docs/Metodologia_arquitectura.md` |
| Historial de implementación del Bi-Encoder y Cross-Encoder (fuente para redacción; pendiente de actualización) | `docs/Anexos/historial_desarrollo_pipeline_neuronal.md` |
| Comandos de splitting, entrenamiento, evaluación y exportación | `docs/comandos_proyecto.md` |
| Correspondencia entre el repositorio y los capítulos de tesis | `docs/contexto_manuscrito_tesis.md` |
| Redacción del manuscrito | `manuscript/` |
| Presentación de avance y póster | `~/Documents/Maestria/Tesis/Presentacion/` |
| Material complementario para redacción académica y extensiones de ingeniería: fundamentos, entorno, métricas, incertidumbre, agente LLM, RAG e interfaz interactiva | `docs/Anexos/` |

---

## Entorno Python

El entorno actual del proyecto se administra con `uv` y vive en `.venv/` dentro del directorio del proyecto.
- Para comandos Python no interactivos, usar: `.venv/bin/python ...`
- Para instalar o sincronizar dependencias, usar `uv sync`.

El entorno local anterior del proyecto es `tesis` (micromamba).
- Activa con: `micromamba activate tesis`
- Claude (shell no-interactivo) debe usar: `micromamba run -n tesis python ...`
- Será deprecado en favor de `uv` para entornos reproducibles y portables en cuanto se audite el repositorio end-to-end, solo usar si `.venv` falla.


---

## Reglas

- `~/Data/INER/raw/`: **NUNCA leer ni modificar** sin permiso explícito
- Verificar `.gitignore` antes de cualquier commit
- `.env` nunca al repo, usar `.env.example`
- Antes de instalar cualquier librería: **preguntar primero** y si es aprobada, agregarla a `pyproject.toml`

## Gestión de memoria y bitácora

Existen dos archivos llamados `MEMORY.md` — no confundirlos:

| Archivo | Rol | Estilo |
|---|---|---|
| `MEMORY.md` (raíz del repo) | Bitácora oficial: historial de fases y estado actual, métricas, preguntas abiertas | Extenso y detallado |
| `~/.claude/.../memory/MEMORY.md` | Índice de mis archivos de memoria automática inter-sesión | Ligero — una línea por entrada |

Los archivos en `~/.claude/.../memory/` son la memoria automatica de uso exclusivo inter-sesión de Claude: mantienen la lista de pendientes y próximos pasos para saber en qué nos quedamos al iniciar cada sesión. Solo deben contener insights no obvios y acciones concretas pendientes. Todo lo detallado vive en los `.md` del repo.

---
