# Entorno y dependencias en Python moderno
## micromamba + uv + pyproject.toml

> Documento de referencia y nota personal: guía para entender cómo manejar entornos y dependencias en Python de forma moderna y eficiente. Se enfoca en la separación clara entre el entorno del sistema (Python, CUDA, etc.) y las dependencias del proyecto (paquetes Python), usando micromamba para la primera capa y uv para la segunda, con `pyproject.toml` como fuente única de verdad para las dependencias.

Este documento complementa [[design_decisions.md]] y queda referenciado desde [[CLAUDE.md]] como apoyo para decisiones de entorno del proyecto.

---

## El problema: dos tipos de dependencias

Cuando trabajas en un proyecto de Python tienes dos capas de dependencias completamente distintas:

```
Capa 1: el intérprete y librerías del sistema
────────────────────────────────────────────
Python 3.11
CUDA 12.x
libssl
...

Capa 2: paquetes Python de tu proyecto
────────────────────────────────────────────
torch
transformers
sentence-transformers
pandas
...
```

El caos histórico viene de que la gente mezclaba ambas capas en un solo lugar.
**Conda/miniconda** intentó resolver ambas a la vez — manejaba tanto el intérprete
como los paquetes Python. Funcionaba, pero era lento y pesado.

La comunidad moderna separó responsabilidades:

```
micromamba  →  resuelve solo la Capa 1
uv          →  resuelve solo la Capa 2
```

---

## micromamba — solo la Capa 1

Es miniconda pero reescrito en C++. Hace exactamente lo mismo — crear entornos
aislados con una versión específica de Python y librerías del sistema — pero es
**10-50x más rápido** resolviendo dependencias.

```bash
# miniconda / anaconda — lo que usabas antes
conda create -n tesis python=3.11
conda activate tesis

# micromamba — exactamente lo mismo, distinta herramienta
micromamba create -n tesis python=3.11
micromamba activate tesis
```

Los comandos son casi idénticos. La diferencia no es conceptual sino de velocidad
e instalación — micromamba es un solo binario, no necesita instalar toda la suite
de Anaconda.

**Lo que micromamba NO hace:** instalar paquetes Python de PyPI (`torch`,
`transformers`, etc.) de forma eficiente. Para eso existía `pip`, y ahora existe `uv`.

---

## uv — solo la Capa 2

Es un reemplazo de `pip` reescrito en Rust. Resuelve e instala paquetes Python de PyPI.

```bash
# antes
pip install torch transformers pandas

# ahora
uv pip install torch transformers pandas
```

La diferencia de velocidad es brutal — uv puede ser **10-100x más rápido** que pip
en resolver dependencias complejas como las de ML.

**Lo que uv NO hace:** crear entornos ni manejar versiones de Python. Para eso
sigue existiendo micromamba.

---

## Cómo encajan juntos

```bash
micromamba create -n tesis python=3.11   # crea el entorno con Python
micromamba activate tesis                # activa el entorno
uv pip install torch transformers        # instala paquetes EN ese entorno
```

Micromamba pone la caja, uv llena la caja. Son complementarios, no competidores.

```
┌─────────────────────────────────┐
│  entorno: tesis  (micromamba)   │
│  Python 3.11                    │
│  ┌───────────────────────────┐  │
│  │ torch        (uv)         │  │
│  │ transformers (uv)         │  │
│  │ pandas       (uv)         │  │
│  │ record_linkage -e  (uv)   │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
```

---

## La confusión histórica: pip vs conda

La confusión venía de que conda hacía las dos capas a la vez, mezclando conceptos
que en realidad son independientes.

```
conda install torch   →  busca en los canales de Anaconda (conda-forge, defaults)
pip install torch     →  busca en PyPI
```

Son dos repositorios de paquetes distintos. El problema era que mucha gente mezclaba
ambos dentro del mismo entorno:

```bash
# ❌ patrón conflictivo — muy común antes
micromamba create -n tesis python=3.11
micromamba activate tesis
conda install pandas          # con conda
pip install transformers      # con pip porque no estaba en conda
pip install torch             # y más con pip...
```

Conda no sabía qué había instalado pip y viceversa — el entorno eventualmente
se corrompía o tenía versiones incompatibles de forma silenciosa.

**La regla moderna y limpia:**

```
micromamba  →  SOLO para crear el entorno y Python
               (y excepcionalmente CUDA, cuDNN — cosas del sistema)

uv / pip    →  TODO lo demás, siempre desde PyPI
```

En la práctica para cualquier proyecto:

```bash
# Crear entorno — micromamba
micromamba create -n tesis python=3.11
micromamba activate tesis

# Todo lo demás — uv, nunca conda install
uv pip install torch transformers sentence-transformers
uv pip install pandas pyarrow jupyter
uv pip install -e .   # tu propio paquete instalado en modo editable
```

Nunca más `conda install` para paquetes Python. La separación queda limpia y
sin riesgo de conflictos entre repositorios.

---

## `requirements.txt` vs `environment.yaml`

```
requirements.txt   →  solo paquetes Python (Capa 2)
                       lo entiende pip y uv
                       pandas==2.1.0
                       torch>=2.0

environment.yaml   →  entorno completo (Capa 1 + Capa 2)
                       lo entiende conda/micromamba
                       name: tesis
                       dependencies:
                         - python=3.11    ← Capa 1
                         - cuda=12.1      ← Capa 1
                         - pip:
                           - pandas       ← Capa 2
                           - torch        ← Capa 2
```

`environment.yaml` intentaba ser todo en uno — de ahí la mezcla y la confusión.
Funcionaba hasta que mezclabas `conda install` con `pip install` y el entorno
se corrompía silenciosamente.

La estrategia de mantener `requirements.txt` a mano con solo las dependencias
explícitas era correcta intuitivamente. `pyproject.toml` formaliza exactamente eso.

---

## `setup.py` y `setup.cfg` — por qué quizá no los conocías

Eran los archivos que hacían tu código **instalable como paquete** antes de que
existiera `pyproject.toml`. Si nunca instalaste tu propio código como paquete
(siempre corriste scripts sueltos sin hacer `pip install -e .`), nunca los necesitaste.

```python
# setup.py — un script Python ejecutable
from setuptools import setup, find_packages

setup(
    name="record_linkage",
    version="0.1.0",
    packages=find_packages(where="src"),
    install_requires=[
        "torch>=2.0",
        "transformers>=4.35",
    ],
)
```

```ini
# setup.cfg — la versión declarativa del mismo setup.py
[metadata]
name = record_linkage
version = 0.1.0

[options]
install_requires =
    torch>=2.0
    transformers>=4.35
```

El problema: había que mantener `setup.py` (o `setup.cfg`) **y** `requirements.txt`
sincronizados a mano — dos fuentes de verdad para las mismas dependencias,
inevitable que se desincronizaran.

`pyproject.toml` consolidó todo en un solo archivo estándar:

```
Antes                          Ahora
─────────────────────          ──────────────
requirements.txt               pyproject.toml
setup.py              →        (todo en uno)
setup.cfg
```

---

## pyproject.toml

Es el estándar oficial de Python (PEP 517/518) que reemplaza a tres archivos
que antes existían por separado: `requirements.txt`, `setup.py` y `setup.cfg`.
Todo en un solo archivo declarativo.

Tiene tres secciones principales:

### `[build-system]` — boilerplate, no se toca

Le dice a Python qué herramienta construye e instala tu paquete:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### `[project]` — el corazón

Metadatos y dependencias **directas** (solo las que tú explícitamente necesitas,
no las transitivas — esas las resuelve uv y las guarda en `uv.lock`):

```toml
[project]
name = "record-linkage"
version = "0.1.0"
description = "Hybrid neural architecture for entity resolution in clinical databases"
requires-python = ">=3.11"

dependencies = [
    "torch>=2.0",
    "transformers>=4.35",
    "sentence-transformers>=2.2",
    "pandas>=2.0",
    "pyarrow>=14.0",
    "python-dotenv>=1.0",
    "faiss-cpu>=1.7",
    "scikit-learn>=1.3",
    "numpy>=1.24",
]

[project.optional-dependencies]
dev = [
    "jupyter>=1.0",
    "ipykernel>=6.0",
    "pytest>=7.0",
    "ruff>=0.1",
]
```

Las dependencias `dev` son opcionales — solo las usas tú localmente,
no forman parte del paquete de producción.

### `[tool.hatch.build.targets.wheel]` — src layout

Le dice a hatchling dónde vive tu código:

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/record_linkage"]
```

Sin esto no sabe que usas el `src` layout y no encuentra tu paquete.

---

## ¿Puede uv reemplazar a micromamba?

Sí — uv moderno (v0.2+) puede crear entornos y manejar versiones de Python
sin micromamba:

```bash
# uv solo — sin micromamba
uv python install 3.11
uv venv --python 3.11
uv pip install torch transformers
```

**Pero hay un asterisco crítico: CUDA.**

CUDA no es un paquete Python — es una librería del sistema que vive en la
Capa 1. Para entrenar Transformers con GPU, micromamba garantiza que cuDNN
y las librerías CUDA estén correctamente alineadas con tu hardware.

```
uv solo          →  suficiente para proyectos CPU puros
micromamba + uv  →  necesario cuando hay GPU + CUDA de por medio
```

Para proyectos de ML con entrenamiento de modelos, la regla es clara:

```bash
# Capa 1 — micromamba (Python + CUDA del sistema)
micromamba create -n tesis python=3.11
micromamba activate tesis

# Capa 2 — uv (todos los paquetes Python)
uv pip install torch transformers sentence-transformers
uv pip install -e ".[dev]"   # instala record_linkage + dependencias dev
```

Alguien que usa solo uv probablemente trabaja en proyectos CPU-only
o no entrena modelos pesados. Para entrenar Transformers con una RTX 3050,
micromamba + uv es la combinación correcta.
