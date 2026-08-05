# src/record_linkage/config.py
#
# Punto central de configuración del proyecto de tesis.
# Todas las rutas del sistema se definen aquí.

# Como importar:
#   from record_linkage.config import DATASET_PARQUET, SPLITS_DIR

from pathlib import Path
from dotenv import load_dotenv
import os

# ── Raíz del repo (siempre relativa a este archivo, sube 2 niveles en la jerarquía de directorios) ──────────────────────
# Funciona sin importar desde dónde se ejecute el código
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

# ── Carga de variables de entorno ────────────────────────────────────────────
# Busca el .env en la raíz del repo
_ENV_PATH = REPO_ROOT / ".env"
load_dotenv(_ENV_PATH)

# ── Raíz de datos externos (fuera del repo, nunca en git) ────────────────────
# Se lee del .env para que cada máquina apunte a su propia ruta.
# Si no existe la variable, cae al default ~/Data/INER
_DATA_ROOT_DEFAULT = Path.home() / "Data" / "INER"
DATA_ROOT: Path = Path(os.environ.get("INER_DATA_ROOT", _DATA_ROOT_DEFAULT))

# ── Rutas internas al repo ────────────────────────────────────────────────────
DOCS_DIR:      Path = REPO_ROOT / "docs"
NOTEBOOKS_DIR: Path = REPO_ROOT / "notebooks"
SCRIPTS_DIR:   Path = REPO_ROOT / "scripts"

# ── Subdirectorios de datos ───────────────────────────────────────────────────
PROCESSED_DIR:    Path = DATA_ROOT / "processed"
MODELS_DIR:       Path = DATA_ROOT / "models"
EMBEDDINGS_DIR:   Path = DATA_ROOT / "embeddings"

# ── Outputs del proyecto (figuras, métricas, evaluación) ────────────────────
OUTPUTS_DIR:      Path = DATA_ROOT / "outputs"
FIGURES_DIR:      Path = OUTPUTS_DIR / "figures"
TRAINING_DIR:     Path = OUTPUTS_DIR / "training"
EVALUATION_DIR:   Path = OUTPUTS_DIR / "evaluation"

# ── Contrato consultoría → tesis ───────────────────────────────────────────────
# La consultoría (repo consultoria-iner) produce:
#   <DATA_ROOT>/processed/default/output/<variant>/dataset.parquet
# La tesis consume esos artefactos directamente: no preprocesa, no serializa,
# no construye ground truth. Las variantes posibles:
#   tok_skipnull, tok_keepnull, notok_skipnull, notok_keepnull
CONSULTORIA_OUTPUT_DIR: Path = PROCESSED_DIR / "default" / "output"
DEFAULT_VARIANT: str = "tok_skipnull"
DATASET_PARQUET: Path = CONSULTORIA_OUTPUT_DIR / DEFAULT_VARIANT / "dataset.parquet"
ENTITY_IDS_PARQUET: Path = CONSULTORIA_OUTPUT_DIR / "entity_ids.parquet"

# ── Artefactos derivados por la tesis ──────────────────────────────────────────
# Los splits son artefactos de la tesis, no de consultoría: viven en su propio
# directorio y NO contaminan el output de consultoría.
SPLITS_DIR: Path = DATA_ROOT / "tesis" / "splits"


# ── Validación opcional (útil al arrancar un script o notebook) ───────────────
def check_paths() -> None:
    """Imprime el estado de todas las rutas críticas del proyecto."""
    paths = {
        "REPO_ROOT":            REPO_ROOT,
        "DOCS_DIR":             DOCS_DIR,
        "NOTEBOOKS_DIR":        NOTEBOOKS_DIR,
        "DATA_ROOT":            DATA_ROOT,
        "PROCESSED_DIR":        PROCESSED_DIR,
        "CONSULTORIA_OUTPUT_DIR": CONSULTORIA_OUTPUT_DIR,
        "DATASET_PARQUET":      DATASET_PARQUET,
        "SPLITS_DIR":           SPLITS_DIR,
        "MODELS_DIR":           MODELS_DIR,
        "EMBEDDINGS_DIR":       EMBEDDINGS_DIR,
        "OUTPUTS_DIR":          OUTPUTS_DIR,
        "FIGURES_DIR":          FIGURES_DIR,
        "TRAINING_DIR":         TRAINING_DIR,
        "EVALUATION_DIR":       EVALUATION_DIR,
    }
    print("── Rutas del proyecto ──────────────────────")
    for name, path in paths.items():
        status = "EXISTE" if path.exists() else "NO EXISTE"
        print(f"  {status}  {name}: {path}")


if __name__ == "__main__":
    check_paths()