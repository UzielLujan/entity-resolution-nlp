# src/record_linkage/config.py
# Configuración central del proyecto: entity resolution para INER. Todas las rutas del sistema se definen aquí.

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Raíz del repo (siempre relativa a este archivo, sube 2 niveles) ──────────────────────
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

# ── Carga de variables de entorno ────────────────────────────────────────────
_ENV_PATH = REPO_ROOT / ".env"
load_dotenv(_ENV_PATH)

# ── Raíz de datos externos (fuera del repo, nunca en git) ────────────────────
# Se lee del .env para que cada máquina apunte a su propia ruta.
_DATA_ROOT_VALUE = os.environ.get("INER_DATA_ROOT", "").strip()
if not _DATA_ROOT_VALUE:
    raise RuntimeError(
        "INER_DATA_ROOT no está configurado. "
        "Copia .env.example como .env y define una ruta absoluta."
    )
_DATA_ROOT_PATH = Path(_DATA_ROOT_VALUE).expanduser()
if not _DATA_ROOT_PATH.is_absolute():
    raise RuntimeError("INER_DATA_ROOT debe ser una ruta absoluta.")
DATA_ROOT: Path = _DATA_ROOT_PATH.resolve()

# ── Entrada de datos preparados ─────────────────────────────────────────────────────────
PREPARED_DATA_DIR: Path = DATA_ROOT / "processed" / "default" / "output"
DEFAULT_VARIANT: str = "tok_skipnull"
SUPPORTED_VARIANTS: tuple[str, ...] = (
    "tok_skipnull",
    "tok_keepnull",
    "notok_skipnull",
    "notok_keepnull",
)

# ── Artefactos del pipeline neuronal ──────────────────────────────────────────
MODELING_DIR: Path = DATA_ROOT / "modeling"
MODELING_DATA_DIR: Path = MODELING_DIR / "data"

MODELS_DIR: Path = MODELING_DIR / "models"
PRETRAINED_MODELS_DIR: Path = MODELS_DIR / "pretrained"
BIENCODER_MODELS_DIR: Path = MODELS_DIR / "biencoder"
CROSSENCODER_MODELS_DIR: Path = MODELS_DIR / "crossencoder"

EMBEDDINGS_DIR: Path = MODELING_DIR / "embeddings"

OUTPUTS_DIR: Path = MODELING_DIR / "outputs"
EVALUATION_DIR: Path = OUTPUTS_DIR / "evaluation"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
DIAGNOSTICS_DIR: Path = OUTPUTS_DIR / "diagnostics"

def _validate_variant(variant: str) -> str:
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(
            f"Variante no soportada: {variant}. "
            f"Opciones: {', '.join(SUPPORTED_VARIANTS)}"
        )
    return variant

def prepared_dataset_path(variant: str = DEFAULT_VARIANT) -> Path:
    """Dataset serializado producido por el pipeline de datos."""
    return PREPARED_DATA_DIR / _validate_variant(variant) / "dataset.parquet"

def variant_data_dir(variant: str = DEFAULT_VARIANT) -> Path:
    """Directorio de datos derivados para una variante."""
    return MODELING_DATA_DIR / _validate_variant(variant)

def split_path(variant: str = DEFAULT_VARIANT) -> Path:
    """Dataset con asignación train/val/test para una variante."""
    return variant_data_dir(variant) / "split.parquet"

def pairs_path(split_name: str, variant: str = DEFAULT_VARIANT) -> Path:
    """Pares de train, val o test para el Cross-Encoder."""
    if split_name not in {"train", "val", "test"}:
        raise ValueError("split_name debe ser train, val o test")
    return variant_data_dir(variant) / f"pairs_{split_name}.parquet"

def embeddings_path(variant: str = DEFAULT_VARIANT) -> Path:
    """Embeddings exportados para una variante."""
    return EMBEDDINGS_DIR / _validate_variant(variant) / "embeddings.parquet"

def biencoder_run_dir(run_name: str, variant: str = DEFAULT_VARIANT) -> Path:
    """Directorio de un run del Bi-Encoder."""
    return BIENCODER_MODELS_DIR / _validate_variant(variant) / run_name

def crossencoder_run_dir(run_name: str, variant: str = DEFAULT_VARIANT) -> Path:
    """Directorio de un run del Cross-Encoder."""
    return CROSSENCODER_MODELS_DIR / _validate_variant(variant) / run_name

# ── Validación del entorno ────────────────────────────────────────────────────
def check_paths() -> None:
    """Verifica la entrada mínima y muestra las rutas derivadas del pipeline."""
    print("── Configuración del pipeline neuronal ──────────────────────")
    print(f"  REPO_ROOT:      {REPO_ROOT}")
    print(f"  INER_DATA_ROOT: {DATA_ROOT}")

    errors = []
    if not DATA_ROOT.is_dir():
        errors.append(f"INER_DATA_ROOT no existe o no es un directorio: {DATA_ROOT}")

    print("\n── Datasets preparados ──────────────────────────────────────")
    for variant in SUPPORTED_VARIANTS:
        path = prepared_dataset_path(variant)
        required = variant == DEFAULT_VARIANT
        status = "OK" if path.is_file() else "FALTA"
        label = "requerido" if required else "opcional"
        print(f"  {status:5}  {variant:18} ({label})  {path}")
        if required and not path.is_file():
            errors.append(f"Falta el dataset por default: {path}")

    print("\n── Salidas del pipeline neuronal ────────────────────────────")
    for name, path in {
        "modeling": MODELING_DIR,
        "datos": MODELING_DATA_DIR,
        "modelos": MODELS_DIR,
        "embeddings": EMBEDDINGS_DIR,
        "outputs": OUTPUTS_DIR,
    }.items():
        status = "EXISTE" if path.is_dir() else "SE CREARÁ"
        print(f"  {status:9}  {name:10} {path}")

    if errors:
        raise FileNotFoundError("\n".join(errors))

    print("\nConfiguración válida para ejecutar la variante por default.")


if __name__ == "__main__":
    try:
        check_paths()
    except FileNotFoundError as exc:
        print(f"\nERROR:\n{exc}")
        raise SystemExit(1) from exc
