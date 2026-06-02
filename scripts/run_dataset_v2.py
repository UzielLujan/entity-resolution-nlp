"""CLI para el pipeline de etiquetado v2.

Uso legacy (tesis1, flat):
    python scripts/run_dataset_v2.py --step classify --perfil tesis1
    python scripts/run_dataset_v2.py --step finalize --perfil tesis1

Uso tesis (workspace canónico, estructura clean/interim/output):
    # Solo --step finalize está permitido para tesis (classify destruiría la verdad base).
    # Las 4 variantes del experimento 2×2:
    python scripts/run_dataset_v2.py --step finalize --perfil tesis
    python scripts/run_dataset_v2.py --step finalize --perfil tesis --skip-null
    python scripts/run_dataset_v2.py --step finalize --perfil tesis --no-special-tokens
    python scripts/run_dataset_v2.py --step finalize --perfil tesis --no-special-tokens --skip-null

    # Cada variante cae en su subdir bajo tesis/output/:
    #   tok_keepnull/dataset.parquet
    #   tok_skipnull/dataset.parquet
    #   notok_keepnull/dataset.parquet
    #   notok_skipnull/dataset.parquet
    # run_splitting.py escribe dataset_split.parquet en el mismo subdir.

Umbrales (calibrados empíricamente — ver docs/hallazgos_duplicados.md):
    --umbral-jw   0.88   Jaro-Winkler mínimo para metrica_clasica
    --umbral-lev  0.85   Levenshtein ratio mínimo para metrica_clasica
"""

import argparse
import sys
from pathlib import Path

# Agregar src/ al path para que funcione desde cualquier directorio
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from record_linkage.config import PROCESSED_DIR, perfil_paths
from record_linkage.data.dataset_v2 import build_dataset_v2, _step_finalize


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline de etiquetado v2 — clasificación de pares y generación de dataset."
    )
    parser.add_argument(
        "--step",
        choices=["classify", "finalize"],
        required=True,
        help="'classify': genera pairs_classified.parquet + pairs_for_review.xlsx (editable). "
             "'finalize': lee pairs_for_review.xlsx editado y produce dataset_v2.parquet.",
    )
    parser.add_argument(
        "--perfil",
        type=str,
        default="tesis1",
        help="Perfil de preprocessing (default: tesis1). Determina la ruta "
             "DATA_DIR/processed/<perfil>/ de la que se leen CSVs y donde se escriben artefactos. "
             "Se ignora si pasas --econo / --comor / --ts / --output explícitos.",
    )
    parser.add_argument(
        "--econo",
        type=Path,
        default=None,
        help="Ruta al CSV limpio de Económico (default: DATA_DIR/processed/econo_clean.csv)",
    )
    parser.add_argument(
        "--comor",
        type=Path,
        default=None,
        help="Ruta al CSV limpio de Comorbilidad (default: DATA_DIR/processed/comorbilidad_clean.csv)",
    )
    parser.add_argument(
        "--ts",
        type=Path,
        default=None,
        help="Ruta al CSV limpio de Trabajo Social (default: DATA_DIR/processed/trabajo_social_clean.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directorio de salida (default: DATA_DIR/processed/)",
    )
    parser.add_argument(
        "--umbral-jw",
        type=float,
        default=0.88,
        help="Umbral Jaro-Winkler para metrica_clasica (default: 0.88 — calibrado)",
    )
    parser.add_argument(
        "--umbral-lev",
        type=float,
        default=0.85,
        help="Umbral Levenshtein para metrica_clasica (default: 0.85 — calibrado)",
    )
    parser.add_argument(
        "--no-special-tokens",
        action="store_true",
        help="Serializar sin tokens [BLK_*] (modo zero-shot)",
    )
    parser.add_argument(
        "--skip-null",
        action="store_true",
        help="Omitir columnas con valor nulo en la serialización (texto más compacto). "
             "Eje independiente de --no-special-tokens; los 4 combos definen el experimento 2×2.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Nombre del subdirectorio de la variante en output/ (solo perfil tesis). "
             "Si no se da, se deriva de la config: <tok|notok>_<keepnull|skipnull>. "
             "El parquet final siempre se llama dataset.parquet dentro de ese dir.",
    )
    return parser.parse_args()


def resolve_paths(args):
    """Resuelve rutas a partir de args y del perfil.

    Las rutas base vienen de `record_linkage.config.PROCESSED_DIR`, que respeta
    la variable de entorno `INER_DATA_ROOT` (default: `~/Data/INER`).
    """
    perfil_dir = PROCESSED_DIR / args.perfil

    econo = args.econo or perfil_dir / "econo_clean.csv"
    comor = args.comor or perfil_dir / "comorbilidad_clean.csv"
    ts    = args.ts    or perfil_dir / "trabajo_social_clean.csv"
    output_dir = args.output or perfil_dir

    return [econo, comor, ts], output_dir


def _derive_variant_dir(use_block_tokens: bool, skip_null: bool) -> str:
    """Nombre del subdirectorio de la variante en output/."""
    tok = "tok" if use_block_tokens else "notok"
    nul = "skipnull" if skip_null else "keepnull"
    return f"{tok}_{nul}"


def _run_tesis_finalize(args):
    """Path canónico para perfil tesis: re-serializa desde clean/ con config, escribe a output/<variante>/."""
    if args.step == "classify":
        print("ERROR: --step classify está bloqueado para perfil tesis (sobreescribiría pairs_for_review.xlsx).")
        print("       La verdad base vive en tesis/interim/pairs_for_review.xlsx — no se regenera.")
        print("       Usa --step finalize con flags de serialización para las variantes del 2×2.")
        return 1

    paths = perfil_paths("tesis")
    csv_names = ["econo_clean.csv", "comorbilidad_clean.csv", "trabajo_social_clean.csv"]
    csv_paths = [paths["clean"] / name for name in csv_names]
    for p in csv_paths:
        if not p.exists():
            raise FileNotFoundError(f"CSV limpio no encontrado: {p}")

    use_block_tokens = not args.no_special_tokens
    skip_null = args.skip_null

    # Cada variante vive en output/<variant>/, con nombres canónicos dataset.parquet y dataset_split.parquet.
    # Esto unifica la convención con perfiles legacy (donde dataset.parquet era el único archivo final)
    # y permite que run_splitting.py derive su salida sin sufijos de variante.
    variant_dir_name = args.output_name or _derive_variant_dir(use_block_tokens, skip_null)
    variant_dir = paths["output"] / variant_dir_name
    output_path = variant_dir / "dataset.parquet"

    print(f"\n=== Finalize para perfil tesis ===")
    print(f"  Config: use_block_tokens={use_block_tokens}, skip_null={skip_null}")
    print(f"  Salida: {output_path}")

    _step_finalize(
        records_path=paths["interim"] / "records_interim.parquet",
        xlsx_path=paths["interim"] / "pairs_for_review.xlsx",
        output_path=output_path,
        csv_paths=csv_paths,
        use_block_tokens=use_block_tokens,
        skip_null=skip_null,
    )
    return 0


def main():
    args = parse_args()

    if args.perfil == "tesis":
        return _run_tesis_finalize(args)

    # Legacy path: tesis1, tesis0, etc. — estructura plana
    csv_paths, output_dir = resolve_paths(args)

    if args.step == "classify":
        for path in csv_paths:
            if not path.exists():
                raise FileNotFoundError(
                    f"CSV no encontrado: {path}\n"
                    "Especificar ruta con --econo / --comor / --ts o configurar DATA_DIR en .env"
                )

    build_dataset_v2(
        csv_paths=csv_paths,
        output_dir=output_dir,
        use_block_tokens=not args.no_special_tokens,
        step=args.step,
        umbral_jw=args.umbral_jw,
        umbral_lev=args.umbral_lev,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
