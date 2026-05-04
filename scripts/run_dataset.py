"""CLI para construir dataset serializado (Perfiles Tesis).

Invocaciones comunes:
    python scripts/run_dataset.py --perfil tesis0 --no-special-tokens            # Dataset para Zero-Shot
    python scripts/run_dataset.py --perfil tesis0 --no-special-tokens --solo-nombres # Solo nombres (Sin Tokens)
    python scripts/run_dataset.py --perfil tesis1                                # Dataset estándar para fine-tuning
    python scripts/run_dataset.py --perfil tesis2 --check-paths                  # Validar rutas

Lee CSVs preprocesados desde PROCESSED_DIR/<perfil>/, serializa registros con bloques
semánticos, asigna entity_ids y guarda dataset.parquet en la misma subcarpeta.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from record_linkage.config import PROCESSED_DIR
from record_linkage.data.dataset import assign_entity_ids, build_dataset


def _build_nombres_dataset(csv_paths, source_db_names, output_path):
    """Genera parquet con text = solo el campo de nombre por registro.

    Reutiliza assign_entity_ids para que entity_id sea idéntico al del parquet sin_tokens.
    Salida: record_id, source_db, text ("col: valor"), entity_id.
    """
    import pandas as pd

    _COLS = {
        "Comorbilidad":   {"exp": "expediente",  "nombre": "nombre",              "partes": None},
        "Económico":      {"exp": "EXP",          "nombre": "NOMBRE_DEL_PACIENTE", "partes": None},
        "Trabajo Social": {"exp": "EXPEDIENTE",   "nombre": "NOMBRE_COMPLETO",     "partes": ["APELLIDO PATERNO", "APELLIDO MATERNO", "NOMBRE"]},
    }

    dfs = []
    for csv_path, source_db in zip(csv_paths, source_db_names):
        df = pd.read_csv(Path(str(csv_path)))
        df["source_db"] = source_db
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df.insert(0, "record_id", range(len(df)))

    def _nombre_raw(row):
        cfg = _COLS.get(row["source_db"], {})
        col = cfg.get("nombre")
        if col and col in row.index and pd.notna(row.get(col)):
            return str(row[col]).strip()
        partes = cfg.get("partes") or []
        tokens = [str(row[p]).strip() for p in partes
                  if p in row.index and pd.notna(row.get(p)) and str(row.get(p)).strip()]
        return " ".join(tokens)

    def _nombre_text(row):
        cfg = _COLS.get(row["source_db"], {})
        col = cfg.get("nombre")
        nombre = _nombre_raw(row)
        label = col if (col and col in row.index and pd.notna(row.get(col))) else "nombre"
        return f"{label}: {nombre}" if nombre else ""

    def _expediente(row):
        cfg = _COLS.get(row["source_db"], {})
        return row.get(cfg.get("exp"))

    df_entity = df[["record_id", "source_db"]].copy()
    df_entity["expediente"] = df.apply(_expediente, axis=1)
    df_entity["nombre"] = df.apply(_nombre_raw, axis=1)
    df_entity = assign_entity_ids(df_entity)

    df_out = df[["record_id", "source_db"]].copy()
    df_out["text"] = df.apply(_nombre_text, axis=1)
    df_out = df_out.merge(df_entity[["record_id", "entity_id"]], on="record_id")

    output_path = Path(str(output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(output_path, engine="pyarrow", index=False, compression="snappy")

    print(f"✓ Dataset nombres guardado: {output_path}")
    print(f"  Registros:        {len(df_out):,}")
    print(f"  Entidades únicas: {df_out['entity_id'].nunique():,}")


def main():
    parser = argparse.ArgumentParser(
        description="Construye dataset serializado para entrenamiento (Perfil B — tesis)"
    )

    parser.add_argument(
        "--perfil",
        choices=["tesis0", "tesis1", "tesis2", "iner"],
        default="tesis1",
        help="Perfil: tesis0 (base), tesis1 (mínima intervención), tesis2 (limpieza + renombrado), iner (completa)"
    )

    parser.add_argument(
        "--no-special-tokens",
        action="store_true",
        help="Desactiva tokens [BLK_*] y usa serialización Clave:Valor (modo Sin Tokens)"
    )

    parser.add_argument(
        "--check-paths",
        action="store_true",
        help="Validar rutas antes de procesar"
    )

    parser.add_argument(
        "--solo-nombres",
        action="store_true",
        help="Genera dataset con text = solo campo nombre (requiere --perfil tesis0 y --no-special-tokens). Salida en tesis0_sin_tokens_solo_nombres/"
    )

    args = parser.parse_args()

    use_block_tokens = not args.no_special_tokens
    source_dir = Path(PROCESSED_DIR) / args.perfil.lower()

    # Evitar sobrescribir el dataset de fine-tuning con el de zero-shot
    if args.no_special_tokens:
        profile_dir = Path(PROCESSED_DIR) / f"{args.perfil.lower()}_sin_tokens"
    else:
        profile_dir = Path(PROCESSED_DIR) / args.perfil.lower()

    output_path = profile_dir / "dataset.parquet"

    csv_names = ["comorbilidad_clean", "econo_clean", "trabajo_social_clean"]
    csv_paths = [source_dir / f"{name}.csv" for name in csv_names]

    if args.check_paths:
        print(f"Validando rutas (Perfil {args.perfil})...")
        all_exist = True
        for csv_path in csv_paths:
            exists = csv_path.exists()
            status = "✓" if exists else "✗"
            print(f"  {status} {csv_path}")
            if not exists:
                all_exist = False

        if not all_exist:
            print(f"\nError: Algunos CSVs no existen. Ejecuta primero:")
            print(f"  python scripts/run_preprocessing.py --perfil {args.perfil}")
            return 1

        print(f"Dataset output: {output_path}\n")

    if args.solo_nombres:
        if args.perfil != "tesis0" or not args.no_special_tokens:
            print("Error: --solo-nombres solo es válido con --perfil tesis0 y --no-special-tokens")
            return 1
        output_nombres = Path(PROCESSED_DIR) / "tesis0_sin_tokens_solo_nombres" / "dataset.parquet"
        print("Construyendo dataset solo-nombres...")
        _build_nombres_dataset(csv_paths, ["Comorbilidad", "Económico", "Trabajo Social"], output_nombres)
        return 0

    print(f"Construyendo dataset (Perfil {args.perfil})...")
    try:
        build_dataset(
            csv_paths=csv_paths,
            output_path=output_path,
            source_db_names=["Comorbilidad", "Económico", "Trabajo Social"],
            use_block_tokens=use_block_tokens,
        )
        print(f"\nDataset completado exitosamente")
        print(f"   Archivo: {output_path}")
        return 0

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print(f"  Asegúrate de preprocesar primero con --perfil {args.perfil}")
        return 1
    except Exception as e:
        print(f"\nError inesperado: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
