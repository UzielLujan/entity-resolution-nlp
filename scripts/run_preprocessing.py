"""
Script de entrada para el pipeline de limpieza de CSVs crudos del INER.

Uso:
    python scripts/run_preprocessing.py --perfil A    # Limpieza completa para INER
    python scripts/run_preprocessing.py --perfil B    # Mínima intervención para tesis
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Agregar src/ al path para que funcione desde cualquier directorio
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from record_linkage.config import RAW_FILES, PROCESSED_DIR, check_paths
from record_linkage.data.preprocessing import run_profile_a, run_profile_b


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de limpieza de CSVs crudos del INER",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python scripts/run_preprocessing.py --perfil A
  python scripts/run_preprocessing.py --perfil B
        """
    )
    parser.add_argument(
        '--perfil',
        choices=['A', 'B'],
        default='A',
        help="Perfil de ejecución: A=limpieza completa (INER), B=mínima intervención (tesis)"
    )
    parser.add_argument(
        '--check-paths',
        action='store_true',
        help="Verificar que existen todas las rutas necesarias y salir"
    )

    args = parser.parse_args()

    if args.check_paths:
        check_paths()
        return

    # Seleccionar función según perfil
    profile_fn = run_profile_a if args.perfil == 'A' else run_profile_b
    profile_name = f"Perfil {args.perfil}"

    print(f"\n{'='*60}")
    print(f"  Limpieza de CSVs INER — {profile_name}")
    print(f"{'='*60}\n")

    # Crear directorio de salida si no existe
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Procesar cada CSV
    for csv_name, raw_path in RAW_FILES.items():
        if not raw_path.exists():
            print(f"✗ {csv_name}: archivo no encontrado en {raw_path}")
            continue

        try:
            print(f"Procesando {csv_name}...", end=" ")
            df = pd.read_csv(raw_path)
            df_clean = profile_fn(df, csv_name)

            out_path = PROCESSED_DIR / f"{csv_name}_clean.csv"
            df_clean.to_csv(out_path, index=False)

            print(f"✓ ({len(df_clean)} registros)")
            print(f"  → {out_path.name}\n")
        except Exception as e:
            print(f"✗ Error: {e}\n")
            raise

    print(f"{'='*60}")
    print(f"  Limpieza completada — {profile_name}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
