"""CLI para construir dataset serializado (Perfil B — tesis).

Invocación:
    python scripts/run_dataset.py --output ~/Data/INER/processed/dataset.parquet --check-paths
    python scripts/run_dataset.py --output ~/Data/INER/processed/dataset.parquet

El script lee CSVs preprocesados con Perfil B, serializa registros con bloques semánticos,
asigna entity_ids basado en llave determinista, y guarda como .parquet para entrenamiento.
"""

import argparse
import sys
from pathlib import Path

# Agregar src/ al path para importar record_linkage
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from record_linkage.config import PROCESSED_DIR
from record_linkage.data.dataset import build_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Construye dataset serializado para entrenamiento (Perfil B — tesis)"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Ruta donde guardar dataset.parquet (ej: ~/Data/INER/processed/dataset.parquet)"
    )

    parser.add_argument(
        "--check-paths",
        action="store_true",
        help="Validar rutas antes de procesar"
    )

    args = parser.parse_args()

    output_path = Path(args.output).expanduser()
    processed_dir = Path(PROCESSED_DIR).expanduser()

    # Construir rutas a CSVs preprocesados (Perfil B)
    csv_names = [
        "INER_COVID19_Pacientes_DiagnosticoComorbilidad_clean",
        "INER_COVID19_CostoPacientes_Econo_clean",
        "INER_COVID19_TrabajoSocial_clean",
    ]

    csv_paths = [processed_dir / f"{name}.csv" for name in csv_names]

    # Validar rutas si se solicita
    if args.check_paths:
        print("🔍 Validando rutas...")
        all_exist = True
        for csv_path in csv_paths:
            exists = csv_path.exists()
            status = "✓" if exists else "✗"
            print(f"  {status} {csv_path}")
            if not exists:
                all_exist = False

        if not all_exist:
            print("\n❌ Algunas rutas no existen. Verifica PROCESSED_DIR en config.py")
            return 1

        print(f"✓ Dataset output: {output_path}\n")

    # Construir dataset
    print("🔨 Construyendo dataset...")
    try:
        df = build_dataset(
            csv_paths=csv_paths,
            output_path=output_path,
            source_db_names=["Comorbilidad", "Económico", "Trabajo Social"]
        )
        print(f"\n✅ Dataset completado exitosamente")
        print(f"   Archivo: {output_path}")
        return 0

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("   Asegúrate de que los CSVs fueron preprocesados con Perfil B")
        return 1
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
