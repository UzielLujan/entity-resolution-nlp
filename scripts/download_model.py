"""Descarga modelos de HuggingFace Hub y los guarda localmente.

Uso:
    python scripts/download_model.py --model dccuchile/bert-base-spanish-wwm-cased --name BETO
    python scripts/download_model.py --model PlanTL-GOB-ES/roberta-base-bne --name RoBERTa-bne
    python scripts/download_model.py --all

Los modelos se guardan en ~/Data/INER/models/pretrained/<name>/
para transferirlos al cluster sin necesidad de internet en los nodos de cómputo.
"""

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from record_linkage.config import MODELS_DIR
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling, Transformer

KNOWN_MODELS = {
    "BETO": "dccuchile/bert-base-spanish-wwm-cased",
    # roberta-base-bne fue removido de HuggingFace Hub (repo vacío desde 2026-04-28)
    # Se usa el modelo biomédico-clínico del mismo grupo, más relevante para el dominio INER
    "RoBERTa-biomedical": "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es",
    # Baseline de similitud semántica: ya fine-tuned para similitud, multilingüe (incluye español)
    # Sirve como cota superior pre-MNRL para diagnosticar si las métricas bajas son del modelo o los datos
    "paraphrase-multilingual": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
}


def _load_transformer(model_id: str) -> Transformer:
    """Carga un backbone HuggingFace como Transformer de sentence-transformers.

    Maneja modelos sin 'model_type' en config.json (e.g. PlanTL-GOB-ES/roberta-base-bne)
    cargándolos directamente con RobertaModel, que conoce su propia arquitectura,
    y re-exportando con config.json correcto antes de cargar como Transformer.
    """
    try:
        return Transformer(model_id)
    except ValueError as e:
        if "model_type" not in str(e):
            raise

    # Cargar con RobertaModel directamente (bypassa AutoConfig)
    from transformers import RobertaModel, RobertaTokenizerFast
    print("  Cargando con RobertaModel (sin AutoConfig)...")
    hf_model = RobertaModel.from_pretrained(model_id)
    tokenizer = RobertaTokenizerFast.from_pretrained(model_id)

    # Guardar con config.json que incluye model_type
    patched_dir = Path(tempfile.mkdtemp(prefix="st_roberta_bne_"))
    hf_model.config.model_type = "roberta"
    hf_model.save_pretrained(patched_dir)
    tokenizer.save_pretrained(patched_dir)

    print("  Cargando desde directorio parcheado...")
    return Transformer(str(patched_dir))


SPECIAL_TOKENS = [
    "[BLK_ID]", "[BLK_ADMIN]", "[BLK_CLIN]", "[BLK_GEO]", "[BLK_SOCIO]",
    "[COL]", "[VAL]",
]


def download_model(model_id: str, output_name: str) -> Path:
    """Descarga (o carga desde disco) un backbone y lo guarda con tokens especiales registrados."""
    output_dir = MODELS_DIR / "pretrained" / output_name

    if output_dir.exists():
        print(f"Cargando desde disco: {output_dir}")
        model = SentenceTransformer(str(output_dir))
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Descargando: {model_id}")
        print(f"  → {output_dir}")
        transformer = _load_transformer(model_id)
        pooling = Pooling(
            transformer.get_embedding_dimension(),
            pooling_mode="mean",
        )
        model = SentenceTransformer(modules=[transformer, pooling])

    tokenizer = model.tokenizer
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    model._first_module().auto_model.resize_token_embeddings(len(tokenizer))
    model.save(str(output_dir))

    print(f"  ✓ Guardado en {output_dir} ({num_added} tokens especiales añadidos)")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Descarga modelos de HuggingFace Hub")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", help="ID de HuggingFace Hub (e.g. dccuchile/bert-base-spanish-wwm-cased)")
    group.add_argument("--all", action="store_true", help="Descarga todos los modelos conocidos")
    parser.add_argument("--name", help="Nombre local del modelo (requerido con --model)")

    args = parser.parse_args()

    if args.all:
        for name, model_id in KNOWN_MODELS.items():
            download_model(model_id, name)
    else:
        if not args.name:
            print("Error: --name es requerido cuando se usa --model")
            return 1
        download_model(args.model, args.name)

    return 0


if __name__ == "__main__":
    sys.exit(main())
