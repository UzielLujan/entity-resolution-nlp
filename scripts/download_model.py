"""Descarga modelos desde Hugging Face o recarga copias locales para prepararlos como SentenceTransformer.

El mismo flujo se aplica a los 3 modelos seleccionados:
1. Descarga o recarga el modelo, registra en su propio tokenizador los tokens especiales [BLK_*], [COL] y [VAL],
2. Redimensiona su matriz de embeddings y
3. Guarda el artefacto listo para usar.

El redimensionamiento asigna vectores iniciales a los tokens nuevos. Durante el
entrenamiento del Bi-Encoder, train_biencoder.py reemplaza esos vectores con una
inicializacion semantica basada en palabras ancla.

Uso:
    .venv/bin/python scripts/download_model.py --model dccuchile/bert-base-spanish-wwm-cased --name BETO
    .venv/bin/python scripts/download_model.py --model PlanTL-GOB-ES/roberta-base-biomedical-clinical-es --name RoBERTa-biomedical
    .venv/bin/python scripts/download_model.py --all

Los modelos se guardan en $INER_DATA_ROOT/modeling/models/pretrained/<name>/
para transferirlos al cluster sin necesidad de internet en los nodos de cómputo.
"""

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from record_linkage.config import PRETRAINED_MODELS_DIR
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling, Transformer


KNOWN_MODELS = {
    "BETO": "dccuchile/bert-base-spanish-wwm-cased",
    # Modelo biomédico-clínico, más relevante para el dominio INER
    "RoBERTa-biomedical": "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es",
    # Modelo ya fine-tuned para similitud, multilingüe (incluye español), como su nombre indica fue especializado para parafraseo, por lo que su dominio de origen no es el del INER. Se incluye como baseline de similitud semántica.
    "paraphrase-multilingual": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
}


def _load_transformer(model_id: str) -> Transformer:
    """Carga un modelo de Hugging Face como módulo Transformer de sentence-transformers.

    Primero intenta la carga estándar. Si el modelo falla porque su config.json no
    contiene el campo `model_type`, aplica una ruta alternativa específicamente
    para modelos RoBERTa. Los pesos del modelo no se modifican; solo se corrige la metadata
    necesaria para que el cargador genérico pueda identificar la arquitectura.

    Args:
        model_id: Identificador de Hugging Face o ruta local al modelo.

    Returns:
        Un módulo Transformer compatible con SentenceTransformer.

    Raises:
        ValueError: Si la carga falla por una razón distinta a la ausencia
            de `model_type`.
    """

    # Ruta normal: sentence-transformers identifica la arquitectura a partir
    # de la configuración del modelo y carga automáticamente backbone y tokenizer.
    try:
        return Transformer(model_id)
    except ValueError as e:
        if "model_type" not in str(e):
            raise

    # Importaciones locales: solo son necesarias para la ruta de compatibilidad
    # con modelos RoBERTa cuya configuración no puede leer AutoConfig.
    from transformers import RobertaModel, RobertaTokenizerFast

    print("  Cargando explícitamente como RoBERTa (sin AutoConfig)...")

    hf_model = RobertaModel.from_pretrained(model_id)
    tokenizer = RobertaTokenizerFast.from_pretrained(model_id)

    # Copia temporal con una configuración compatible
    patched_dir = Path(tempfile.mkdtemp(prefix="st_roberta_bne_"))
    hf_model.config.model_type = "roberta"

    # Exportamos pesos, configuración y tokenizer al directorio corregido.
    hf_model.save_pretrained(patched_dir)
    tokenizer.save_pretrained(patched_dir)

    # Reutilizamos el flujo estándar de sentence-transformers usando la copia
    # temporal ya corregida.
    print("  Cargando desde el directorio temporal corregido...")
    return Transformer(str(patched_dir))


SPECIAL_TOKENS = [
    "[BLK_ID]", "[BLK_ADMIN]", "[BLK_CLIN]", "[BLK_GEO]", "[BLK_SOCIO]",
    "[COL]", "[VAL]",
]


def download_model(model_id: str, output_name: str) -> Path:
    """Prepara y guarda un modelo con sus tokens especiales registrados.

    Cada modelo conserva su tokenizador propio.
    Si el modelo local ya contiene los tokens, no se vuelven a añadir. Un directorio
    de salida incompleto se rechaza explícitamente para no cargarlo como modelo.
    """
    output_dir = PRETRAINED_MODELS_DIR / output_name

    if output_dir.exists():
        # Si ya existe un modelo completo, se reutiliza sin descargarlo de nuevo.
        if not (output_dir / "modules.json").is_file():
            raise RuntimeError(
                f"El directorio de salida existe pero no contiene un modelo completo: {output_dir}. "
                "Elimínalo antes de reintentar la descarga."
            )
        print(f"Cargando desde disco: {output_dir}")
        model = SentenceTransformer(str(output_dir))
    else:
        # Primera preparación: cargar el backbone y construir el modelo completo
        # con una capa de mean pooling.
        print(f"Descargando: {model_id}")
        print(f"  → {output_dir}")
        transformer = _load_transformer(model_id)
        pooling = Pooling(
            transformer.get_embedding_dimension(),
            pooling_mode="mean",
        )
        model = SentenceTransformer(modules=[transformer, pooling])

    tokenizer = model.tokenizer
    # Añadir los tokens especiales al vocabulario del modelo.
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    # Ampliar la matriz de embeddings para que incluya los tokens recién registrados.
    model._first_module().auto_model.resize_token_embeddings(len(tokenizer))
    # Guardar el modelo completo, incluido el tokenizador actualizado.
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
