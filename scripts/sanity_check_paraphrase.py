"""Verificación de que paraphrase-multilingual funciona como modelo de similitud semántica.

Evalúa el modelo sobre pares de paráfrasis en español y pares negativos (no relacionados).
Similitud coseno esperada: ~0.85+ en paráfrasis, ~0.0–0.3 en negativos.

Un resultado alto en paráfrasis confirma que el modelo resuelve su task original.
Si el mismo modelo falla en Record Linkage del INER, el problema es de dominio, no del modelo.

Uso:
    python scripts/sanity_check_paraphrase.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from record_linkage.config import MODELS_DIR
from record_linkage.models.biencoder import build_biencoder, encode_texts

MODEL_NAME = "paraphrase-multilingual"

# Pares de paráfrasis obvias en español
PARAPHRASE_PAIRS = [
    ("El paciente tiene diabetes mellitus tipo 2.", "El enfermo padece diabetes."),
    ("Ingresó al hospital el 15 de marzo.", "Fue admitido el 15 de marzo."),
    ("Presenta obesidad severa.", "El paciente tiene obesidad mórbida."),
    ("Falleció durante su estancia hospitalaria.", "El paciente murió en el hospital."),
    ("Reside en el Estado de México.", "Su lugar de residencia es el Estado de México."),
    ("Tiene 67 años de edad.", "El paciente es un adulto mayor de 67 años."),
    ("Fue dado de alta por mejoría.", "Egresó del hospital al mejorar su condición."),
    ("Padece insuficiencia renal crónica.", "Tiene enfermedad renal crónica."),
    ("Trabaja como empleado doméstico.", "Su ocupación es el servicio doméstico."),
    ("No cuenta con derechohabiencia.", "El paciente no tiene seguro médico."),
]

# Pares negativos: frases sin relación semántica
NEGATIVE_PAIRS = [
    ("El paciente tiene diabetes mellitus tipo 2.", "Fue dado de alta por mejoría."),
    ("Ingresó al hospital el 15 de marzo.", "Padece insuficiencia renal crónica."),
    ("Reside en el Estado de México.", "Presenta obesidad severa."),
    ("Trabaja como empleado doméstico.", "Falleció durante su estancia hospitalaria."),
    ("Tiene 67 años de edad.", "No cuenta con derechohabiencia."),
]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # embeddings ya normalizados


def main():
    model_path = MODELS_DIR / "pretrained" / MODEL_NAME
    if not model_path.exists():
        print(f"ERROR: Modelo no encontrado en {model_path}")
        print("Ejecuta primero: python scripts/download_model.py --model paraphrase-multilingual")
        return 1

    print(f"Cargando {MODEL_NAME} desde {model_path}...")
    model = build_biencoder(model_path)
    model.max_seq_length = 512
    print(f"Modelo cargado — dim={model.get_embedding_dimension()}\n")

    all_texts = [t for pair in PARAPHRASE_PAIRS + NEGATIVE_PAIRS for t in pair]
    embeddings = encode_texts(model, all_texts, batch_size=32, show_progress=False)

    n_para = len(PARAPHRASE_PAIRS)
    para_embs = embeddings[: n_para * 2].reshape(n_para, 2, -1)
    neg_embs  = embeddings[n_para * 2 :].reshape(len(NEGATIVE_PAIRS), 2, -1)

    print("=" * 70)
    print("PARES DE PARÁFRASIS")
    print("=" * 70)
    para_sims = []
    for (a, b), embs in zip(PARAPHRASE_PAIRS, para_embs):
        sim = cosine_similarity(embs[0], embs[1])
        para_sims.append(sim)
        marca = "✓" if sim >= 0.75 else "✗"
        print(f"  {marca} {sim:.4f}  \"{a}\"")
        print(f"           \"{b}\"")

    print()
    print("=" * 70)
    print("PARES NEGATIVOS (no relacionados)")
    print("=" * 70)
    neg_sims = []
    for (a, b), embs in zip(NEGATIVE_PAIRS, neg_embs):
        sim = cosine_similarity(embs[0], embs[1])
        neg_sims.append(sim)
        print(f"         {sim:.4f}  \"{a}\"")
        print(f"                    \"{b}\"")

    print()
    print("=" * 70)
    print("RESUMEN")
    print("=" * 70)
    mu_para = float(np.mean(para_sims))
    mu_neg  = float(np.mean(neg_sims))
    print(f"  Similitud media — paráfrasis : {mu_para:.4f}")
    print(f"  Similitud media — negativos  : {mu_neg:.4f}")
    print(f"  Margen (μ_para - μ_neg)      : {mu_para - mu_neg:.4f}")
    print()

    if mu_para >= 0.75:
        print("  VEREDICTO: el modelo resuelve su task original correctamente.")
        print("  Si falla en Record Linkage del INER, el problema es de dominio.")
    else:
        print("  VEREDICTO: similitud baja en paráfrasis — revisar carga del modelo.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
