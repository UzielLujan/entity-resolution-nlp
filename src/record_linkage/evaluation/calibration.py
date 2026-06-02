"""Calibración de probabilidades e incertidumbre por vínculo (Vía A).

Funciones puras — sin I/O, sin modelos, sin estado. Operan sobre arrays de
logits/probabilidades/etiquetas. Pensadas para el Cross-Encoder (Etapa 2), cuya
salida P(match) es la decisión final por par candidato.

Flujo (temperature scaling post-hoc, sobre el CE YA entrenado — NO re-entrena):
  1. fit_temperature(logits_val, labels_val) -> T        (ajuste sobre validación)
  2. apply_temperature(logits, T) -> probs               (probabilidad calibrada)
  3. expected_calibration_error(probs, labels) -> ECE    (diagnóstico)
  4. reliability_curve(probs, labels) -> dict            (datos para el diagrama)
  5. Incertidumbre por vínculo:
       - binary_entropy(probs)          → H(p), usar sobre prob. CALIBRADA (máx. en p=0.5)
       - decision_margin(probs, tau)    → |p - tau_dec|, usar PRE-calibración

Notas de diseño:
  - Temperature scaling: p = sigmoid(logit / T). Un solo escalar T fijado minimizando
    la NLL binaria sobre validación. Es monótono → no cambia el ranking ni la decisión,
    solo recalibra el VALOR de la probabilidad. Ref: Guo et al. (2017),
    "On Calibration of Modern Neural Networks".
  - Separación (umbral de decisión) y calibración (valor de probabilidad) son ejes
    independientes: T no mueve el punto logit=0 (p=0.5).
"""

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoide numéricamente estable."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Aplica temperature scaling: p = sigmoid(logit / T).

    Args:
        logits:      (n,) logits crudos del CE (pre-sigmoide).
        temperature: escalar T > 0. T>1 suaviza (corrige sobreconfianza); T<1 agudiza.

    Returns:
        (n,) probabilidades calibradas en (0, 1).
    """
    if temperature <= 0:
        raise ValueError(f"temperature debe ser > 0, recibido {temperature}")
    return _sigmoid(np.asarray(logits, dtype=float) / temperature)


def fit_temperature(
    logits_val: np.ndarray,
    labels_val: np.ndarray,
    bounds: tuple = (0.05, 10.0),
) -> float:
    """Ajusta la temperatura T minimizando la NLL binaria sobre validación.

    Método post-hoc sobre el modelo congelado — NO re-entrena el CE. Solo requiere
    los logits del CE ya entrenado evaluado sobre el split de validación.

    Args:
        logits_val: (n,) logits crudos (pre-sigmoide) sobre validación.
        labels_val: (n,) etiquetas en {0, 1}.
        bounds:     rango de búsqueda para T.

    Returns:
        T óptimo (float).
    """
    from scipy.optimize import minimize_scalar

    logits = np.asarray(logits_val, dtype=float)
    labels = np.asarray(labels_val, dtype=float)
    eps = 1e-12

    def _nll(temperature: float) -> float:
        p = np.clip(_sigmoid(logits / temperature), eps, 1.0 - eps)
        return float(-np.mean(labels * np.log(p) + (1.0 - labels) * np.log(1.0 - p)))

    result = minimize_scalar(_nll, bounds=bounds, method="bounded")
    return float(result.x)


def _binary_confidence_accuracy(probs: np.ndarray, labels: np.ndarray):
    """Confianza max(p, 1-p) y acierto de la predicción (p>=0.5) por muestra."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    preds = (probs >= 0.5).astype(int)
    confidences = np.maximum(probs, 1.0 - probs)
    correct = (preds == labels).astype(float)
    return confidences, correct


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """ECE — brecha promedio (ponderada por bin) entre confianza y accuracy.

    Para clasificación binaria, la confianza de cada predicción es max(p, 1-p) y el
    acierto es si la clase predicha (p>=0.5) coincide con la etiqueta (Guo et al., 2017).

    Returns:
        ECE en [0, 1]. 0 = perfectamente calibrado.
    """
    confidences, correct = _binary_confidence_accuracy(probs, labels)
    n = len(confidences)
    if n == 0:
        return 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if i == 0:
            mask |= confidences <= lo  # incluye el borde inferior
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        acc_bin = correct[mask].mean()
        conf_bin = confidences[mask].mean()
        ece += (cnt / n) * abs(acc_bin - conf_bin)
    return float(ece)


def reliability_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Datos para el reliability diagram: confianza media vs accuracy por bin.

    Returns:
        dict con listas (longitud <= n_bins, omite bins vacíos):
        'bin_confidence', 'bin_accuracy', 'bin_count'.
        Un modelo bien calibrado tiene bin_confidence ≈ bin_accuracy (diagonal).
    """
    confidences, correct = _binary_confidence_accuracy(probs, labels)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    conf_list, acc_list, cnt_list = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if i == 0:
            mask |= confidences <= lo
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        conf_list.append(float(confidences[mask].mean()))
        acc_list.append(float(correct[mask].mean()))
        cnt_list.append(cnt)
    return {"bin_confidence": conf_list, "bin_accuracy": acc_list, "bin_count": cnt_list}


def binary_entropy(probs: np.ndarray, base: str = "bits") -> np.ndarray:
    """Incertidumbre por vínculo: entropía de Bernoulli H(p).

    H(p) = -p log p - (1-p) log(1-p). Máxima en p=0.5 (1 bit en base 2), cero en
    p=0 y p=1. Usar sobre probabilidad CALIBRADA: solo tras calibrar, p=0.5 es el
    punto 50/50 real y la entropía mide la incertidumbre genuina del modelo.

    Args:
        probs: (n,) probabilidades calibradas.
        base:  "bits" (log2, H en [0,1]) o "nats" (log natural).

    Returns:
        (n,) incertidumbre por par.
    """
    p = np.clip(np.asarray(probs, dtype=float), 1e-12, 1.0 - 1e-12)
    log = np.log2 if base == "bits" else np.log
    return -(p * log(p) + (1.0 - p) * log(1.0 - p))


def decision_margin(probs: np.ndarray, tau_dec: float) -> np.ndarray:
    """Incertidumbre PRE-calibración: distancia al umbral de decisión.

    Devuelve |p - tau_dec|. Margen pequeño → cerca de la frontera → ALTA incertidumbre.
    Útil cuando la probabilidad aún no está calibrada y p=0.5 no es el 50/50 real;
    respeta el umbral empírico (p. ej. tau_dec=0.12 del CE).

    Args:
        probs:   (n,) scores del modelo (sin calibrar).
        tau_dec: umbral de decisión match/no-match (NO la T de calibración ni la τ de MNRL).

    Returns:
        (n,) margen al umbral. Para una incertidumbre que crece cerca de la frontera,
        usar una transformación decreciente (p. ej. -|p - tau_dec| o 1 - |p - tau_dec|/max).
    """
    return np.abs(np.asarray(probs, dtype=float) - tau_dec)
