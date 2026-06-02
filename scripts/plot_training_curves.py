"""Genera gráficas de curvas de pérdida desde training_history.json.

Modos:
  - Individual: train_loss y val_loss por época para un run, con marcador en best_epoch.
  - Comparación: val_loss de varios runs en una sola figura.

Uso:
    python scripts/plot_training_curves.py --checkpoint beto_mnrl_hpc_run_e
    python scripts/plot_training_curves.py --checkpoint beto_mnrl_hpc_run_e beto_mnrl_hpc_run_f
    python scripts/plot_training_curves.py --all
    python scripts/plot_training_curves.py --all --compare

Las figuras se guardan en ~/Data/INER/outputs/figures/
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from record_linkage.config import FIGURES_DIR, MODELS_DIR


# === Naming convention oficial para resultados de tesis ===
# Mapea checkpoint_name interno → (título_display, filename_corto)
# Convención: Bi-Encoder · <Modelo> · <versión dataset> · <variante o run>

_MODEL_DISPLAY = {"beto": "BETO", "roberta": "RoBERTa-bio"}

_CHECKPOINT_PATTERNS = [
    # beto_mnrl_hpc_run_e → ("Bi-Encoder · BETO v1 · Run E", "be_beto_v1_runE")
    (re.compile(r"^beto_mnrl_hpc_run_([a-z])$"),
     lambda m: ("beto", "v1", f"Run {m.group(1).upper()}", f"be_beto_v1_run{m.group(1).upper()}")),
    # roberta_bio_hpc_run_a → ("Bi-Encoder · RoBERTa-bio v1 · Run A", "be_roberta_v1_runA")
    (re.compile(r"^roberta_bio_hpc_run_([a-z])$"),
     lambda m: ("roberta", "v1", f"Run {m.group(1).upper()}", f"be_roberta_v1_run{m.group(1).upper()}")),
    # beto_mnrl_hpc_v2_tok_skipnull → ("Bi-Encoder · BETO v2 · tok_skipnull", "be_beto_v2_tok_skipnull")
    (re.compile(r"^beto_mnrl_hpc_v2_(.+)$"),
     lambda m: ("beto", "v2", m.group(1), f"be_beto_v2_{m.group(1)}")),
    # roberta_bio_hpc_v2_... (cuando llegue)
    (re.compile(r"^roberta_bio_hpc_v2_(.+)$"),
     lambda m: ("roberta", "v2", m.group(1), f"be_roberta_v2_{m.group(1)}")),
]


def _pretty(checkpoint_name: str) -> tuple[str, str]:
    """Devuelve (display_title_prefix, filename_short) para un checkpoint.

    Si el nombre no matchea ningún patrón conocido, regresa el nombre crudo
    como fallback (no rompe runs futuros con naming distinto).
    """
    for pattern, mk in _CHECKPOINT_PATTERNS:
        m = pattern.match(checkpoint_name)
        if m:
            model_key, version, variant, fname = mk(m)
            model_display = _MODEL_DISPLAY.get(model_key, model_key.upper())
            display = f"Bi-Encoder · {model_display} · {version} · {variant}"
            return display, fname
    # Fallback
    return checkpoint_name, f"training_curves_{checkpoint_name}"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


def load_history(checkpoint_name: str) -> dict | None:
    hist_path = MODELS_DIR / "checkpoints" / checkpoint_name / "training_history.json"
    if not hist_path.exists():
        print(f"  AVISO: no se encontró {hist_path}")
        return None
    with open(hist_path) as f:
        return json.load(f)


def _run_label(checkpoint_name: str, data: dict) -> str:
    args = data.get("args", {})
    model = args.get("model", "?")
    lr = args.get("base_lr", "?")
    return f"{model} lr={lr}"


def plot_individual(checkpoint_name: str, data: dict, output_dir: Path) -> None:
    """Una figura por run: train y val en el mismo plot con ejes Y independientes
    (twin axes). Necesario porque train_loss colapsa a ~0 y val_loss se queda en ~1.1;
    una sola escala los aplastaría — twin axes los mantiene legibles.
    """
    history = data["history"]
    best_epoch = data.get("best_epoch")
    args = data.get("args", {})

    epochs     = [r["epoch"] for r in history]
    train_loss = [r["train_loss"] for r in history]
    val_loss   = [r["val_loss"] for r in history]

    display_prefix, filename = _pretty(checkpoint_name)
    out_path = output_dir / f"{filename}.png"
    if out_path.exists():
        print(f"  Omitido (ya existe): {out_path}")
        return

    best_val = next((r["val_loss"] for r in history if r["epoch"] == best_epoch), None)
    n_epochs_run = len(history)
    val_str = f"  ·  val={best_val:.4f}" if best_val is not None else ""
    title = (
        f"{display_prefix}{val_str}\n"
        f"lr={args.get('base_lr','?')}  ·  temp={args.get('temperature','?')}  ·  "
        f"batch={args.get('batch_size','?')}  ·  {n_epochs_run} épocas"
    )

    # Twin axes — val_loss en eje izquierdo (naranja), train_loss en eje derecho (azul)
    fig, ax_val = plt.subplots(figsize=(9, 5))
    fig.suptitle(title, fontsize=11)

    ax_val.plot(epochs, val_loss, color="#FF5722", linewidth=1.8, label="val_loss")
    ax_val.set_ylabel("Val loss", color="#FF5722")
    ax_val.tick_params(axis="y", labelcolor="#FF5722")
    ax_val.set_xlabel("Época")
    ax_val.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax_tr = ax_val.twinx()
    ax_tr.plot(epochs, train_loss, color="#2196F3", linewidth=1.8,
               linestyle="--", label="train_loss")
    ax_tr.set_ylabel("Train loss", color="#2196F3")
    ax_tr.tick_params(axis="y", labelcolor="#2196F3")

    if best_epoch:
        ax_val.axvline(best_epoch, color="red", linestyle=":", linewidth=1.2)
        if best_val:
            ax_val.scatter([best_epoch], [best_val], color="red", zorder=5, s=50,
                           label=f"best ep{best_epoch} ({best_val:.4f})")

    lines_val, labels_val = ax_val.get_legend_handles_labels()
    lines_tr, labels_tr = ax_tr.get_legend_handles_labels()
    ax_val.legend(lines_val + lines_tr, labels_val + labels_tr, fontsize=9, loc="upper right")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {out_path}")


def plot_comparison(checkpoints_data: list[tuple[str, dict]], output_dir: Path,
                    output_filename: str = "comparison.png") -> None:
    """Val loss de todos los runs en una sola figura."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Comparación val_loss · Bi-Encoder", fontsize=12)

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0",
              "#FF9800", "#00BCD4", "#E91E63", "#607D8B"]

    for idx, (name, data) in enumerate(checkpoints_data):
        history = data["history"]
        best_epoch = data.get("best_epoch")
        epochs   = [r["epoch"] for r in history]
        val_loss = [r["val_loss"] for r in history]
        # Label corto para legend: usa la parte que viene después de "Bi-Encoder · "
        display, _ = _pretty(name)
        label = display.replace("Bi-Encoder · ", "")
        color = colors[idx % len(colors)]

        ax.plot(epochs, val_loss, linewidth=1.8, label=label, color=color)

        if best_epoch:
            best_val = next((r["val_loss"] for r in history if r["epoch"] == best_epoch), None)
            if best_val:
                ax.scatter([best_epoch], [best_val], color=color, zorder=5, s=60,
                           marker="*")

    ax.set_xlabel("Época")
    ax.set_ylabel("Val loss")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    out_path = output_dir / output_filename
    if out_path.exists():
        print(f"  Omitido (ya existe): {out_path}")
        plt.close(fig)
        return
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {out_path}")


def list_available_checkpoints() -> list[str]:
    ckpt_root = MODELS_DIR / "checkpoints"
    if not ckpt_root.exists():
        return []
    return sorted(
        d.name for d in ckpt_root.iterdir()
        if d.is_dir() and (d / "training_history.json").exists()
    )


def main():
    parser = argparse.ArgumentParser(description="Grafica curvas de pérdida del entrenamiento")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--checkpoint", nargs="+", dest="checkpoints",
        metavar="RUN_NAME",
        help="Nombre(s) del run. Acepta uno o varios.",
    )
    group.add_argument(
        "--all", action="store_true",
        help="Grafica todos los runs con training_history.json",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Genera figura de comparación val_loss además de las individuales",
    )
    args = parser.parse_args()

    if args.all:
        checkpoints = list_available_checkpoints()
        if not checkpoints:
            print(f"ERROR: no se encontraron runs en {MODELS_DIR / 'checkpoints'}")
            return 1
        print(f"Runs encontrados: {checkpoints}")
        args.compare = True  # con --all siempre genera comparación
    else:
        checkpoints = args.checkpoints

    output_dir = FIGURES_DIR / "training_curves"
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = []
    for name in checkpoints:
        data = load_history(name)
        if data:
            loaded.append((name, data))

    if not loaded:
        print("ERROR: no se pudo cargar ningún training_history.json")
        return 1

    print(f"\nGenerando curvas individuales...")
    for name, data in loaded:
        plot_individual(name, data, output_dir)

    if args.compare and len(loaded) > 1:
        print(f"\nGenerando figura de comparación...")
        plot_comparison(loaded, output_dir)

    print(f"\n✓ Figuras en {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
