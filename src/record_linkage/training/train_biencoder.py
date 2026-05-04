"""
Etapa 1 — Bi-Encoder con Multiple Negatives Ranking Loss (MNRL).

Uso (smoke test local, 1 época):
    python scripts/run_train_biencoder.py \
        --model BETO \
        --parquet ~/Data/INER/processed/tesis1/dataset_split.parquet \
        --output ~/Data/INER/models/checkpoints/beto_mnrl_run03 \
        --epochs 1 --batch-size 8 --n-aug 0 --max-seq-length 384

Uso en HPC (ajustar batch-size y epochs según recursos):
    python scripts/run_train_biencoder.py --model BETO --epochs 10 --batch-size 64 --n-aug 0
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

import warnings
warnings.filterwarnings("ignore", message="Detected call of.*lr_scheduler")

from record_linkage.config import MODELS_DIR, PROCESSED_DIR, TRAINING_DIR
from record_linkage.data.augmentation import AugmentationConfig, augment
from record_linkage.models.biencoder import build_biencoder
from record_linkage.utils.mnrl import dump_mnrl_batch

# Palabras ancla para warm initialization
_WARM_ANCHORS = {
    "[BLK_ID]":    ["identidad", "identificación", "paciente", "nombre", "persona"],
    "[BLK_CLIN]":  ["clínico", "diagnóstico", "médico", "patología", "enfermedad"],
    "[BLK_GEO]":   ["geográfico", "ubicación", "dirección", "ciudad", "municipio"],
    "[BLK_SOCIO]": ["socioeconómico", "social", "económico", "ocupación", "ingreso"],
    "[BLK_ADMIN]": ["administrativo", "expediente", "folio", "registro"],
    "[COL]":       ["campo", "columna", "atributo", "variable"],
    "[VAL]":       ["valor", "dato", "resultado", "información"],
}


# =============================================================================
# Warm initialization
# =============================================================================

def warm_init_special_tokens(st_model) -> None:
    """Inicializa embeddings de tokens especiales como centroide de palabras ancla.

    Evita que resize_token_embeddings() deje los tokens nuevos en ruido aleatorio,
    reduciendo gradientes iniciales inestables que pueden dañar capas adyacentes.
    """
    tokenizer = st_model.tokenizer
    emb_weight = st_model._first_module().auto_model.get_input_embeddings().weight

    with torch.no_grad():
        for token, anchors in _WARM_ANCHORS.items():
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id == tokenizer.unk_token_id:
                continue
            vecs = []
            for word in anchors:
                ids = tokenizer(word, add_special_tokens=False)["input_ids"]
                vecs.extend(emb_weight[i].clone() for i in ids)
            if vecs:
                emb_weight[token_id] = torch.stack(vecs).mean(0)

    print(f"  Warm init: {len(_WARM_ANCHORS)} tokens especiales inicializados")


# =============================================================================
# Dataset
# =============================================================================

class BiEncoderDataset(Dataset):
    """Pares (anchor, positive) para MNRL.

    - Pares naturales:  registros cross-db del mismo entity_id (sin augmentación)
    - Pares sintéticos: (registro, augment(registro)) — augmentación on-the-fly
    """

    def __init__(self, df: pd.DataFrame, n_augmentations: int, aug_config: AugmentationConfig):
        self.aug_config = aug_config
        self.natural_pairs = []
        self.synthetic_anchors = []

        for _, group in df.groupby("entity_id"):
            by_source = group.groupby("source_db")["text"].apply(list).to_dict()
            sources = list(by_source.keys())
            for i in range(len(sources)):
                for j in range(i + 1, len(sources)):
                    for ta in by_source[sources[i]]:
                        for tb in by_source[sources[j]]:
                            self.natural_pairs.append((ta, tb))

        self.synthetic_anchors = df["text"].tolist() * n_augmentations
        random.shuffle(self.synthetic_anchors)

        self._n_natural = len(self.natural_pairs)

    def __len__(self):
        return self._n_natural + len(self.synthetic_anchors)

    def __getitem__(self, idx):
        if idx < self._n_natural:
            return self.natural_pairs[idx]
        anchor = self.synthetic_anchors[idx - self._n_natural]
        return anchor, augment(anchor, self.aug_config)


# =============================================================================
# Optimizer con LR diferencial por capa
# =============================================================================

def build_optimizer(st_model, base_lr: float, decay: float) -> AdamW:
    """AdamW con LR decreciente hacia las capas inferiores del Transformer.

    Capa de salida (n-1) → base_lr. Embeddings → base_lr * decay^n_layers.
    Preserva conocimiento morfológico de español en capas bajas de BETO.
    """
    no_decay = {"bias", "LayerNorm.weight"}
    named_params = list(st_model._first_module().auto_model.named_parameters())
    n_layers = st_model._first_module().auto_model.config.num_hidden_layers
    groups = []

    def _add(params, lr):
        wd  = [p for n, p in params if not any(nd in n for nd in no_decay)]
        nwd = [p for n, p in params if     any(nd in n for nd in no_decay)]
        if wd:  groups.append({"params": wd,  "lr": lr, "weight_decay": 0.01})
        if nwd: groups.append({"params": nwd, "lr": lr, "weight_decay": 0.0})

    _add([(n, p) for n, p in named_params if "embeddings" in n],
         base_lr * (decay ** n_layers))

    for idx in range(n_layers):
        depth = n_layers - 1 - idx
        _add([(n, p) for n, p in named_params if f"encoder.layer.{idx}." in n],
             base_lr * (decay ** depth))

    pooler = [(n, p) for n, p in named_params if "pooler" in n]
    if pooler:
        _add(pooler, base_lr)

    return AdamW(groups)


# =============================================================================
# MNRL forward pass y loss
# =============================================================================

def _encode_batch(st_model, texts: list, device: torch.device) -> torch.Tensor:
    """Codifica un batch con gradiente activo. Retorna embeddings L2-normalizados."""
    enc = st_model.tokenizer(
        texts, padding=True, truncation=True,
        max_length=st_model.max_seq_length, return_tensors="pt",
    ).to(device)
    out = st_model._first_module().auto_model(**enc)
    mask = enc["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
    pooled = torch.sum(out.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
    return F.normalize(pooled, p=2, dim=1)


def _mnrl_loss(emb_a: torch.Tensor, emb_b: torch.Tensor, temperature: float) -> torch.Tensor:
    sim = (emb_a @ emb_b.T) / temperature
    labels = torch.arange(len(emb_a), device=emb_a.device)
    return F.cross_entropy(sim, labels)


# =============================================================================
# Epoch de entrenamiento y evaluación
# =============================================================================

def train_epoch(st_model, loader, optimizer, scheduler, scaler, device, temperature, epoch,
                log_every=50, viz=False, viz_dir=None, steps_per_epoch=None):
    st_model._first_module().auto_model.train()
    total_loss, n_steps = 0.0, 0
    t0 = time.time()
    use_amp = device.type == "cuda"

    viz_steps = set()
    if viz and epoch == 1 and viz_dir is not None and steps_per_epoch:
        viz_steps = {0, steps_per_epoch // 2, steps_per_epoch - 1}

    for step, (texts_a, texts_b) in enumerate(loader):
        with torch.autocast("cuda", enabled=use_amp):
            emb_a = _encode_batch(st_model, list(texts_a), device)
            emb_b = _encode_batch(st_model, list(texts_b), device)
            loss = _mnrl_loss(emb_a, emb_b, temperature)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            st_model._first_module().auto_model.parameters(), max_norm=1.0
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step in viz_steps:
            dump_mnrl_batch(list(texts_a), list(texts_b), emb_a, emb_b, step, epoch, viz_dir)

        total_loss += loss.item()
        n_steps += 1

        if (step + 1) % log_every == 0:
            print(f"  época {epoch} | step {step+1:>5} | "
                  f"loss {total_loss/n_steps:.4f} | {time.time()-t0:.0f}s")

    return total_loss / n_steps


def eval_loss(st_model, df_val: pd.DataFrame, batch_size: int, temperature: float,
              device: torch.device) -> float:
    """Loss MNRL sobre pares naturales de val (sin augmentación)."""
    st_model._first_module().auto_model.eval()

    pairs = []
    for _, group in df_val.groupby("entity_id"):
        by_source = group.groupby("source_db")["text"].apply(list).to_dict()
        sources = list(by_source.keys())
        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                for ta in by_source[sources[i]]:
                    for tb in by_source[sources[j]]:
                        pairs.append((ta, tb))

    if not pairs:
        return float("nan")

    total, n = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            ta_batch, tb_batch = zip(*pairs[start:start + batch_size])
            emb_a = _encode_batch(st_model, list(ta_batch), device)
            emb_b = _encode_batch(st_model, list(tb_batch), device)
            total += _mnrl_loss(emb_a, emb_b, temperature).item()
            n += 1

    return total / n if n else float("nan")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Entrena Bi-Encoder con MNRL")
    parser.add_argument("--model",          default="BETO")
    parser.add_argument("--parquet",        default=None)
    parser.add_argument("--output",         default=None)
    parser.add_argument("--epochs",         type=int,   default=2)
    parser.add_argument("--batch-size",     type=int,   default=8)
    parser.add_argument("--n-aug",          type=int,   default=0,
                        help="Pares sintéticos por registro (default: 0)")
    parser.add_argument("--base-lr",        type=float, default=2e-5)
    parser.add_argument("--decay-factor",   type=float, default=0.95)
    parser.add_argument("--warmup-ratio",   type=float, default=0.06)
    parser.add_argument("--temperature",    type=float, default=0.05)
    parser.add_argument("--max-seq-length", type=int,   default=384,
                        help="Longitud máxima de secuencia (default: 384 para local, 512 en HPC)")
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--viz",            action="store_true",
                        help="Guarda textos y matrices MNRL de 3 batches en outputs/training/<run>/viz/")
    parser.add_argument("--patience",      type=int, default=3,
                        help="Épocas sin mejora en val_loss antes de detener (0 = desactivado)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDispositivo: {device}")

    model_path   = MODELS_DIR / "pretrained" / args.model
    parquet_path = Path(args.parquet) if args.parquet else PROCESSED_DIR / "tesis1" / "dataset_split.parquet"
    output_dir   = Path(args.output)  if args.output  else MODELS_DIR / "checkpoints" / f"{args.model}_mnrl"
    output_dir.mkdir(parents=True, exist_ok=True)

    viz_dir = TRAINING_DIR / output_dir.name / "viz" if args.viz else None

    print(f"\nCargando modelo: {model_path}")
    st_model = build_biencoder(model_path)
    st_model.max_seq_length = args.max_seq_length
    st_model = st_model.to(device)

    print("\nAplicando warm initialization...")
    warm_init_special_tokens(st_model)

    print(f"\nCargando dataset: {parquet_path}")
    df_all   = pd.read_parquet(parquet_path)
    df_train = df_all[df_all["split"] == "train"].reset_index(drop=True)
    df_val   = df_all[df_all["split"] == "val"].reset_index(drop=True)
    print(f"  Train: {len(df_train):,} registros | Val: {len(df_val):,} registros")

    aug_config    = AugmentationConfig()
    train_dataset = BiEncoderDataset(df_train, n_augmentations=args.n_aug, aug_config=aug_config)
    train_loader  = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=0, drop_last=True,
    )
    print(f"  Pares totales: {len(train_dataset):,} "
          f"({train_dataset._n_natural:,} naturales + "
          f"{len(train_dataset.synthetic_anchors):,} sintéticos)")

    optimizer    = build_optimizer(st_model, base_lr=args.base_lr, decay=args.decay_factor)
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    def _lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = LambdaLR(optimizer, _lr_lambda)

    print(f"\n  Pasos totales: {total_steps:,} | Warmup: {warmup_steps:,} | "
          f"LR base: {args.base_lr} | Temperatura: {args.temperature}")

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0
    print(f"\n{'='*60}")
    print(f"  Entrenamiento MNRL — {args.epochs} época(s) — {args.model}")
    print(f"{'='*60}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train_epoch(
            st_model, train_loader, optimizer, scheduler, scaler,
            device, args.temperature, epoch,
            viz=args.viz, viz_dir=viz_dir, steps_per_epoch=len(train_loader),
        )
        vl_loss = eval_loss(st_model, df_val, args.batch_size, args.temperature, device)
        elapsed = time.time() - t0

        print(f"\n  Época {epoch}/{args.epochs} — "
              f"train_loss={tr_loss:.4f}  val_loss={vl_loss:.4f}  {elapsed:.0f}s")

        ckpt = output_dir / f"epoch_{epoch:02d}"
        st_model.save(str(ckpt))
        print(f"  Checkpoint guardado: {ckpt}")

        if not np.isnan(vl_loss) and vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_epoch = epoch
            patience_counter = 0
            best_ckpt = output_dir / "best"
            st_model.save(str(best_ckpt))
            print(f"  Mejor val_loss={best_val_loss:.4f} → best/ actualizado")
        elif args.patience > 0:
            patience_counter += 1
            print(f"  Sin mejora en val_loss ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print(f"\n  Early stopping — {args.patience} épocas sin mejora.")
                history.append({
                    "epoch":      epoch,
                    "train_loss": round(tr_loss, 6),
                    "val_loss":   round(vl_loss, 6),
                    "elapsed_s":  round(elapsed, 1),
                })
                break

        history.append({
            "epoch":      epoch,
            "train_loss": round(tr_loss, 6),
            "val_loss":   round(vl_loss, 6),
            "elapsed_s":  round(elapsed, 1),
        })

    hist_path = output_dir / "training_history.json"
    with open(hist_path, "w") as f:
        json.dump({"args": vars(args), "history": history, "best_epoch": best_epoch},
                  f, indent=2, ensure_ascii=False)

    print(f"\n✓ Mejor época: {best_epoch} (val_loss={best_val_loss:.4f})")
    print(f"✓ Historial: {hist_path}")
    print(f"✓ Checkpoints: {output_dir}\n")
    return 0


if __name__ == "__main__":
    main()
