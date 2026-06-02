"""Entrenamiento del Cross-Encoder (Etapa 2) con BCEWithLogitsLoss.

Consume un parquet de pares etiquetados (`text_a`, `text_b`, `label` ∈ {0, 1}) y
entrena un BERT/RoBERTa con head de regresión sobre la tarea binaria match/no_match.

Los pares de training típicamente vienen de hard negative mining (corriendo el BE
fine-tuneado sobre el dataset completo), complementados con positivos conocidos.

Uso:
    python scripts/train_crossencoder.py --model BETO --pairs-train ...parquet
        --pairs-val ...parquet --output ...
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

from record_linkage.config import MODELS_DIR
from record_linkage.models.crossencoder import build_crossencoder
from record_linkage.training.bce import make_bce_loss


# =============================================================================
# Dataset
# =============================================================================

class PairDataset(Dataset):
    """Dataset de pares (record_id_a, record_id_b, label) con lookup de texto.

    El parquet de pares contiene solo índices del grafo (record_id_a, record_id_b, label).
    El texto se obtiene de `records_df` (dataset.parquet, con cols `record_id` y `text`)
    via lookup en memoria — evita duplicación de texto en los pair files.
    """

    def __init__(self, df_pairs: pd.DataFrame, records_df: pd.DataFrame,
                 tokenizer, max_length: int = 512):
        required = {"record_id_a", "record_id_b", "label"}
        missing = required - set(df_pairs.columns)
        if missing:
            raise ValueError(f"Parquet de pares no tiene columnas requeridas: {missing}")
        if "text" not in records_df.columns or "record_id" not in records_df.columns:
            raise ValueError("records_df necesita columnas 'record_id' y 'text'")

        self.pairs = df_pairs.reset_index(drop=True)
        self.id_to_text = dict(zip(records_df["record_id"].astype(int),
                                   records_df["text"].astype(str)))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]
        text_a = self.id_to_text[int(row["record_id_a"])]
        text_b = self.id_to_text[int(row["record_id_b"])]
        encoded = self.tokenizer(
            text_a, text_b,
            padding="max_length", truncation=True, max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "token_type_ids": encoded.get("token_type_ids", torch.zeros_like(encoded["input_ids"])).squeeze(0),
            "label":          torch.tensor(float(row["label"]), dtype=torch.float),
        }


# =============================================================================
# Training loop
# =============================================================================

def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, log_every: int = 50):
    model.train()
    total_loss = 0.0
    n_seen = 0
    t0 = time.time()

    for step, batch in enumerate(loader, start=1):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels         = batch["label"].to(device)

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).logits.squeeze(-1)

        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * len(labels)
        n_seen     += len(labels)

        if step % log_every == 0:
            print(f"  step {step:5d} | loss {total_loss/n_seen:.4f} | {time.time()-t0:.0f}s")

    return total_loss / max(n_seen, 1)


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_seen     = 0
    all_logits = []
    all_labels = []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels         = batch["label"].to(device)

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).logits.squeeze(-1)

        loss = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        n_seen     += len(labels)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    val_loss = total_loss / max(n_seen, 1)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    return val_loss, logits, labels


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Entrena el Cross-Encoder con BCE")
    parser.add_argument("--model",          default="BETO",
                        help="Nombre del modelo en models/pretrained/")
    parser.add_argument("--dataset",        required=True,
                        help="Parquet con los registros (cols: record_id, text). "
                             "Típicamente dataset.parquet o dataset_split.parquet del perfil.")
    parser.add_argument("--pairs-train",    required=True,
                        help="Parquet con pares de training (cols: record_id_a, record_id_b, label)")
    parser.add_argument("--pairs-val",      default=None,
                        help="Parquet con pares de validación (mismo esquema)")
    parser.add_argument("--output",         default=None,
                        help="Directorio de checkpoints (default: checkpoints/<model>_ce)")
    parser.add_argument("--epochs",         type=int,   default=3)
    parser.add_argument("--batch-size",     type=int,   default=16)
    parser.add_argument("--lr",             type=float, default=2e-5)
    parser.add_argument("--warmup-ratio",   type=float, default=0.1)
    parser.add_argument("--max-seq-length", type=int,   default=512)
    parser.add_argument("--pos-weight",     type=float, default=1.0,
                        help="Peso BCE para clase positiva (sube si positivos son raros)")
    parser.add_argument("--patience",       type=int,   default=2,
                        help="Épocas sin mejora en val_loss antes de detener (0 = desactivado)")
    parser.add_argument("--only-best",      action="store_true",
                        help="Solo guarda best/ — omite epoch_NN/ por época. Ahorra disco.")
    parser.add_argument("--seed",           type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDispositivo: {device}")

    model_path = MODELS_DIR / "pretrained" / args.model
    output_dir = Path(args.output) if args.output else MODELS_DIR / "checkpoints" / f"{args.model}_ce"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCargando modelo: {model_path}")
    model, tokenizer = build_crossencoder(model_path, num_labels=1)
    model = model.to(device)

    print(f"\nCargando registros: {args.dataset}")
    records_df = pd.read_parquet(args.dataset)
    if "text" not in records_df.columns or "record_id" not in records_df.columns:
        raise ValueError(f"--dataset debe tener cols record_id y text: {args.dataset}")
    print(f"  {len(records_df):,} registros | {records_df['text'].str.len().mean():.0f} chars/text (media)")

    print(f"\nCargando pares de entrenamiento: {args.pairs_train}")
    train_df = pd.read_parquet(args.pairs_train)
    print(f"  {len(train_df):,} pares | {(train_df['label']==1).sum():,} positivos | "
          f"{(train_df['label']==0).sum():,} negativos")

    val_df = None
    if args.pairs_val:
        val_df = pd.read_parquet(args.pairs_val)
        print(f"Cargando pares de validación: {args.pairs_val}")
        print(f"  {len(val_df):,} pares | {(val_df['label']==1).sum():,} positivos | "
              f"{(val_df['label']==0).sum():,} negativos")

    train_ds = PairDataset(train_df, records_df, tokenizer, max_length=args.max_seq_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    val_loader = None
    if val_df is not None:
        val_ds = PairDataset(val_df, records_df, tokenizer, max_length=args.max_seq_length)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion = make_bce_loss(args.pos_weight).to(device)

    print(f"\n{'='*60}\nTraining CE — {args.epochs} épocas — lr={args.lr} — batch={args.batch_size}\n{'='*60}")

    history = []
    best_val = float("inf")
    best_epoch: Optional[int] = None
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Época {epoch}/{args.epochs} ---")
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        epoch_time = time.time() - t0

        val_loss = None
        if val_loader is not None:
            val_loss, _, _ = evaluate_epoch(model, val_loader, criterion, device)
            print(f"  Época {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  ({epoch_time:.0f}s)")
        else:
            print(f"  Época {epoch}: train_loss={train_loss:.4f}  ({epoch_time:.0f}s)")

        # Checkpoint por época (omitir si --only-best)
        if not args.only_best:
            ckpt_dir = output_dir / f"epoch_{epoch:02d}"
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"  Checkpoint: {ckpt_dir}")

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        # Best + early stopping
        if val_loss is not None:
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                no_improve = 0
                best_dir = output_dir / "best"
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                print(f"  ✓ Mejor val_loss={val_loss:.4f} → best/ actualizado")
            else:
                no_improve += 1
                if args.patience > 0 and no_improve >= args.patience:
                    print(f"  Early stopping — {args.patience} épocas sin mejora.")
                    break

    # Guardar history
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump({
            "args":       vars(args),
            "history":    history,
            "best_epoch": best_epoch,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Training history: {history_path}")
    if best_epoch:
        print(f"✓ Mejor epoch: {best_epoch} (val_loss={best_val:.4f})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
