"""Visualización del espacio métrico aprendido por el Bi-Encoder.

Codifica todos los registros de un split, reduce de 768D → 3D (o 2D) con UMAP
(métrica coseno, alineado al espacio del BE), y produce un scatter interactivo
de Plotly como HTML.

El resultado es la prueba visual directa del aprendizaje métrico: si Δ > 2,
las "islas" de cada paciente (sus 2-3 registros en distintas bases) deben
aparecer compactas y separadas entre sí.

Diseño alineado a `docs/Anexos/demo-propuesta.md` § Paso 3:
- UMAP con métrica cosine (no euclidiana — el espacio del BE es coseno)
- Colores fijos por source_db: rojo Comorbilidad, verde Económico, azul TS
- Singletons opcionales con opacity diferenciada

Uso:
    # 3D por default sobre split=test del ganador del 2 x 2
    python scripts/visualize_embeddings.py \\
        --checkpoint beto_mnrl_hpc_v2_tok_skipnull \\
        --dataset ~/Data/INER/processed/tesis/output/tok_skipnull/dataset_split.parquet

    # 2D
    python scripts/visualize_embeddings.py --checkpoint <run> --dataset <parquet> --dims 2

    # Solo entidades vinculables (sin singletons)
    python scripts/visualize_embeddings.py --checkpoint <run> --dataset <parquet> --no-singletons

    # Fondo oscuro estilo presentación
    python scripts/visualize_embeddings.py --checkpoint <run> --dataset <parquet> --dark-bg
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import umap
import plotly.express as px

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from record_linkage.config import FIGURES_DIR, perfil_paths
from record_linkage.evaluation.biencoder_eval import (
    load_dataset_split,
    resolve_checkpoint_path,
)
from record_linkage.models.biencoder import build_biencoder, encode_texts
from record_linkage.utils.pairs import _COL_MAP


# Paleta del doc demo-propuesta.md
SOURCE_COLORS = {
    "Comorbilidad":   "#D85A30",
    "Económico":      "#1D9E75",
    "Trabajo Social": "#378ADD",
}


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convierte #RRGGBB a rgba(r,g,b,a) para opacity per-point en Plotly 3D."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _fmt_value(v) -> str:
    """Formatea valor para display: '?' si NaN, int si numérico, str otherwise."""
    if v is None or pd.isna(v):
        return "?"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip()


def build_identity_map(clean_dir: Path) -> dict:
    """Construye un dict {record_id: 'NOMBRE | exp=NNN'} leyendo las 3 CSVs limpias.

    Los record_ids se asignan secuencialmente en orden [econo, comorbilidad, trabajo_social]
    durante _step_classify de dataset_v2 — esa convención es la que aquí replicamos.

    Args:
        clean_dir: directorio con `econo_clean.csv`, `comorbilidad_clean.csv`,
                   `trabajo_social_clean.csv`.

    Returns:
        dict record_id → "NOMBRE | exp=NNN" para hover de Plotly.
    """
    csv_order = [
        ("econo",          "econo_clean.csv"),
        ("comorbilidad",   "comorbilidad_clean.csv"),
        ("trabajo_social", "trabajo_social_clean.csv"),
    ]

    identity_map = {}
    record_id = 0
    for csv_type, fname in csv_order:
        path = clean_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"CSV limpio no encontrado: {path}")
        df = pd.read_csv(path)
        exp_col, nombre_col = _COL_MAP[csv_type]
        for _, row in df.iterrows():
            nombre = _fmt_value(row.get(nombre_col))
            exp    = _fmt_value(row.get(exp_col))
            identity_map[record_id] = f"{nombre} | exp={exp}"
            record_id += 1

    return identity_map


def main():
    parser = argparse.ArgumentParser(description="Visualiza el espacio métrico del Bi-Encoder con UMAP+Plotly")
    parser.add_argument("--checkpoint", required=True,
                        help="Nombre del run en checkpoints/ (usa best/ por defecto)")
    parser.add_argument("--dataset", required=True,
                        help="Ruta al dataset.parquet o dataset_split.parquet")
    parser.add_argument("--split", default="test",
                        choices=["train", "val", "test", "all"],
                        help="Split a visualizar (default: test). 'all' = sin filtrar por split")
    parser.add_argument("--dims", type=int, default=3, choices=[2, 3],
                        help="Dimensiones del scatter (default: 3)")
    parser.add_argument("--no-singletons", action="store_true",
                        help="Omitir singletons; solo muestra entidades vinculables")
    parser.add_argument("--n-neighbors", type=int, default=15,
                        help="UMAP n_neighbors — balance local/global (default: 15, recomendado en demo-propuesta)")
    parser.add_argument("--min-dist", type=float, default=0.1,
                        help="UMAP min_dist — compacidad de clusters (default: 0.1)")
    parser.add_argument("--epoch", type=int, default=None,
                        help="Epoch específico en lugar de best/")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size para encoding (default: 64)")
    parser.add_argument("--max-seq-length", type=int, default=384,
                        help="Max seq length para encoding (default: 384)")
    parser.add_argument("--dark-bg", action="store_true",
                        help="Fondo oscuro estilo presentación (#0f0f13). Default: blanco.")
    parser.add_argument("--marker-size", type=int, default=5,
                        help="Tamaño de los puntos (default: 5 para 3D, sube si hay pocos puntos)")
    parser.add_argument("--singleton-opacity", type=float, default=0.25,
                        help="Opacidad de los singletons (0.0–1.0, default: 0.25). Sube para verlos más")
    parser.add_argument("--output-html", default=None,
                        help="Ruta del HTML (default: outputs/figures/embeddings_<checkpoint>_<split>_<dims>D.html)")
    parser.add_argument("--clean-dir", default=None,
                        help="Directorio con los CSVs limpios para lookup de nombre/expediente. "
                             "Default: derivado del perfil 'tesis' (tesis/clean/)")
    highlight_group = parser.add_mutually_exclusive_group()
    highlight_group.add_argument("--highlight-entity", type=int, default=None,
                                  help="entity_id específico a resaltar — añade annotations permanentes "
                                       "y marker grande sobre sus 2-3 registros del cluster")
    highlight_group.add_argument("--highlight-random-cluster", action="store_true",
                                  help="Selecciona aleatoriamente (con --seed) un cluster con 3 registros "
                                       "(uno por source_db) para destacar")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # === Carga modelo
    ckpt_path = resolve_checkpoint_path(args.checkpoint, args.epoch)
    print(f"\nCargando BE: {ckpt_path}")
    t0 = time.time()
    model = build_biencoder(ckpt_path)
    model.max_seq_length = args.max_seq_length
    print(f"  BE cargado en {time.time()-t0:.1f}s | max_seq={args.max_seq_length}")
    print(f"  Dispositivo: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # === Carga dataset
    dataset_path = Path(args.dataset)
    split = None if args.split == "all" else args.split
    df = load_dataset_split(dataset_path, split=split)

    # === Identificar vinculables vs singletons
    entity_sources = df.groupby("entity_id")["source_db"].nunique()
    linkable_ids = set(entity_sources[entity_sources > 1].index)
    df = df.copy()
    df["is_linkable"] = df["entity_id"].isin(linkable_ids)

    if args.no_singletons:
        df = df[df["is_linkable"]].reset_index(drop=True)
        print(f"  Solo vinculables: {len(df):,} registros, {len(linkable_ids):,} entidades")
    else:
        n_link = int(df["is_linkable"].sum())
        n_sing = len(df) - n_link
        print(f"  Total: {len(df):,} | vinculables: {n_link:,} | singletons: {n_sing:,}")

    # === Encoding
    print(f"\nEncodeando {len(df):,} registros (batch={args.batch_size})...")
    t0 = time.time()
    embeddings = encode_texts(model, df["text"].tolist(), batch_size=args.batch_size)
    print(f"  Embeddings: {time.time()-t0:.1f}s | shape={embeddings.shape}")

    # === UMAP
    print(f"\nReduciendo {embeddings.shape[1]}D → {args.dims}D con UMAP "
          f"(n_neighbors={args.n_neighbors}, min_dist={args.min_dist}, metric=cosine)...")
    t0 = time.time()
    reducer = umap.UMAP(
        n_components=args.dims,
        metric="cosine",
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.seed,
    )
    emb_low = reducer.fit_transform(embeddings)
    print(f"  UMAP: {time.time()-t0:.1f}s")

    # === DataFrame para Plotly
    viz = pd.DataFrame({
        "u1": emb_low[:, 0],
        "u2": emb_low[:, 1],
    })
    if args.dims == 3:
        viz["u3"] = emb_low[:, 2]
    viz["source_db"]   = df["source_db"].values
    viz["entity_id"]   = df["entity_id"].values
    viz["record_id"]   = df["record_id"].values
    viz["is_linkable"] = df["is_linkable"].values
    # Identidad compacta para hover: "NOMBRE | exp=NNN" — lookup desde CSVs limpios
    clean_dir = Path(args.clean_dir) if args.clean_dir else perfil_paths("tesis")["clean"]
    print(f"  Lookup de identidad desde: {clean_dir}")
    identity_map = build_identity_map(clean_dir)
    viz["identidad"]   = df["record_id"].map(identity_map).values

    # === Selección de cluster a destacar (annotations permanentes)
    highlight_entity = None
    if args.highlight_entity is not None:
        highlight_entity = args.highlight_entity
        if highlight_entity not in set(viz["entity_id"]):
            raise ValueError(f"entity_id={highlight_entity} no está en el split '{args.split}'")
    elif args.highlight_random_cluster:
        # Buscar entities con 3 registros, uno por source_db (cluster "perfecto")
        candidates = (
            viz.groupby("entity_id")
            .agg(n_records=("source_db", "size"), n_sources=("source_db", "nunique"))
            .query("n_records == 3 and n_sources == 3")
        )
        if len(candidates) == 0:
            print("  ⚠ No hay entities con exactamente 3 registros (1 por source_db) en este split — sin highlight")
        else:
            rng = np.random.default_rng(args.seed)
            highlight_entity = int(rng.choice(candidates.index))
            print(f"  Cluster aleatorio seleccionado: entity_id={highlight_entity}")

    highlight_df = None
    if highlight_entity is not None:
        highlight_df = viz[viz["entity_id"] == highlight_entity].copy()
        print(f"  Destacando {len(highlight_df)} puntos del entity_id={highlight_entity}:")
        for _, r in highlight_df.iterrows():
            print(f"    {r['source_db']:<16}: {r['identidad']}")

    # === Plotly figure
    title = (f"Espacio métrico aprendido — {args.checkpoint} | "
             f"split={args.split} | {args.dims}D | {len(viz):,} puntos")

    # hover_data como dict: True = mostrar, False = ocultar.
    # Ocultamos u1/u2/u3 (axes coords) y is_linkable (interno) — solo metadata útil.
    hover_data = {
        "entity_id":   True,
        "record_id":   True,
        "source_db":   True,
        "identidad":   True,
        "u1": False, "u2": False,
        "is_linkable": False,
    }
    if args.dims == 3:
        hover_data["u3"] = False
        fig = px.scatter_3d(
            viz, x="u1", y="u2", z="u3",
            color="source_db",
            color_discrete_map=SOURCE_COLORS,
            hover_data=hover_data,
            title=title,
        )
    else:
        fig = px.scatter(
            viz, x="u1", y="u2",
            color="source_db",
            color_discrete_map=SOURCE_COLORS,
            hover_data=hover_data,
            title=title,
        )

    # Opacity per-point: scatter_3d.marker.opacity NO acepta arrays —
    # truco: pasar colores rgba con alpha embebido por punto.
    if not args.no_singletons:
        for tr in fig.data:
            mask = (viz["source_db"] == tr.name).values
            is_link = viz.loc[mask, "is_linkable"].values
            src_color = SOURCE_COLORS[tr.name]
            rgba_list = [
                _hex_to_rgba(src_color, 0.9 if il else args.singleton_opacity)
                for il in is_link
            ]
            tr.marker.color = rgba_list

    fig.update_traces(marker=dict(size=args.marker_size, line=dict(width=0)))

    # === Fix legend colors: cuando hay rgba per-point (modo con singletons), la legend
    # se confunde y toma un color promedio. Solución: ocultar los traces originales del
    # legend y añadir "legend proxy" traces invisibles con color sólido por source_db.
    import plotly.graph_objects as go

    if not args.no_singletons:
        # Ocultar legend de los traces reales (que tienen rgba per-point)
        for tr in fig.data:
            if tr.name in SOURCE_COLORS:
                tr.showlegend = False
        # Añadir proxy traces con color sólido (puntos vacíos, solo para legend)
        for src, color in SOURCE_COLORS.items():
            if src in viz["source_db"].values:
                proxy_args = dict(
                    mode="markers",
                    marker=dict(size=12, color=color, line=dict(width=0)),
                    name=src,
                    showlegend=True,
                    hoverinfo="skip",
                )
                if args.dims == 3:
                    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], **proxy_args))
                else:
                    fig.add_trace(go.Scatter(x=[None], y=[None], **proxy_args))

    # === Highlight: solo annotations permanentes apuntando a los puntos originales.
    # Mantenemos los puntos con su color/tamaño/style normal — las flechas y los
    # textos son suficientes para identificar el cluster.
    if highlight_df is not None and len(highlight_df) > 0:
        import math
        ann_text_color = "#FFFFFF" if args.dark_bg else "#000000"
        ann_bg = "rgba(30,30,35,0.92)" if args.dark_bg else "rgba(255,253,200,0.95)"
        ann_border = "#FFD700"
        n = len(highlight_df)
        radius = 110  # pixels
        if args.dims == 3:
            scene_annotations = []
            for i, (_, r) in enumerate(highlight_df.iterrows()):
                angle = i * 2 * math.pi / n + math.pi / 6
                ax = radius * math.cos(angle)
                ay = -radius * math.sin(angle)
                scene_annotations.append(dict(
                    x=r["u1"], y=r["u2"], z=r["u3"],
                    text=f"<b>{r['source_db']}</b><br>{r['identidad']}",
                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
                    arrowcolor=ann_border,
                    ax=ax, ay=ay,
                    font=dict(size=12, color=ann_text_color),
                    bgcolor=ann_bg, bordercolor=ann_border, borderwidth=1,
                    borderpad=4,
                ))
            fig.update_layout(scene=dict(annotations=scene_annotations))
        else:
            for i, (_, r) in enumerate(highlight_df.iterrows()):
                angle = i * 2 * math.pi / n + math.pi / 6
                ax = radius * math.cos(angle)
                ay = -radius * math.sin(angle)
                fig.add_annotation(
                    x=r["u1"], y=r["u2"],
                    text=f"<b>{r['source_db']}</b><br>{r['identidad']}",
                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
                    arrowcolor=ann_border, ax=ax, ay=ay,
                    font=dict(size=12, color=ann_text_color),
                    bgcolor=ann_bg, bordercolor=ann_border,
                    borderwidth=1, borderpad=4,
                )

    # === Legend: tamaño consistente en todos los modos (no solo dark-bg)
    fig.update_layout(legend=dict(font=dict(size=14), itemsizing="constant"))

    # Background — usar el template oficial plotly_dark, luego override de bg específico
    # (template coordina legend, modebar, axis labels, gridlines de forma consistente).
    if args.dark_bg:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0f0f13",
            plot_bgcolor="#0f0f13",
            legend=dict(
                bgcolor="rgba(30,30,35,0.85)",
                bordercolor="#444",
                borderwidth=1,
                font=dict(size=14),
                itemsizing="constant",
            ),
        )
        if args.dims == 3:
            fig.update_scenes(
                xaxis=dict(backgroundcolor="#0f0f13", gridcolor="#333", zerolinecolor="#444"),
                yaxis=dict(backgroundcolor="#0f0f13", gridcolor="#333", zerolinecolor="#444"),
                zaxis=dict(backgroundcolor="#0f0f13", gridcolor="#333", zerolinecolor="#444"),
            )
    else:
        # Fondo blanco — Plotly default deja un gris tenue en el cubo de ejes 3D.
        # Lo igualamos al blanco para que el scene se funda con el paper.
        if args.dims == 3:
            fig.update_scenes(
                xaxis=dict(backgroundcolor="#FFFFFF", gridcolor="#DDDDDD", zerolinecolor="#CCCCCC"),
                yaxis=dict(backgroundcolor="#FFFFFF", gridcolor="#DDDDDD", zerolinecolor="#CCCCCC"),
                zaxis=dict(backgroundcolor="#FFFFFF", gridcolor="#DDDDDD", zerolinecolor="#CCCCCC"),
            )

    # === Output
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if args.output_html:
        out_path = Path(args.output_html)
    else:
        suffix = ""
        if args.no_singletons:
            suffix += "_no_singletons"
        if highlight_entity is not None:
            suffix += f"_hl{highlight_entity}"
        out_path = FIGURES_DIR / f"embeddings_{args.checkpoint}_{args.split}_{args.dims}D{suffix}.html"

    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"\n✓ HTML guardado: {out_path}")
    print(f"  Abrir con:  xdg-open {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
