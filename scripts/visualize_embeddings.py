"""Visualización del espacio métrico aprendido por el Bi-Encoder.

Lee los embeddings canónicos exportados, selecciona los registros de un split y
los reduce de 768D → 3D (o 2D) con UMAP (métrica coseno). Produce un scatter
interactivo de Plotly como HTML.

El resultado es la prueba visual directa del aprendizaje métrico: si Δ > 2,
las "islas" de cada paciente (sus 2-3 registros en distintas bases) deben
aparecer compactas y separadas entre sí.

Diseño alineado a `docs/Anexos/demo-propuesta.md` § Paso 3:
- UMAP con métrica cosine (no euclidiana — el espacio del BE es coseno)
- Colores fijos por source_db: rojo Comorbilidad, verde Económico, azul TS
- Entidades de una sola base de datos opcionales con opacity diferenciada

Uso:
    # 3D por default sobre split=test de la variante canónica
    python scripts/visualize_embeddings.py \\
        --dataset ~/Data/INER/modeling/data/tok_skipnull/split.parquet

    # 2D
    python scripts/visualize_embeddings.py --dataset <parquet> --dims 2

    # Solo entidades vinculables (sin entidades de una sola base de datos)
    python scripts/visualize_embeddings.py --dataset <parquet> --no-single-source

    # Fondo oscuro estilo presentación
    python scripts/visualize_embeddings.py --dataset <parquet> --dark-bg
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import umap
import plotly.express as px

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from record_linkage.config import (
    DEFAULT_VARIANT,
    FIGURES_DIR,
    SUPPORTED_VARIANTS,
    embeddings_path,
)
from record_linkage.data.columns import SOURCE_IDENTITY_COLUMNS
from record_linkage.evaluation.biencoder_eval import load_dataset_split


def _extract_field(text: str, col: str) -> str:
    """Extrae el valor de `[COL] <col> [VAL] <valor>` del texto serializado.

    El valor termina en ` [COL]` (siguiente campo), en ` [BLK_` (siguiente bloque)
    o al final del string. Devuelve '?' si el campo no está.
    """
    marker = f"[COL] {col} [VAL] "
    start = text.find(marker)
    if start == -1:
        return "?"
    start += len(marker)
    end_candidates = [text.find(" [COL] ", start), text.find(" [BLK_", start)]
    end_candidates = [e for e in end_candidates if e != -1]
    end = min(end_candidates) if end_candidates else len(text)
    return text[start:end].strip() or "?"


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


def build_identity_map(dataset_df: pd.DataFrame) -> dict:
    """Construye un dict {record_id: (nombre, expediente)} desde el dataset.

    Toma los campos de identidad preservados en la serialización según la
    fuente, sin depender de los CSVs limpios.

    Args:
        dataset_df: DataFrame con columnas `record_id` y `text`.

    Returns:
        dict record_id → (nombre, expediente) para hover de Plotly.
    """
    identity_map = {}
    for record_id, source_db, text in zip(
        dataset_df["record_id"], dataset_df["source_db"], dataset_df["text"]
    ):
        exp_col, nombre_col = SOURCE_IDENTITY_COLUMNS[source_db]
        nombre = _extract_field(text, nombre_col)
        exp = _extract_field(text, exp_col)
        identity_map[record_id] = nombre, exp
    return identity_map


def main():
    parser = argparse.ArgumentParser(description="Visualiza el espacio métrico del Bi-Encoder con UMAP+Plotly")
    parser.add_argument("--variant", choices=SUPPORTED_VARIANTS, default=DEFAULT_VARIANT)
    parser.add_argument("--dataset", required=True,
                        help="Ruta al dataset.parquet o dataset_split.parquet")
    parser.add_argument(
        "--embeddings", type=Path, default=None,
        help="Parquet [record_id, embedding] (default: modeling/embeddings/<variant>/embeddings.parquet)",
    )
    parser.add_argument("--split", default="test",
                        choices=["train", "val", "test", "all"],
                        help="Split a visualizar (default: test). 'all' = sin filtrar por split")
    parser.add_argument("--dims", type=int, default=3, choices=[2, 3],
                        help="Dimensiones del scatter (default: 3)")
    parser.add_argument("--no-single-source", action="store_true",
                        help="Omitir entidades de una sola base; solo muestra entidades vinculables")
    parser.add_argument("--n-neighbors", type=int, default=15,
                        help="UMAP n_neighbors — balance local/global (default: 15, recomendado en demo-propuesta)")
    parser.add_argument("--min-dist", type=float, default=0.1,
                        help="UMAP min_dist — compacidad de clusters (default: 0.1)")
    parser.add_argument("--dark-bg", action="store_true",
                        help="Fondo oscuro estilo presentación (#0f0f13). Default: blanco.")
    parser.add_argument("--marker-size", type=int, default=5,
                        help="Tamaño de los puntos (default: 5 para 3D, sube si hay pocos puntos)")
    parser.add_argument("--single-source-opacity", type=float, default=0.25,
                        help="Opacidad de entidades de una sola base (0.0–1.0, default: 0.25).")
    parser.add_argument("--output-html", default=None,
                        help="Ruta del HTML (default: outputs/figures/embeddings/<checkpoint de procedencia>_<split>_<dims>D.html)")
    highlight_group = parser.add_mutually_exclusive_group()
    highlight_group.add_argument("--highlight-entity", type=int, default=None,
                                  help="entity_id específico a resaltar — añade annotations permanentes "
                                       "y marker grande sobre sus 2-3 registros del cluster")
    highlight_group.add_argument("--highlight-random-cluster", action="store_true",
                                  help="Selecciona aleatoriamente (con --seed) un cluster con 3 registros "
                                       "(uno por source_db) para destacar")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # === Carga dataset
    dataset_path = Path(args.dataset)
    split = None if args.split == "all" else args.split
    df = load_dataset_split(dataset_path, split=split)

    # === Identificar entidades vinculables vs de una sola base
    entity_sources = df.groupby("entity_id")["source_db"].nunique()
    linkable_ids = set(entity_sources[entity_sources > 1].index)
    df = df.copy()
    df["is_linkable"] = df["entity_id"].isin(linkable_ids)

    if args.no_single_source:
        df = df[df["is_linkable"]].reset_index(drop=True)
        print(f"  Solo vinculables: {len(df):,} registros, {len(linkable_ids):,} entidades")
    else:
        n_link = int(df["is_linkable"].sum())
        n_single_source = len(df) - n_link
        print(f"  Total: {len(df):,} registros | de entidades vinculables: {n_link:,} registros | "
              f"de entidades de una sola base: {n_single_source:,} registros")

    # === Embeddings canónicos
    embeddings_file = args.embeddings.expanduser() if args.embeddings else embeddings_path(args.variant)
    if not embeddings_file.is_file():
        raise FileNotFoundError(f"Embeddings no encontrados: {embeddings_file}")
    embeddings_df = pd.read_parquet(embeddings_file, columns=["record_id", "embedding"])
    if embeddings_df["record_id"].duplicated().any():
        raise ValueError(f"Embeddings con record_id duplicados: {embeddings_file}")
    embeddings_df = embeddings_df.set_index("record_id")
    missing_ids = set(df["record_id"]) - set(embeddings_df.index)
    if missing_ids:
        raise ValueError(f"Faltan {len(missing_ids):,} record_id del split en {embeddings_file}")

    print(f"\nCargando embeddings: {embeddings_file}")
    t0 = time.time()
    embeddings = np.stack(embeddings_df.loc[df["record_id"], "embedding"].to_numpy()).astype(np.float32)
    print(f"  Embeddings cargados en {time.time()-t0:.1f}s | shape={embeddings.shape}")

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
    # Campos de identidad para hover, parseados del texto serializado.
    identity_map = build_identity_map(df)
    identities = df["record_id"].map(identity_map)
    viz["nombre"] = [identity[0] for identity in identities]
    viz["expediente"] = [identity[1] for identity in identities]

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
            print(f"    {r['source_db']:<16}: Nombre = {r['nombre']} | Exp = {r['expediente']}")

    # === Plotly figure
    title_suffix = "registros vinculables" if args.no_single_source else "registros"
    title = f"Espacio métrico aprendido | {len(viz):,} {title_suffix}"

    # hover_data como dict: True = mostrar, False = ocultar.
    # Ocultamos u1/u2/u3 (axes coords) y is_linkable (interno) — solo metadata útil.
    hover_data = {
        "entity_id":   True,
        "record_id":   True,
        "source_db":   True,
        "nombre":      True,
        "expediente":  True,
        "u1": False, "u2": False,
        "is_linkable": False,
    }
    labels = {
        "source_db": "Base de datos",
        "entity_id": "Entidad",
        "record_id": "Registro",
        "nombre": "Nombre",
        "expediente": "Expediente",
    }
    if args.dims == 3:
        hover_data["u3"] = False
        fig = px.scatter_3d(
            viz, x="u1", y="u2", z="u3",
            color="source_db",
            color_discrete_map=SOURCE_COLORS,
            hover_data=hover_data,
            labels=labels,
            title=title,
        )
    else:
        fig = px.scatter(
            viz, x="u1", y="u2",
            color="source_db",
            color_discrete_map=SOURCE_COLORS,
            hover_data=hover_data,
            labels=labels,
            title=title,
        )

    # Opacity per-point: scatter_3d.marker.opacity NO acepta arrays —
    # truco: pasar colores rgba con alpha embebido por punto.
    if not args.no_single_source:
        for tr in fig.data:
            mask = (viz["source_db"] == tr.name).values
            is_link = viz.loc[mask, "is_linkable"].values
            src_color = SOURCE_COLORS[tr.name]
            rgba_list = [
                _hex_to_rgba(src_color, 0.9 if il else args.single_source_opacity)
                for il in is_link
            ]
            tr.marker.color = rgba_list

    fig.update_traces(marker=dict(size=args.marker_size, line=dict(width=0)))

    # === Fix legend colors: cuando hay rgba per-point (incluye entidades de una sola base), la legend
    # se confunde y toma un color promedio. Solución: ocultar los traces originales del
    # legend y añadir "legend proxy" traces invisibles con color sólido por source_db.
    import plotly.graph_objects as go

    if not args.no_single_source:
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
                    text=f"<b>{r['source_db']}</b><br>Nombre = {r['nombre']}<br>Exp = {r['expediente']}",
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
                    text=f"<b>{r['source_db']}</b><br>Nombre = {r['nombre']}<br>Exp = {r['expediente']}",
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
    output_dir = FIGURES_DIR / "embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_html:
        out_path = Path(args.output_html)
    else:
        suffix = ""
        if args.no_single_source:
            suffix += "_linkable"
        if args.dark_bg:
            suffix += "_dark"
        if highlight_entity is not None:
            suffix += f"_highlight-{highlight_entity}"
        out_path = output_dir / f"embedding_space_{args.split}_{args.dims}d{suffix}.html"

    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"\n✓ HTML guardado: {out_path}")
    print(f"  Abrir con:  xdg-open {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
