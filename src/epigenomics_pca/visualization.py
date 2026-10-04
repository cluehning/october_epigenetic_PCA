"""Publication-oriented figures and a data-driven research report."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans


COLORS = ["#176B67", "#D36B45", "#5269A3", "#B38A2F", "#7E5B83", "#688849"]


def _labels(metadata: pd.DataFrame, index: pd.Index) -> pd.Series:
    indexed = metadata.set_index("EID", drop=False).reindex(index)
    for column in ("STD_NAME", "ANATOMY", "GROUP"):
        has_labels = (
            column in indexed
            and indexed[column].fillna("").astype(str).str.strip().ne("").any()
        )
        if has_labels:
            return indexed[column].fillna("Unlabeled").astype(str)
    return pd.Series(index.astype(str), index=index, name="Sample")


def _label_palette(labels: pd.Series) -> dict[str, str]:
    """Create one deterministic color for every observed metadata label."""
    colors = sns.color_palette("husl", n_colors=labels.nunique()).as_hex()
    return dict(zip(labels.unique(), colors, strict=True))


def create_figures(
    results: dict[str, object],
    metadata: pd.DataFrame,
    figure_dir: Path,
    cluster_count: int | None = None,
) -> None:
    """Write scree, score, loading, correlation, clustering, and 3D figures."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook", font="DejaVu Sans")
    scores = results["scores"]
    loadings = results["loadings"]
    variance = results["variance"]
    labels = _labels(metadata, scores.index)
    label_palette = _label_palette(labels)
    scores = scores.copy()
    scores["Cell type"] = labels

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    component_numbers = np.arange(1, len(variance) + 1)
    axes[0].bar(
        component_numbers,
        variance["explained_variance_ratio"],
        color=COLORS[0],
    )
    axes[0].set(
        xlabel="Principal component",
        ylabel="Variance explained",
        title="Scree plot",
    )
    axes[1].plot(
        component_numbers,
        variance["cumulative_variance_ratio"],
        marker="o",
        color=COLORS[1],
    )
    axes[1].axhline(
        0.8,
        color="#777777",
        linestyle="--",
        linewidth=1,
        label="80% reference",
    )
    axes[1].set(
        xlabel="Number of components",
        ylabel="Cumulative variance explained",
        ylim=(0, 1.02),
    )
    axes[1].legend(frameon=False)
    fig.savefig(figure_dir / "scree_plot.png", dpi=300)
    plt.close(fig)

    for first, second in (("PC1", "PC2"), ("PC1", "PC3"), ("PC2", "PC3")):
        if first not in scores or second not in scores:
            continue
        fig, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
        sns.scatterplot(
            data=scores,
            x=first,
            y=second,
            hue="Cell type",
            palette=label_palette,
            s=90,
            ax=axis,
        )
        for sample, row in scores.iterrows():
            axis.annotate(
                str(sample),
                (row[first], row[second]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )
        axis.set_title(f"Epigenomes in {first}-{second} space")
        axis.legend(
            title="Cell type",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=False,
        )
        filename = f"{first.lower()}_{second.lower()}_scatter.png"
        figure_path = figure_dir / filename
        fig.savefig(figure_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    if all(component in scores for component in ("PC1", "PC2", "PC3")):
        import plotly.express as px

        interactive = px.scatter_3d(
            scores.reset_index(), x="PC1", y="PC2", z="PC3", color="Cell type",
            hover_name=scores.index.name or "index",
            color_discrete_map=label_palette,
            title="Roadmap epigenomes: first three principal components",
        )
        interactive.update_layout(
            template="plotly_white", legend_title_text="Cell type"
        )
        interactive.write_html(
            figure_dir / "pca_3d.html", include_plotlyjs="cdn"
        )

    for component in ("PC1", "PC2", "PC3"):
        if component not in loadings:
            continue
        strongest = loadings[component].abs().nlargest(15).sort_values()
        fig, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
        axis.barh(
            strongest.index.astype(str),
            loadings.loc[strongest.index, component],
            color=COLORS[2],
        )
        axis.set(
            xlabel="Loading",
            ylabel="Genomic feature",
            title=f"Largest absolute {component} loadings",
        )
        fig.savefig(figure_dir / f"{component.lower()}_loadings.png", dpi=300)
        plt.close(fig)

    correlation = results["scores"].corr()
    fig, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    sns.heatmap(
        correlation,
        vmin=-1,
        vmax=1,
        center=0,
        cmap="vlag",
        square=True,
        ax=axis,
    )
    axis.set_title("Correlation between principal-component scores")
    fig.savefig(figure_dir / "pc_score_correlation.png", dpi=300)
    plt.close(fig)

    available_components = [
        name for name in ("PC1", "PC2", "PC3") if name in scores
    ]
    cluster_input = results["scores"].loc[:, available_components].to_numpy()
    count = cluster_count or min(
        max(2, round(np.sqrt(len(scores)))), len(scores)
    )
    assignments = KMeans(
        n_clusters=count, random_state=42, n_init=20
    ).fit_predict(cluster_input)
    cluster_frame = results["scores"].copy()
    cluster_frame["cluster"] = assignments.astype(str)
    cluster_palette = _label_palette(cluster_frame["cluster"])
    annotated_clusters = cluster_frame.join(
        metadata.set_index("EID", drop=False), how="left"
    )
    annotated_clusters.to_csv(figure_dir / "cluster_assignments.csv")
    if len(available_components) >= 2:
        fig, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
        sns.scatterplot(
            data=cluster_frame,
            x=available_components[0],
            y=available_components[1],
            hue="cluster", palette=cluster_palette, s=100, ax=axis,
        )
        axis.set_title("Unsupervised clusters in PCA space")
        axis.legend(title="K-means cluster", frameon=False)
        fig.savefig(figure_dir / "pca_clusters.png", dpi=300)
        plt.close(fig)


def write_report(
    results: dict[str, object], metadata: pd.DataFrame, report_path: Path
) -> None:
    """Write a report whose numerical claims come from the analyzed data."""
    variance = results["variance"]
    diagnostics = results["diagnostics"]
    associations = results["associations"]
    top = pd.read_csv(report_path.parent / "top_loadings.csv")
    reconstruction_errors = results["reconstruction_errors"]
    source_manifest = report_path.parent / "source_manifest.csv"
    summary_note = ""
    if source_manifest.exists():
        source_rows = pd.read_csv(source_manifest)
        if source_rows["exact"].eq(False).all():
            summary_note = (
                "Remote bins use approximate means from Roadmap's precomputed "
                "zoom summaries.\n\n"
            )
        else:
            summary_note = "Local bins use exact means.\n\n"
    first_three = variance.head(3)
    usable_associations = associations.loc[~associations["saturated"]]
    association_rows = usable_associations.sort_values(
        "eta_squared", ascending=False
    ).head(6)
    if association_rows.empty:
        association_text = (
            "No metadata association is estimable: each tested label was "
            "unique per sample, which would make $\\eta^2=1$ by construction. "
            "Replicated samples per category are needed."
        )
    else:
        association_text = "\n".join(
            f"- {row.metadata_field} vs. {row.component}: "
            f"$\\eta^2={row.eta_squared:.3f}$ "
            f"({row.category_count} observed categories)."
            for row in association_rows.itertuples(index=False)
        )
    loading_text = "\n".join(
        f"- {row.component}: `{row.feature}` (loading {row.loading:+.4f})."
        for row in top.groupby("component", sort=False)
        .head(5)
        .itertuples(index=False)
    )
    variance_text = "\n".join(
        f"- {row.component}: {row.explained_variance_ratio:.1%} "
        f"(cumulative {row.cumulative_variance_ratio:.1%})."
        for row in first_three.itertuples(index=False)
    )
    leading_share = (
        float(first_three["cumulative_variance_ratio"].iloc[-1])
        if len(first_three)
        else 0.0
    )
    rank_three = reconstruction_errors.iloc[
        min(2, len(reconstruction_errors) - 1)
    ]
    rank_two = reconstruction_errors.iloc[
        min(1, len(reconstruction_errors) - 1)
    ]
    sample_rank_limit = diagnostics["sample_count"] - 1
    rank_note = ""
    if diagnostics["component_count"] >= sample_rank_limit:
        rank_note = (
            f"With {diagnostics['sample_count']} samples, centering "
            "limits the "
            f"rank to at most {sample_rank_limit}; the near-zero rank-"
            f"{int(rank_three['retained_components'])} error is therefore "
            "expected and does not establish biological low dimensionality. "
            "Treat the first one or two PCs as exploratory.\n\n"
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_text = (
        "# Results: latent dimensions of epigenetic variation\n\n"
        "## Data and scope\n\n"
        f"The analysis contains **{diagnostics['sample_count']} epigenomes** "
        "and "
        f"**{diagnostics['feature_count']:,} genomic-window features**.\n\n"
        "The matrix contains Roadmap signals on a shared genomic grid, "
        "log-transformed and standardized per feature. See "
        "`sample_metadata.csv` and `source_manifest.csv` for labels and "
        "queried source URLs/settings. Downloaded runs also include SHA-256 "
        "records in `data/raw/download_manifest.csv`.\n\n"
        f"{summary_note}"
        "## Variance captured\n\n"
        f"The first three components explain {leading_share:.1%} of variance "
        "in this matrix. The participation-ratio effective dimension is "
        f"{diagnostics['effective_dimension']:.2f}. The relative "
        f"reconstruction error is "
        f"{diagnostics['rank_1_reconstruction_error']:.3g} at rank 1 and "
        f"{rank_two['relative_reconstruction_error']:.3g} at rank 2 and "
        f"{rank_three['relative_reconstruction_error']:.3g} at rank "
        f"{int(rank_three['retained_components'])}; the full retained-rank "
        f"error is {diagnostics['relative_reconstruction_error']:.3g}.\n\n"
        f"{variance_text}\n\n"
        f"{rank_note}"
        "Among these samples and marks, the first three dominant patterns "
        "summarize the stated share of feature variation. This describes the "
        "selected dataset, not every human epigenome.\n\n"
        "## Metadata associations\n\n"
        "The reported $\\eta^2$ is the fraction of a component's score "
        "variation lying between metadata categories. Singleton categories "
        "are flagged as saturated and omitted from interpretation. Any other "
        "association is descriptive, not causal; labels may be confounded "
        "with study or assay batch.\n\n"
        f"{association_text}\n\n"
        "## Major features\n\n"
        "Largest absolute loadings identify windows most aligned with each "
        "component. Both signs matter, and a component's sign is arbitrary. "
        "Window associations do not by themselves identify a regulatory "
        "mechanism or biological function.\n\n"
        f"{loading_text}\n\n"
        "## Numerical cross-check\n\n"
        "Covariance-eigenvalue and direct-SVD implementations were checked "
        "against scikit-learn by comparing retained eigenvalues. "
        "Loading signs "
        "may differ because an eigenvector and its negative describe the same "
        "axis. See `diagnostics.json`.\n\n"
        "## Interpretation and limitations\n\n"
        "PCA is linear and prioritizes total variance, not biological "
        "importance. Results depend on samples, mark, window size, transform, "
        "and standardization. BigWig gaps are treated as zero signal. Batch "
        "effects, cell composition, and biological heterogeneity can dominate "
        "a component. Standardization gives each retained feature equal "
        "variance and can amplify noisy low-signal windows. Components do not "
        "establish causation, and dimensionality reduction discards "
        "information.\n\n"
        "## Figures and reproducibility\n\n"
        "Figures are in `figures/`. Run `python -m epigenomics_pca all` to "
        "rediscover tracks, download the configured subset, process tracks, "
        "recompute this report, and regenerate figures. Retrieval date, URL, "
        "byte count, and SHA-256 for each track are in the manifest. Use the "
        "`remote` command for approximate zoom summaries without full-track "
        "downloads.\n"
    )
    report_path.write_text(report_text, encoding="utf-8")
