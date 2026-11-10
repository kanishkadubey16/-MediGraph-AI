"""
app/pages/04_model_performance.py
---------------------------------
Page 4: Model Performance and Empirical Evaluation Studio.

Displays:
- Actual test set metrics: MRR, Hits@1, Hits@3, Hits@10, ROC-AUC, Precision, Recall, F1.
- Training loss and validation loss convergence curves over training epochs.
- Validation MRR and Hits@10 progression across epochs.
- Baseline Comparison: Side-by-side empirical comparison between pure DistMult and R-GCN.
- Per-relation breakdown of ranking and link prediction accuracy.
- Hyperparameter and architecture configuration details.

All metrics are loaded directly from `artifacts/metrics/metrics.json` and
`artifacts/metrics/baseline_comparison.json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Bootstrap sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.utils.config import load_config
from app.components.data_loader import load_project_data
from app.components.metric_cards import (
    render_metric_card,
    render_kpi_row,
    render_section_header,
    render_disclaimer_banner,
)

st.set_page_config(
    page_title="Model Performance — MediGraph AI",
    page_icon="📈",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Initialise & Load Configuration / Metrics
# ---------------------------------------------------------------------------
cfg = load_config()

st.title("📈 GNN Model Performance & Empirical Evaluation")
st.markdown(
    "<p style='color: #64748B; font-size: 1.05rem; margin-top: -10px;'>"
    "Rigorous evaluation metrics, training convergence curves, and DistMult baseline ablation."
    "</p>",
    unsafe_allow_html=True,
)

render_disclaimer_banner()

metrics_file = Path("artifacts/metrics/metrics.json")
baseline_file = Path("artifacts/metrics/baseline_comparison.json")
eval_file = Path("artifacts/metrics/eval_metrics.json")

if not metrics_file.exists():
    st.warning(
        "⚠️ **Experiment metrics file not found (`artifacts/metrics/metrics.json`).**\n\n"
        "Train the model to generate empirical metrics:\n"
        "```bash\npython src/training/train.py --epochs 50 --device cpu\n```"
    )
    st.stop()

with open(metrics_file, "r", encoding="utf-8") as f:
    metrics_data = json.load(f)

baseline_data = None
if baseline_file.exists():
    with open(baseline_file, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)

eval_data = None
if eval_file.exists():
    with open(eval_file, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

best_stats = metrics_data.get("best", {})
per_epoch = metrics_data.get("per_epoch", [])
model_cfg = metrics_data.get("config", {}).get("model", {})
train_cfg = metrics_data.get("config", {}).get("training", {})

# ---------------------------------------------------------------------------
# 1. Primary Evaluation KPI Cards
# ---------------------------------------------------------------------------
render_section_header("Held-Out Test Set Performance (R-GCN + DistMult)", "Strict filtered link prediction protocol (zero data leakage)", "🏆")

test_mrr = best_stats.get("test_mrr", 0.0)
test_hits_1 = best_stats.get("test_hits_at_1", 0.0)
test_hits_10 = best_stats.get("test_hits_at_10", 0.0)
test_auc = best_stats.get("test_auc_roc", 0.0)

row1 = [
    {
        "title": "Mean Reciprocal Rank (MRR)",
        "value": f"{test_mrr:.4f}",
        "description": "Inverse rank of true tail",
        "icon": "🎯",
        "badge": "Filtered",
        "color_accent": "teal",
    },
    {
        "title": "Hits @ 10",
        "value": f"{test_hits_10 * 100:.1f}%",
        "description": "True entity ranked in top 10",
        "icon": "🔟",
        "color_accent": "navy",
    },
    {
        "title": "Hits @ 1",
        "value": f"{test_hits_1 * 100:.1f}%",
        "description": "True entity ranked first",
        "icon": "🥇",
        "color_accent": "purple",
    },
    {
        "title": "ROC-AUC Score",
        "value": f"{test_auc:.4f}",
        "description": "Discrimination vs negatives",
        "icon": "📊",
        "color_accent": "amber",
    },
]
render_kpi_row(row1)

if eval_data:
    row2 = [
        {"title": "Hits @ 3", "value": f"{eval_data.get('hits@3', 0.0) * 100:.1f}%", "description": "Ranked in top 3", "icon": "🥉", "color_accent": "teal"},
        {"title": "Precision", "value": f"{eval_data.get('precision', 0.0):.4f}", "description": "Positive predictive value", "icon": "✔️", "color_accent": "navy"},
        {"title": "Recall", "value": f"{eval_data.get('recall', 0.0):.4f}", "description": "True positive rate", "icon": "🔄", "color_accent": "coral"},
        {"title": "Accuracy", "value": f"{eval_data.get('accuracy', 0.0) * 100:.1f}%", "description": "Binary edge classification", "icon": "⚖️", "color_accent": "slate"},
    ]
    render_kpi_row(row2)

# ---------------------------------------------------------------------------
# 2. Training Convergence & Validation Curves
# ---------------------------------------------------------------------------
render_section_header("Training Convergence & Learning Dynamics", "Loss curves and validation rank progression over epochs", "📉")

col_curve1, col_curve2 = st.columns(2)

if per_epoch:
    df_epoch = pd.DataFrame(per_epoch)

    with col_curve1:
        st.markdown("#### Loss Convergence (BCE with Logits)")
        fig_loss = go.Figure()
        fig_loss.add_trace(
            go.Scatter(
                x=df_epoch["epoch"],
                y=df_epoch["train_loss"],
                mode="lines+markers",
                name="Training Loss",
                line=dict(color="#0D9488", width=2.5),
                marker=dict(size=6),
            )
        )
        if "val_loss" in df_epoch.columns:
            fig_loss.add_trace(
                go.Scatter(
                    x=df_epoch["epoch"],
                    y=df_epoch["val_loss"],
                    mode="lines+markers",
                    name="Validation Loss",
                    line=dict(color="#EF4444", width=2, dash="dash"),
                    marker=dict(size=5),
                )
            )
        fig_loss.update_layout(
            xaxis_title="Epoch",
            yaxis_title="Loss",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=30, b=40, l=10, r=10),
            height=340,
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
        )
        st.plotly_chart(fig_loss, use_container_width=True)

    with col_curve2:
        st.markdown("#### Validation MRR Progression")
        fig_mrr = go.Figure()
        fig_mrr.add_trace(
            go.Scatter(
                x=df_epoch["epoch"],
                y=df_epoch["val_mrr"],
                mode="lines+markers",
                name="Validation MRR",
                line=dict(color="#1E3A8A", width=2.5),
                marker=dict(size=6),
            )
        )
        if "val_hits_at_10" in df_epoch.columns:
            fig_mrr.add_trace(
                go.Scatter(
                    x=df_epoch["epoch"],
                    y=df_epoch["val_hits_at_10"],
                    mode="lines+markers",
                    name="Hits@10",
                    line=dict(color="#F59E0B", width=2, dash="dot"),
                    marker=dict(size=5),
                )
            )
        fig_mrr.update_layout(
            xaxis_title="Epoch",
            yaxis_title="Ranking Metric Score",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=30, b=40, l=10, r=10),
            height=340,
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
        )
        st.plotly_chart(fig_mrr, use_container_width=True)

# ---------------------------------------------------------------------------
# 3. Baseline Comparison: R-GCN vs DistMult
# ---------------------------------------------------------------------------
render_section_header("Model Ablation: R-GCN vs DistMult Baseline", "Comparison against DistMult without graph convolution on identical splits", "⚖️")

if baseline_data:
    rgcn_metrics = baseline_data.get("rgcn", {})
    base_metrics = baseline_data.get("baseline_distmult", {})
    delta_mrr = baseline_data.get("delta_mrr", 0.0)

    col_b1, col_b2 = st.columns([1.2, 1.8])

    with col_b1:
        st.markdown("#### Comparative Metric Summary")
        comparison_rows = [
            {"Metric": "MRR (Mean Reciprocal Rank)", "DistMult Baseline": f"{base_metrics.get('mrr', 0.0):.4f}", "R-GCN (Proposed)": f"{rgcn_metrics.get('mrr', 0.0):.4f}", "Delta (Δ)": f"{delta_mrr:+.4f}"},
            {"Metric": "Hits @ 10", "DistMult Baseline": f"{base_metrics.get('hits@10', 0.0):.4f}", "R-GCN (Proposed)": f"{rgcn_metrics.get('hits@10', 0.0):.4f}", "Delta (Δ)": f"{rgcn_metrics.get('hits@10', 0.0) - base_metrics.get('hits@10', 0.0):+.4f}"},
            {"Metric": "Hits @ 1", "DistMult Baseline": f"{base_metrics.get('hits@1', 0.0):.4f}", "R-GCN (Proposed)": f"{rgcn_metrics.get('hits@1', 0.0):.4f}", "Delta (Δ)": f"{rgcn_metrics.get('hits@1', 0.0) - base_metrics.get('hits@1', 0.0):+.4f}"},
            {"Metric": "ROC-AUC", "DistMult Baseline": f"{base_metrics.get('roc_auc', 0.0):.4f}", "R-GCN (Proposed)": f"{rgcn_metrics.get('roc_auc', 0.0):.4f}", "Delta (Δ)": f"{rgcn_metrics.get('roc_auc', 0.0) - base_metrics.get('roc_auc', 0.0):+.4f}"},
            {"Metric": "Accuracy", "DistMult Baseline": f"{base_metrics.get('accuracy', 0.0):.4f}", "R-GCN (Proposed)": f"{rgcn_metrics.get('accuracy', 0.0):.4f}", "Delta (Δ)": f"{rgcn_metrics.get('accuracy', 0.0) - base_metrics.get('accuracy', 0.0):+.4f}"},
        ]
        df_comp = pd.DataFrame(comparison_rows)
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

    with col_b2:
        st.markdown("#### Performance Metric Comparison")
        metrics_to_plot = ["mrr", "hits@10", "accuracy"]
        labels_plot = ["MRR", "Hits@10", "Accuracy"]
        base_vals = [base_metrics.get(m, 0.0) for m in metrics_to_plot]
        rgcn_vals = [rgcn_metrics.get(m, 0.0) for m in metrics_to_plot]

        fig_comp = go.Figure(
            data=[
                go.Bar(name="DistMult Baseline", x=labels_plot, y=base_vals, marker_color="#94A3B8"),
                go.Bar(name="R-GCN (Proposed)", x=labels_plot, y=rgcn_vals, marker_color="#0D9488"),
            ]
        )
        fig_comp.update_layout(
            barmode="group",
            yaxis_title="Score",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=30, b=30, l=10, r=10),
            height=280,
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
        )
        st.plotly_chart(fig_comp, use_container_width=True)
else:
    st.info("Baseline comparison data not found. Run `python src/training/baseline.py` to generate side-by-side ablation.")

# ---------------------------------------------------------------------------
# 4. Architecture Specifications
# ---------------------------------------------------------------------------
render_section_header("Model Architecture & Hyperparameter Configuration", "Configured R-GCN encoder and DistMult bilinear decoder specifications", "⚙️")

arch_cols = st.columns(4)
with arch_cols[0]:
    render_metric_card("Encoder Architecture", "R-GCN (2 Layers)", "Message passing over typed edges", icon="🧠", color_accent="navy")
with arch_cols[1]:
    render_metric_card("Embedding Channels", f"{model_cfg.get('out_channels', 128)}", "Node & relation vector dim", icon="📏", color_accent="teal")
with arch_cols[2]:
    render_metric_card("Basis Decomposition", f"{model_cfg.get('num_bases', 30)} Bases", "Weight sharing across relations", icon="🧱", color_accent="purple")
with arch_cols[3]:
    render_metric_card("Decoder Architecture", "DistMult Bilinear", "score = sum(h * r * t)", icon="✨", color_accent="amber")
