"""
app/components/metric_cards.py
------------------------------
Reusable, styled medical AI metric cards and KPI displays for MediGraph AI.

Design:
- Modern clinical & laboratory UI aesthetic
- Clean white card backgrounds with subtle borders and gentle shadows
- Navy blue (#0F172A, #1E3A8A) and teal (#0D9488, #14B8A6) accents
- Accessible typography with uppercase micro-labels and prominent numeric values
"""

from __future__ import annotations

from typing import Any, List, Optional
import streamlit as st


COLOR_MAP = {
    "teal": {"accent": "#0D9488", "bg": "#F0FDFA", "text": "#115E59"},
    "navy": {"accent": "#1E3A8A", "bg": "#EFF6FF", "text": "#1E40AF"},
    "coral": {"accent": "#EF4444", "bg": "#FEF2F2", "text": "#991B1B"},
    "amber": {"accent": "#F59E0B", "bg": "#FFFBEB", "text": "#92400E"},
    "purple": {"accent": "#8B5CF6", "bg": "#F5F3FF", "text": "#5B21B6"},
    "slate": {"accent": "#64748B", "bg": "#F8FAFC", "text": "#334155"},
}


def render_metric_card(
    title: str,
    value: Any,
    description: Optional[str] = None,
    icon: Optional[str] = None,
    badge: Optional[str] = None,
    color_accent: str = "teal",
) -> None:
    """
    Render a single styled metric card in Streamlit.

    Parameters
    ----------
    title : str
        Header text (e.g. "Total Diseases").
    value : Any
        Primary display metric (e.g. 15, "0.0954", "81.6%").
    description : Optional[str]
        Small caption or subtitle beneath the value.
    icon : Optional[str]
        Emoji or symbol icon (e.g. "🦠", "💊", "🧬").
    badge : Optional[str]
        Small highlight tag (e.g. "+12%", "Ground Truth").
    color_accent : str
        One of "teal", "navy", "coral", "amber", "purple", "slate".
    """
    c_info = COLOR_MAP.get(color_accent, COLOR_MAP["teal"])

    badge_html = (
        f'<span style="background-color: {c_info["bg"]}; color: {c_info["text"]}; '
        f'font-size: 0.72rem; font-weight: 600; padding: 2px 8px; border-radius: 9999px; '
        f'margin-left: 8px;">{badge}</span>'
        if badge
        else ""
    )

    icon_html = f'<span style="font-size: 1.25rem; margin-right: 6px;">{icon}</span>' if icon else ""

    desc_html = (
        f'<div style="color: #64748B; font-size: 0.8rem; margin-top: 4px; line-height: 1.3;">{description}</div>'
        if description
        else ""
    )

    card_html = f"""
    <div style="
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-top: 3px solid {c_info['accent']};
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    ">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
            <div style="color: #64748B; font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">
                {icon_html}{title}
            </div>
            {badge_html}
        </div>
        <div style="color: #0F172A; font-size: 1.85rem; font-weight: 700; line-height: 1.1; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
            {value}
        </div>
        {desc_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_kpi_row(metrics: List[dict]) -> None:
    """
    Render a horizontal grid of metric cards across Streamlit columns.

    Parameters
    ----------
    metrics : List[dict]
        List of dicts with keys matching render_metric_card arguments.
    """
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            render_metric_card(
                title=m["title"],
                value=m["value"],
                description=m.get("description"),
                icon=m.get("icon"),
                badge=m.get("badge"),
                color_accent=m.get("color_accent", "teal"),
            )


def render_section_header(title: str, subtitle: Optional[str] = None, icon: Optional[str] = None) -> None:
    """Render a clean section title with subtle bottom divider."""
    icon_str = f"{icon} " if icon else ""
    st.markdown(
        f"""
        <div style="margin-top: 18px; margin-bottom: 12px; border-bottom: 1px solid #E2E8F0; padding-bottom: 8px;">
            <h3 style="color: #0F172A; font-size: 1.25rem; font-weight: 700; margin: 0; padding: 0;">
                {icon_str}{title}
            </h3>
            {f'<p style="color: #64748B; font-size: 0.88rem; margin: 4px 0 0 0;">{subtitle}</p>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer_banner() -> None:
    """Render the standard biomedical research disclaimer banner."""
    st.markdown(
        """
        <div style="
            background-color: #FFFBEB;
            border-left: 4px solid #F59E0B;
            border-radius: 6px;
            padding: 12px 16px;
            margin: 16px 0;
            color: #92400E;
            font-size: 0.88rem;
            line-height: 1.45;
        ">
            <strong>⚠️ Medical Research Disclaimer:</strong> This platform is an experimental biomedical knowledge
            graph research tool. It does <strong>not</strong> diagnose diseases, prescribe medicines, or provide
            clinical treatment recommendations. All model outputs represent algorithmic affinity scores.
        </div>
        """,
        unsafe_allow_html=True,
    )

