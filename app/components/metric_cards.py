"""
app/components/metric_cards.py
------------------------------
Reusable, styled medical AI metric cards, KPI displays, and banners for MediGraph AI.

Design:
- Modern clinical & laboratory AI SaaS aesthetic
- Clean white card backgrounds with subtle borders, gentle shadows, and micro-interactions
- Navy blue (#0F172A, #1E3A8A) and medical teal (#0D9488, #14B8A6) accents
- Accessible typography with uppercase micro-labels and prominent numeric values
"""

from __future__ import annotations

from typing import Any, List, Optional
import streamlit as st


COLOR_MAP = {
    "teal": {"accent": "#0D9488", "bg": "#F0FDFA", "text": "#115E59", "glow": "rgba(13, 148, 136, 0.15)"},
    "navy": {"accent": "#1E3A8A", "bg": "#EFF6FF", "text": "#1E40AF", "glow": "rgba(30, 58, 138, 0.12)"},
    "coral": {"accent": "#EF4444", "bg": "#FEF2F2", "text": "#991B1B", "glow": "rgba(239, 68, 68, 0.12)"},
    "amber": {"accent": "#F59E0B", "bg": "#FFFBEB", "text": "#92400E", "glow": "rgba(245, 158, 11, 0.15)"},
    "purple": {"accent": "#8B5CF6", "bg": "#F5F3FF", "text": "#5B21B6", "glow": "rgba(139, 92, 246, 0.15)"},
    "emerald": {"accent": "#10B981", "bg": "#ECFDF5", "text": "#065F46", "glow": "rgba(16, 185, 129, 0.15)"},
    "cyan": {"accent": "#06B6D4", "bg": "#ECFEFF", "text": "#155E75", "glow": "rgba(6, 182, 212, 0.15)"},
    "slate": {"accent": "#64748B", "bg": "#F8FAFC", "text": "#334155", "glow": "rgba(100, 116, 139, 0.12)"},
}


def render_metric_card(
    title: str,
    value: Any,
    description: Optional[str] = None,
    icon: Optional[str] = None,
    badge: Optional[str] = None,
    color_accent: str = "teal",
) -> None:
    """Render a single styled medical AI metric card."""
    c_info = COLOR_MAP.get(color_accent, COLOR_MAP["teal"])

    badge_html = (
        f'<span style="background-color: {c_info["bg"]}; color: {c_info["text"]}; '
        f'border: 1px solid {c_info["accent"]}33; font-size: 0.70rem; font-weight: 600; '
        f'padding: 2px 8px; border-radius: 9999px; letter-spacing: 0.02em;">{badge}</span>'
        if badge
        else ""
    )

    icon_html = f'<span style="font-size: 1.2rem; margin-right: 6px;">{icon}</span>' if icon else ""

    desc_html = (
        f'<div style="color: #64748B; font-size: 0.78rem; margin-top: 6px; line-height: 1.35; font-weight: 450;">{description}</div>'
        if description
        else ""
    )

    card_html = f"""
    <div style="
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-top: 3px solid {c_info['accent']};
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 14px;
        box-shadow: 0 2px 5px rgba(15, 23, 42, 0.04), 0 1px 2px rgba(15, 23, 42, 0.02);
        transition: all 0.2s ease-in-out;
    ">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
            <div style="color: #64748B; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; display: flex; align-items: center;">
                {icon_html}{title}
            </div>
            {badge_html}
        </div>
        <div style="color: #0F172A; font-size: 1.95rem; font-weight: 750; line-height: 1.1; letter-spacing: -0.02em; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
            {value}
        </div>
        {desc_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_kpi_row(metrics: List[dict]) -> None:
    """Render a horizontal grid of metric cards across Streamlit columns."""
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
    """Render a clean clinical section title with a styled icon badge."""
    icon_str = f"""<span style="display: inline-flex; align-items: center; justify-content: center; width: 28px; height: 28px; background: #F0FDFA; border-radius: 8px; margin-right: 8px; font-size: 1rem;">{icon}</span>""" if icon else ""
    st.markdown(
        f"""
        <div style="margin-top: 22px; margin-bottom: 14px; border-bottom: 1px solid #E2E8F0; padding-bottom: 10px;">
            <div style="display: flex; align-items: center;">
                {icon_str}
                <h3 style="color: #0F172A; font-size: 1.22rem; font-weight: 700; margin: 0; padding: 0; letter-spacing: -0.01em;">
                    {title}
                </h3>
            </div>
            {f'<p style="color: #64748B; font-size: 0.86rem; margin: 4px 0 0 0; line-height: 1.4;">{subtitle}</p>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer_banner() -> None:
    """Render the standard biomedical research disclaimer banner."""
    st.markdown(
        """
        <div style="
            background: linear-gradient(90deg, #FFFBEB 0%, #FEF3C7 100%);
            border: 1px solid #FDE68A;
            border-left: 5px solid #F59E0B;
            border-radius: 10px;
            padding: 12px 18px;
            margin: 14px 0 20px 0;
            color: #92400E;
            font-size: 0.85rem;
            line-height: 1.45;
            display: flex;
            align-items: center;
            box-shadow: 0 1px 3px rgba(245, 158, 11, 0.08);
        ">
            <span style="font-size: 1.3rem; margin-right: 12px; flex-shrink: 0;">⚠️</span>
            <div>
                <strong>Research & Educational Prototype Only:</strong>
                MediGraph AI is developed strictly for computational biomedical graph representation learning and link prediction research.
                It is <strong>NOT</strong> a certified diagnostic tool, clinical decision support system, or medical treatment recommendation service.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_entity_pill(name: str, entity_type: str, source_id: Optional[str] = None) -> str:
    """Return an HTML snippet for a sleek colored entity badge."""
    c_info = COLOR_MAP.get(entity_type.lower(), COLOR_MAP["slate"])
    source_html = f' <span style="opacity: 0.7; font-size: 0.72rem;">({source_id})</span>' if source_id and source_id != "nan" else ""
    return (
        f'<span style="background-color: {c_info["bg"]}; color: {c_info["text"]}; '
        f'border: 1px solid {c_info["accent"]}44; font-size: 0.78rem; font-weight: 600; '
        f'padding: 3px 10px; border-radius: 6px; display: inline-flex; align-items: center; margin: 2px;">'
        f'{name}{source_html}'
        f'</span>'
    )


def render_breadcrumb(category: str, page_name: str) -> None:
    """Render a sleek SaaS breadcrumb header (e.g., MediGraph AI > Overview > Executive Dashboard)."""
    st.markdown(
        f"""
        <div style="color: #64748B; font-size: 0.82rem; font-weight: 500; margin-bottom: 8px; display: flex; align-items: center; gap: 6px;">
            <span>MediGraph AI</span>
            <span style="color: #CBD5E1;">›</span>
            <span>{category}</span>
            <span style="color: #CBD5E1;">›</span>
            <span style="color: #0D9488; font-weight: 600;">{page_name}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dual_info_cards(
    model_rows: List[tuple[str, str]],
    dataset_rows: List[tuple[str, str]],
) -> None:
    """Render side-by-side Model Information and Dataset Information cards (PRISM style)."""
    col1, col2 = st.columns(2)
    with col1:
        rows_html = "".join([
            f'<div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #F1F5F9; font-size: 0.84rem;">'
            f'<span style="color: #64748B;">{k}</span>'
            f'<strong style="color: #0F172A;">{v}</strong>'
            f'</div>'
            for k, v in model_rows
        ])
        st.markdown(
            f"""
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 18px 22px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                    <span style="font-size: 1.1rem;">🧠</span>
                    <h4 style="margin: 0; color: #0F172A; font-size: 0.96rem; font-weight: 700;">Model Information</h4>
                </div>
                {rows_html}
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        rows_html = "".join([
            f'<div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #F1F5F9; font-size: 0.84rem;">'
            f'<span style="color: #64748B;">{k}</span>'
            f'<strong style="color: #0F172A;">{v}</strong>'
            f'</div>'
            for k, v in dataset_rows
        ])
        st.markdown(
            f"""
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 18px 22px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                    <span style="font-size: 1.1rem;">🌐</span>
                    <h4 style="margin: 0; color: #0F172A; font-size: 0.96rem; font-weight: 700;">Dataset Information</h4>
                </div>
                {rows_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

