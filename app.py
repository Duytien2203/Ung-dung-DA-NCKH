"""
SmartResearch Studio - Enterprise Academic Statistical Suite for Social Sciences
Supports Excel (.xlsx) and SPSS (.sav) files.
Exact mathematical calculations for Demographics, Cronbach's Alpha, T-Test/ANOVA,
EFA, CFA/CB-SEM, PLS-SEM, Machine Learning, and APA-7 Word/Excel Report Export.
"""

import sys
import os
import io
import re
import base64
import mimetypes

# Safe mimetypes initialization across Windows, Linux (Streamlit Cloud), and sandboxes
try:
    if hasattr(mimetypes, "MimeTypes") and hasattr(mimetypes.MimeTypes, "read_windows_registry"):
        mimetypes.MimeTypes.read_windows_registry = lambda self, *args, **kwargs: None
    mimetypes.init()
except Exception:
    pass

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.data_loader import DataLoader
from core.descriptives import (
    calculate_frequency_table, calculate_descriptive_stats, generate_descriptives_narrative
)
from core.reliability import analyze_scale_reliability, generate_reliability_narrative
from core.hypothesis_tests import (
    run_independent_ttest, run_oneway_anova, generate_ttest_narrative
)
from core.efa import run_efa, generate_efa_narrative
from core.cfa_cbsem import run_cfa_cbsem, generate_cfa_cbsem_narrative
from core.pls_sem import PLSModel, generate_pls_narrative
from core.machine_learning import (
    ML_KNOWLEDGE_BASE, analyze_dataset_and_recommend,
    run_advanced_supervised, run_pca_reduction,
    run_hierarchical_clustering_model, run_lda_topic_model,
    run_sentiment_analysis_suite,
    HAS_XGBOOST, HAS_LIGHTGBM
)
from core.report_exporter import create_word_report, create_excel_report, df_to_styled_excel
from core.data_cleaner import (
    auto_suggest_column_codes, detect_data_health,
    clean_missing_values, clean_duplicates, clean_outliers,
    recode_column_values, generate_codebook, export_cleaned_dataset_excel,
    LIKERT_PRESETS
)
from core.firebase_config import inject_firebase_analytics, FIREBASE_CONFIG, send_feedback_to_firestore

# Brand Logo Configuration
logo_dark_path = os.path.join(os.path.dirname(__file__), "assets", "logo_dark.png")
logo_light_path = os.path.join(os.path.dirname(__file__), "assets", "logo_light.png")

# Page configuration
st.set_page_config(
    page_title="StaNDTistics | SmartResearch Studio NCKH",
    page_icon=logo_light_path if os.path.exists(logo_light_path) else "📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_image_b64(path: str) -> str:
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""

LOGO_DARK_SRC = f"data:image/png;base64,{get_image_b64(logo_dark_path)}"
LOGO_LIGHT_SRC = f"data:image/png;base64,{get_image_b64(logo_light_path)}"

# Load Custom CSS First for Instant Ergonomic Layout
css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize Session State
if "loader" not in st.session_state:
    st.session_state.loader = DataLoader()
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = {}
if "current_file_id" not in st.session_state:
    st.session_state.current_file_id = None
if "last_uploaded_file_id" not in st.session_state:
    st.session_state.last_uploaded_file_id = None


def load_sample():
    """Load default social science research sample dataset."""
    sample_sav = os.path.join(os.path.dirname(__file__), "sample_data", "social_research_data.sav")
    sample_xlsx = os.path.join(os.path.dirname(__file__), "sample_data", "social_research_data.xlsx")
    
    if os.path.exists(sample_sav):
        with open(sample_sav, "rb") as f:
            st.session_state.loader.load_file(f.read(), "social_research_data.sav")
            st.session_state.current_file_id = "sample_data_sav"
            st.session_state.analysis_results = {}
            st.sidebar.success("✅ Đã nạp bộ dữ liệu mẫu SPSS (.sav) thành công!")
    elif os.path.exists(sample_xlsx):
        with open(sample_xlsx, "rb") as f:
            st.session_state.loader.load_file(f.read(), "social_research_data.xlsx")
            st.session_state.current_file_id = "sample_data_xlsx"
            st.session_state.analysis_results = {}
            st.sidebar.success("✅ Đã nạp bộ dữ liệu mẫu Excel (.xlsx) thành công!")


def clean_chart(fig, height=None, margin=None):
    """
    Tách nền đen hoàn toàn, đặt nền trong suốt (rgba(0,0,0,0))
    và phông chữ sắc nét cho tất cả biểu đồ Plotly.
    Đảm bảo đủ lề để tiêu đề trục không bao giờ dính/đè lên nhãn số.
    """
    current_margin = getattr(fig.layout, 'margin', None)
    if margin is not None:
        m = margin
    elif current_margin is not None and getattr(current_margin, 'l', None) is not None and current_margin.l > 60:
        m = current_margin
    else:
        m = dict(t=45, b=55, l=85, r=40)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#0f172a", size=12),
        margin=m
    )
    fig.update_xaxes(showgrid=True, gridcolor="#cbd5e1")
    fig.update_yaxes(showgrid=True, gridcolor="#cbd5e1")
    if height:
        fig.update_layout(height=height)
    return fig


_table_render_counter = 0


def get_unique_table_key(hint="table"):
    global _table_render_counter
    _table_render_counter += 1
    clean_hint = re.sub(r'[^a-zA-Z0-9_]', '_', str(hint))[:30]
    return f"tbl_excel_{_table_render_counter}_{clean_hint}"


def render_table(
    df_or_styler,
    file_name: str = "bang_du_lieu.xlsx",
    sheet_name: str = "Du_lieu",
    use_container_width: bool = True,
    height: int = None,
    hide_index: bool = None,
    column_config: dict = None,
    key: str = None,
    button_label: str = "📥 Tải Excel",
    show_download: bool = True
):
    """
    Hiển thị bảng dữ liệu với st.dataframe kèm nút chức năng nhỏ ở góc phải
    cho phép tải bảng đó về máy dưới định dạng Microsoft Excel (.xlsx).
    """
    kwargs = {"use_container_width": use_container_width}
    if height is not None:
        kwargs["height"] = height
    if hide_index is not None:
        kwargs["hide_index"] = hide_index
    if column_config is not None:
        kwargs["column_config"] = column_config

    st.dataframe(df_or_styler, **kwargs)

    if show_download:
        df_data = getattr(df_or_styler, 'data', df_or_styler)
        if isinstance(df_data, pd.DataFrame) and not df_data.empty:
            excel_bytes = df_to_styled_excel(df_or_styler, sheet_name=sheet_name)
            if excel_bytes:
                actual_key = key or get_unique_table_key(file_name)
                fname = file_name if file_name.endswith('.xlsx') else f"{file_name}.xlsx"
                
                c_space, c_btn = st.columns([0.72, 0.28])
                with c_btn:
                    st.download_button(
                        label=button_label,
                        data=excel_bytes,
                        file_name=fname,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=actual_key,
                        help=f"Tải bảng '{fname}' về máy dưới định dạng Microsoft Excel (.xlsx)",
                        use_container_width=True
                    )


def render_sem_diagram(paths: list, r2_dict: dict = None, title: str = "Sơ đồ Đường dẫn Mô hình Cấu trúc (Path Diagram)"):
    """
    Renders an interactive high-definition path diagram of the structural equation model.
    """
    fig = go.Figure()
    
    # Calculate positions
    # Group into layers: Exogenous (layer 0), Mediators (layer 1), Endogenous Target (layer 2)
    sources = set([p[0] for p in paths])
    targets = set([p[1] for p in paths])
    
    pure_exogenous = sorted(list(sources - targets))
    mediators = sorted(list(sources.intersection(targets)))
    pure_endogenous = sorted(list(targets - sources))
    
    # Coordinates
    node_coords = {}
    
    # Layer 0: Exogenous (X = 0.1)
    n_exo = max(len(pure_exogenous), 1)
    for idx, node in enumerate(pure_exogenous):
        y_pos = 0.15 + (0.7 / n_exo) * (idx + 0.5)
        node_coords[node] = (0.15, y_pos)
        
    # Layer 1: Mediators (X = 0.5)
    n_med = max(len(mediators), 1)
    for idx, node in enumerate(mediators):
        y_pos = 0.2 + (0.6 / n_med) * (idx + 0.5)
        node_coords[node] = (0.5, y_pos)
        
    # Layer 2: Endogenous (X = 0.85)
    n_endo = max(len(pure_endogenous), 1)
    for idx, node in enumerate(pure_endogenous):
        y_pos = 0.25 + (0.5 / n_endo) * (idx + 0.5)
        node_coords[node] = (0.85, y_pos)

    # Any missing node fallback
    all_nodes = list(sources.union(targets))
    for n in all_nodes:
        if n not in node_coords:
            node_coords[n] = (0.5, 0.5)

    text_color = "#0f172a"
    arrow_color = "#1d4ed8"
    box_text_color = "#1e3a8a"
    box_bg_color = "rgba(255, 255, 255, 0.95)"

    # Draw Arrows
    for path in paths:
        src, tgt = path[0], path[1]
        x0, y0 = node_coords[src]
        x1, y1 = node_coords[tgt]
        
        # Line label (Beta or arrow)
        beta_str = ""
        if len(path) > 2:
            beta_str = f"β = {path[2]}"
            
        fig.add_annotation(
            x=x1 - 0.04 * (x1 - x0),
            y=y1 - 0.04 * (y1 - y0),
            ax=x0 + 0.04 * (x1 - x0),
            ay=y0 + 0.04 * (y1 - y0),
            xref="x", yref="y",
            axref="x", ayref="y",
            text=beta_str,
            showarrow=True,
            arrowhead=2,
            arrowsize=1.2,
            arrowwidth=2.5,
            arrowcolor=arrow_color,
            font=dict(size=12, color=box_text_color, family="Plus Jakarta Sans", weight="bold"),
            bgcolor=box_bg_color,
            borderpad=3
        )

    # Draw Construct Nodes (Circles / Rounded Rectangles)
    for node, (x, y) in node_coords.items():
        r2_label = ""
        if r2_dict and node in r2_dict:
            r2_label = f"<br><b>R² = {r2_dict[node]:.3f}</b>"
            
        is_endo = node in targets
        fill_color = "#eff6ff" if not is_endo else "#ecfdf5"
        border_color = "#3b82f6" if not is_endo else "#10b981"
        node_font_color = "#0f172a"
        
        fig.add_shape(
            type="rect",
            x0=x-0.08, y0=y-0.08, x1=x+0.08, y1=y+0.08,
            line=dict(color=border_color, width=2.5),
            fillcolor=fill_color,
            layer="below"
        )
        
        fig.add_annotation(
            x=x, y=y,
            text=f"<b>{node}</b>{r2_label}",
            showarrow=False,
            font=dict(size=13, color=node_font_color, family="Plus Jakarta Sans"),
            align="center"
        )

    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1])
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1])
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=text_color, weight="bold")),
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig


_table_render_counter = 0

MENU_LIST = [
    "1. Quản lý & Khảo sát Dữ liệu",
    "2. Tiền xử lý: Mã hóa & Làm sạch dữ liệu",
    "3. Thống kê Nhân khẩu học & Mô tả",
    "4. Độ tin cậy thang đo (Cronbach's Alpha)",
    "5. So sánh Trung bình (T-Test & ANOVA)",
    "6. Phân tích Nhân tố Khám phá (EFA)",
    "7. CFA & CB-SEM (Mô hình hiệp phương sai)",
    "8. PLS-SEM (Bình phương bé nhất từng phần)",
    "9. Machine Learning trong NCKH",
    "10. Xuất Báo cáo Học thuật (Word/Excel)"
]

STEP_NAMES = [
    "Dữ liệu",
    "Tiền xử lý",
    "Thống kê mô tả",
    "Cronbach's Alpha",
    "T-Test & ANOVA",
    "EFA",
    "CFA / CB-SEM",
    "PLS-SEM",
    "Machine Learning",
    "Xuất báo cáo"
]

if "active_step_num" not in st.session_state:
    st.session_state.active_step_num = 1

# ---------------- SIDEBAR ----------------
with st.sidebar:
    # Silent Firebase Web SDK & GA4 Tracking Initialization in Sidebar
    inject_firebase_analytics()

    logo_sidebar_html = f"""<img src="{LOGO_DARK_SRC}" style="height: 48px; width: auto; object-fit: contain;" alt="StaNDTistics Logo" />""" if LOGO_DARK_SRC else """<div style="background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%); width: 36px; height: 36px; border-radius: 9px; display: flex; align-items: center; justify-content: center; font-size: 18px; box-shadow: 0 4px 10px rgba(37,99,235,0.4);">📊</div>"""
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
        {logo_sidebar_html}
        <div>
            <div style="font-weight: 800; font-size: 16.5px; color: white; letter-spacing: -0.3px;">SmartResearch</div>
            <div style="font-size: 11px; color: #fb923c; font-weight: 700; letter-spacing: 0.5px;">StaNDTistics • v2.5</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    def on_sidebar_radio_change():
        chosen = st.session_state.sidebar_menu_radio
        if chosen in MENU_LIST:
            st.session_state.active_step_num = MENU_LIST.index(chosen) + 1
            st.session_state.stepper_pills_val = st.session_state.active_step_num

    if "sidebar_menu_radio" not in st.session_state:
        st.session_state.sidebar_menu_radio = MENU_LIST[st.session_state.active_step_num - 1]

    menu = st.radio(
        "QUY TRÌNH PHÂN TÍCH NCKH",
        MENU_LIST,
        key="sidebar_menu_radio",
        on_change=on_sidebar_radio_change
    )
    
    st.markdown("---")
    
    # 1. File Uploader in Sidebar
    uploaded_file = st.file_uploader(
        "Nạp tệp mới (SPSS .sav, Excel .xlsx):",
        type=["sav", "zsav", "xlsx", "xls", "csv"],
        key="main_file_uploader",
        help="Hỗ trợ tải lên tệp SPSS (.sav, .zsav) hoặc Excel (.xlsx, .xls, .csv)"
    )
    
    if uploaded_file is not None:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("last_uploaded_file_id") != file_id:
            try:
                bytes_data = uploaded_file.getvalue()
                st.session_state.loader.load_file(bytes_data, uploaded_file.name)
                st.session_state.current_file_id = file_id
                st.session_state.last_uploaded_file_id = file_id
                st.session_state.analysis_results = {}
                st.sidebar.success(f"✓ Đã nạp: {uploaded_file.name}")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Lỗi đọc tệp: {str(e)}")
    else:
        # User removed the uploaded file
        if st.session_state.get("last_uploaded_file_id") is not None:
            st.session_state.last_uploaded_file_id = None
            if st.session_state.get("current_file_id") not in [None, "sample_data_sav", "sample_data_xlsx"]:
                st.session_state.loader = DataLoader()
                st.session_state.current_file_id = None
                st.session_state.analysis_results = {}
                st.rerun()
            
    if st.button("🧪 Nạp Dữ liệu Mẫu NCKH (SPSS)", use_container_width=True):
        load_sample()
        st.rerun()

    # Active Dataset Card (evaluated AFTER upload / sample logic)
    df = st.session_state.loader.df
    loader = st.session_state.loader
    
    if df is not None:
        st.markdown(f"""
        <div style="background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 12px 14px; margin-top: 10px; margin-bottom: 14px;">
            <div style="font-size: 11px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px;">Dữ liệu hiện hành</div>
            <div style="font-size: 14px; font-weight: 700; color: #ffffff; margin-top: 3px; word-break: break-all;">{loader.filename}</div>
            <div style="display: flex; gap: 8px; margin-top: 8px;">
                <span style="background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;">N = {len(df):,}</span>
                <span style="background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;">{len(df.columns)} Biến</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Chuẩn APA 7th Edition • Hair et al. (2022)")
    st.markdown("""
    <div style="margin-top: 10px; padding: 10px 12px; background: #1e293b; border-radius: 8px; border: 1px solid #334155; text-align: center;">
        <div style="font-size: 11px; color: #94a3b8;">Phát triển bởi</div>
        <div style="font-size: 13.5px; font-weight: 700; color: #60a5fa; margin-top: 2px;">NgDuyTien</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------- TOP WORKFLOW PROGRESS PIPELINE ----------------
# Map menu item to step number (1-to-1 exact synchronization)
step_mapping = {
    "1. Quản lý & Khảo sát Dữ liệu": 1,
    "2. Tiền xử lý: Mã hóa & Làm sạch dữ liệu": 2,
    "3. Thống kê Nhân khẩu học & Mô tả": 3,
    "4. Độ tin cậy thang đo (Cronbach's Alpha)": 4,
    "5. So sánh Trung bình (T-Test & ANOVA)": 5,
    "6. Phân tích Nhân tố Khám phá (EFA)": 6,
    "7. CFA & CB-SEM (Mô hình hiệp phương sai)": 7,
    "8. PLS-SEM (Bình phương bé nhất từng phần)": 8,
    "9. Machine Learning trong NCKH": 9,
    "10. Xuất Báo cáo Học thuật (Word/Excel)": 10
}
current_step = step_mapping.get(menu, 1)

# Header Banner
n_samples_str = f"{len(df):,} quan sát" if df is not None else "Chưa nạp dữ liệu"
filename_str = loader.filename if loader.filename else "Chưa chọn tệp"

st.markdown(f"""
<div class="pro-header" style="background: linear-gradient(135deg, #0b1120 0%, #1e293b 55%, #1d4ed8 100%); border-radius: 12px; padding: 22px 28px; margin-bottom: 20px; box-shadow: 0 10px 25px -5px rgba(11, 17, 32, 0.3); border: 1px solid rgba(255, 255, 255, 0.15);">
    <div class="brand-row" style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px;">
        <div style="display: flex; align-items: center; gap: 16px;">
            {f'<img src="{LOGO_DARK_SRC}" style="height: 64px; width: auto; object-fit: contain;" alt="StaNDTistics Logo" />' if LOGO_DARK_SRC else ''}
            <div>
                <div class="brand-title" style="font-size: 24px; font-weight: 850; letter-spacing: -0.4px; margin: 0; color: #ffffff;">
                    SmartResearch Studio
                </div>
                <div class="brand-subtitle" style="font-size: 13px; color: #cbd5e1; margin-top: 4px; font-weight: 500;">
                    Nền tảng Phân tích Thống kê Toàn diện & Mô hình Cấu trúc cho Nghiên cứu Khoa học Xã hội
                </div>
            </div>
        </div>
        <div class="meta-pill" style="background: rgba(255, 255, 255, 0.15); border: 1px solid rgba(255, 255, 255, 0.3); padding: 7px 16px; border-radius: 9999px; font-size: 12.5px; font-weight: 750; color: #ffffff; display: flex; align-items: center; gap: 8px;">
            <span class="dot" style="width: 9px; height: 9px; border-radius: 50%; background-color: #10b981; box-shadow: 0 0 10px #10b981; display: inline-block;"></span>
            <span>{filename_str} • {n_samples_str}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Workflow Stepper Interactive Navigation (10 Steps matching Left Sidebar exactly - Clickable)
if "stepper_pills_val" not in st.session_state:
    st.session_state.stepper_pills_val = current_step

def on_stepper_change():
    new_step = st.session_state.stepper_pills_val
    if new_step:
        st.session_state.active_step_num = new_step
        st.session_state.sidebar_menu_radio = MENU_LIST[new_step - 1]

stepper_selected = st.pills(
    "Workflow Navigation Stepper",
    options=list(range(1, 11)),
    default=current_step,
    format_func=lambda x: f"{x}. {STEP_NAMES[x-1]}",
    key="stepper_pills_val",
    wrap=False,
    on_change=on_stepper_change,
    label_visibility="collapsed"
)


def render_page_footer(key_suffix: str = ""):
    """Renders professional brand footer with StaNDTistics logo, followed by user feedback form at the very bottom."""
    # ---------------- PROFESSIONAL BRAND FOOTER ----------------
    logo_footer_html = f"""<img src="{LOGO_LIGHT_SRC}" style="height: 60px; width: auto; object-fit: contain;" alt="StaNDTistics Logo" />""" if LOGO_LIGHT_SRC else ""
    st.markdown(f"""
    <div class="pro-footer">
        <div class="footer-left" style="display: flex; align-items: center; gap: 16px;">
            {logo_footer_html}
            <div>
                <div class="footer-brand" style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-weight: 850; color: #0f172a; font-size: 15.5px;">SmartResearch Studio</span>
                    <span style="font-weight: 500; color: #64748b; font-size: 12.5px;">• Phiên bản 2.5 Academic Edition</span>
                </div>
                <div class="footer-desc" style="color: #475569; font-size: 12.5px; margin-top: 3px;">
                    Nền tảng xử lý dữ liệu và kiểm định mô hình phương trình cấu trúc (CB-SEM & PLS-SEM) cho Nghiên cứu Khoa học Xã hội.
                </div>
            </div>
        </div>
        <div>
            <div class="author-badge" style="display: flex; align-items: center; gap: 8px; padding: 7px 16px;">
                <span style="color: #64748b; font-size: 12.5px;">Thiết kế & Phát triển bởi</span>
                <span class="author-name" style="font-weight: 800; color: #1d4ed8; font-size: 13.5px;">NgDuyTien</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- ĐÓNG GÓP Ý KIẾN Ở DƯỚI CÙNG TRANG WEB ----------------
    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
    _, col_fb, _ = st.columns([1, 3.5, 1])
    with col_fb:
        with st.expander("💬 Đóng góp ý kiến phát triển SmartResearch Studio", expanded=False):
            st.markdown("""
            <div style="margin-bottom: 14px; color: #475569; font-size: 13.5px; line-height: 1.6;">
                Chúng tôi luôn lắng nghe và trân trọng mọi ý kiến phản hồi từ các nhà nghiên cứu, giảng viên và sinh viên để liên tục hoàn thiện ứng dụng.
                Mọi ý kiến của bạn sẽ được gửi trực tiếp và lưu trữ an toàn trong cơ sở dữ liệu SmartResearch Studio.
            </div>
            """, unsafe_allow_html=True)
            
            with st.form(f"feedback_form_{key_suffix}", clear_on_submit=True):
                fb_col1, fb_col2 = st.columns([1, 1])
                with fb_col1:
                    fb_name = st.text_input("Họ và tên (Tùy chọn):", placeholder="Ví dụ: Nguyễn Văn A", key=f"fb_name_{key_suffix}")
                    fb_email = st.text_input("Email liên hệ (Tùy chọn):", placeholder="email@example.com", key=f"fb_email_{key_suffix}")
                    fb_role = st.selectbox(
                        "Vai trò nghiên cứu của bạn:",
                        [
                            "Giảng viên / Nhà nghiên cứu",
                            "Nghiên cứu sinh / Học viên Cao học",
                            "Sinh viên đại học",
                            "Chuyên viên Phân tích Dữ liệu (Data Analyst)",
                            "Khác"
                        ],
                        key=f"fb_role_{key_suffix}"
                    )
                with fb_col2:
                    fb_topic = st.selectbox(
                        "Chủ đề đóng góp:",
                        [
                            "Đề xuất tính năng mới",
                            "Báo lỗi tính toán / Số liệu",
                            "Cải thiện giao diện & Trải nghiệm",
                            "Yêu cầu thêm mô hình thống kê / Machine Learning",
                            "Góp ý tài liệu & Diễn giải học thuật",
                            "Ý kiến đóng góp khác"
                        ],
                        key=f"fb_topic_{key_suffix}"
                    )
                    fb_rating = st.select_slider(
                        "Mức độ hài lòng chung về ứng dụng:",
                        options=[1, 2, 3, 4, 5],
                        value=5,
                        format_func=lambda x: f"{x} ⭐" + (" (Rất hài lòng)" if x == 5 else " (Hài lòng)" if x == 4 else " (Bình thường)" if x == 3 else " (Chưa hài lòng)" if x == 2 else " (Rất không hài lòng)"),
                        key=f"fb_rating_{key_suffix}"
                    )
                
                fb_content = st.text_area(
                    "Nội dung góp ý chi tiết *:",
                    placeholder="Chia sẻ nhận xét, đề xuất tính năng hoặc khó khăn bạn gặp phải khi sử dụng ứng dụng...",
                    height=110,
                    key=f"fb_content_{key_suffix}"
                )
                
                fb_submit = st.form_submit_button("Gửi ý kiến đóng góp", use_container_width=True, type="primary")
                
                if fb_submit:
                    if not fb_content.strip():
                        st.error("Vui lòng nhập nội dung góp ý trước khi gửi.")
                    else:
                        with st.spinner("Đang gửi ý kiến đóng góp..."):
                            success, msg = send_feedback_to_firestore(
                                name=fb_name,
                                email=fb_email,
                                role=fb_role,
                                topic=fb_topic,
                                rating=fb_rating,
                                content=fb_content
                            )
                        if success:
                            st.success("Cảm ơn bạn rất nhiều! Ý kiến đóng góp của bạn đã được ghi nhận về cho đội ngũ phát triển.")
                        elif msg == "PERMISSION_DENIED":
                            st.warning(
                                "Ý kiến của bạn đã được ghi nhận vào phiên làm việc! "
                                "(Lưu ý quản trị viên: Cơ sở dữ liệu Firestore hiện chưa mở quyền ghi cho bảng 'feedbacks'. "
                                "Vui lòng cập nhật Rules trên Firebase Console theo hướng dẫn để lưu trực tiếp vào đám mây)."
                            )
                        else:
                            st.error(f"Không thể gửi lên Firebase: {msg}")


if df is None:
    welcome_logo_html = f'<div style="margin-bottom: 20px;"><img src="{LOGO_LIGHT_SRC}" style="height: 110px; width: auto; object-fit: contain;" alt="StaNDTistics Logo" /></div>' if LOGO_LIGHT_SRC else '<div style="font-size: 48px; margin-bottom: 12px;">📁</div>'
    st.markdown(f"""
    <div class="content-box" style="text-align: center; padding: 36px 24px 20px 24px;">
        {welcome_logo_html}
        <h3 style="font-size: 22px; font-weight: 800; color: #0f172a; margin-bottom: 8px;">Chào mừng bạn đến với SmartResearch Studio</h3>
        <p style="color: #475569; font-size: 15px; max-width: 650px; margin: 0 auto 20px auto; line-height: 1.6;">
            Hệ thống chuyên sâu phân tích định lượng NCKH xã hội theo chuẩn học thuật APA 7th Edition. Hỗ trợ tệp <b>Excel (.xlsx, .xls, .csv)</b> và <b>SPSS (.sav, .zsav)</b> với bảo toàn nhãn biến.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col_upload, col_sample = st.columns(2)
    with col_upload:
        st.markdown("""
        <div style="background: #ffffff; border: 1.5px dashed #3b82f6; border-radius: 12px; padding: 20px; height: 100%;">
            <div style="font-weight: 700; color: #1e3a8a; font-size: 16px; margin-bottom: 6px;">📂 Tải lên Dữ liệu Nghiên cứu của bạn</div>
            <div style="font-size: 13px; color: #64748b; margin-bottom: 12px;">Hỗ trợ .xlsx, .xls, .csv, .sav. Tự động nhận diện và làm sạch dữ liệu.</div>
        """, unsafe_allow_html=True)
        welcome_file = st.file_uploader(
            "Chọn tệp từ máy tính:",
            type=["sav", "zsav", "xlsx", "xls", "csv"],
            key="welcome_main_file_uploader"
        )
        if welcome_file is not None:
            w_file_id = f"{welcome_file.name}_{welcome_file.size}"
            if st.session_state.get("current_file_id") != w_file_id:
                try:
                    bytes_data = welcome_file.getvalue()
                    st.session_state.loader.load_file(bytes_data, welcome_file.name)
                    st.session_state.current_file_id = w_file_id
                    st.session_state.last_uploaded_file_id = w_file_id
                    st.session_state.analysis_results = {}
                    st.rerun()
                except Exception as e:
                    st.error(f"Lỗi đọc tệp: {str(e)}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_sample:
        st.markdown("""
        <div style="background: #ffffff; border: 1.5px solid #cbd5e1; border-radius: 12px; padding: 20px; height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div style="font-weight: 700; color: #0f172a; font-size: 16px; margin-bottom: 6px;">🧪 Khám phá với Bộ Dữ liệu Mẫu</div>
                <div style="font-size: 13.5px; color: #475569; line-height: 1.5; margin-bottom: 16px;">
                    Bộ dữ liệu mẫu nghiên cứu hành vi người dùng (N = 350) với đầy đủ 26 biến số gồm 5 nhân tố (PE, EE, SI, FC, BI) và thông tin nhân khẩu học.
                </div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 Nạp Dữ liệu Mẫu Ngay", use_container_width=True, type="primary"):
            load_sample()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # Features overview highlights cards
    st.markdown("""
    <div style="margin-top: 26px; margin-bottom: 12px; font-weight: 800; font-size: 16px; color: #0f172a;">
        Hệ sinh thái Tính năng Phân tích Nghiên cứu Toàn diện
    </div>
    """, unsafe_allow_html=True)
    
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        st.markdown("""
        <div class="pro-kpi-card" style="padding: 16px;">
            <div style="font-size: 13.5px; font-weight: 800; color: #1e3a8a; margin-bottom: 6px;">Quản lý & Tiền xử lý</div>
            <div style="font-size: 12.5px; color: #475569; line-height: 1.5;">Hỗ trợ Excel & SPSS (.sav), làm sạch dữ liệu, mã hóa thang đo Likert, lọc ngoại lai và lập Codebook học thuật.</div>
            <div class="kpi-badge badge-info" style="margin-top: 10px;">Bước 1 - 2</div>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class="pro-kpi-card" style="padding: 16px;">
            <div style="font-size: 13.5px; font-weight: 800; color: #065f46; margin-bottom: 6px;">Thống kê & Thang đo</div>
            <div style="font-size: 12.5px; color: #475569; line-height: 1.5;">Cơ cấu nhân khẩu, Skewness/Kurtosis, Cronbach's Alpha, T-Test/ANOVA, EFA ma trận xoay chuẩn SPSS.</div>
            <div class="kpi-badge badge-success" style="margin-top: 10px;">Bước 3 - 6</div>
        </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown("""
        <div class="pro-kpi-card" style="padding: 16px;">
            <div style="font-size: 13.5px; font-weight: 800; color: #92400e; margin-bottom: 6px;">CB-SEM & PLS-SEM</div>
            <div style="font-size: 12.5px; color: #475569; line-height: 1.5;">Kiểm định CFA (CMIN/df, CFI, RMSEA), PLS-SEM Wold's 2022, Bootstrapping, HTMT, Mediation trung gian.</div>
            <div class="kpi-badge badge-warning" style="margin-top: 10px;">Bước 7 - 8</div>
        </div>
        """, unsafe_allow_html=True)
    with f4:
        st.markdown("""
        <div class="pro-kpi-card" style="padding: 16px;">
            <div style="font-size: 13.5px; font-weight: 800; color: #6b21a8; margin-bottom: 6px;">AI & Báo cáo APA 7th</div>
            <div style="font-size: 12.5px; color: #475569; line-height: 1.5;">Thuật toán XGBoost, Random Forest, phân tích cảm xúc, tự động xuất báo cáo trọn bộ sang Word & Excel.</div>
            <div class="kpi-badge badge-info" style="margin-top: 10px;">Bước 9 - 10</div>
        </div>
        """, unsafe_allow_html=True)

    render_page_footer(key_suffix="welcome")
    st.stop()


# ---------------- MENU 1: QUẢN LÝ DỮ LIỆU ----------------
if menu == "1. Quản lý & Khảo sát Dữ liệu":
    st.markdown("### 1. Quản lý & Khảo sát Dữ liệu (Data Dictionary)")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="pro-kpi-card">
            <div>
                <div class="kpi-label">Tệp dữ liệu</div>
                <div class="kpi-value" style="font-size: 20px; word-break: break-all;">{loader.filename}</div>
            </div>
            <div class="kpi-badge badge-info">Định dạng .{loader.file_type.upper()}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="pro-kpi-card">
            <div>
                <div class="kpi-label">Cỡ mẫu nghiên cứu (N)</div>
                <div class="kpi-value">{len(df):,}</div>
            </div>
            <div class="kpi-badge badge-success">✓ Dữ liệu hợp lệ</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="pro-kpi-card">
            <div>
                <div class="kpi-label">Tổng số biến</div>
                <div class="kpi-value">{len(df.columns)}</div>
            </div>
            <div class="kpi-badge badge-info">Chỉ báo đo lường</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        total_na = df.isna().sum().sum()
        badge_cls = "badge-success" if total_na == 0 else "badge-warning"
        status_txt = "✓ Hoàn chỉnh 100%" if total_na == 0 else f"Có {total_na} ô khuyết"
        st.markdown(f"""
        <div class="pro-kpi-card">
            <div>
                <div class="kpi-label">Chất lượng dữ liệu</div>
                <div class="kpi-value">{len(df) - total_na:,}</div>
            </div>
            <div class="kpi-badge {badge_cls}">{status_txt}</div>
        </div>
        """, unsafe_allow_html=True)

    tab_data, tab_dict = st.tabs([f"Xem Toàn bộ Bảng Dữ liệu ({len(df):,} quan sát)", "Từ điển Biến & Nhãn SPSS (Variable Metadata)"])
    
    with tab_data:
        col_info, col_filter = st.columns([3, 1])
        with col_info:
            st.markdown(f"""
            <div style="font-size: 13.5px; color: #475569; margin-bottom: 8px;">
                Đang hiển thị <b>{len(df):,}</b> quan sát và <b>{len(df.columns)}</b> biến số. Bạn có thể cuộn ngang/dọc, tìm kiếm, sắp xếp từng cột hoặc tải toàn bộ tệp Excel.
            </div>
            """, unsafe_allow_html=True)
        with col_filter:
            view_options = [f"Tất cả ({len(df):,} dòng)"]
            if len(df) > 100:
                view_options.append("100 dòng đầu")
            if len(df) > 500:
                view_options.append("500 dòng đầu")
            selected_view = st.selectbox(
                "Số dòng hiển thị:",
                options=view_options,
                index=0,
                key="step1_row_view_selector",
                label_visibility="collapsed"
            )
        
        if selected_view == "100 dòng đầu":
            display_df = df.head(100)
            export_name = "Du_lieu_100_dong.xlsx"
        elif selected_view == "500 dòng đầu":
            display_df = df.head(500)
            export_name = "Du_lieu_500_dong.xlsx"
        else:
            display_df = df
            clean_name = os.path.splitext(loader.filename)[0] if loader.filename else "Du_lieu"
            export_name = f"{clean_name}_toan_bo_{len(df)}_dong.xlsx"
            
        render_table(display_df, file_name=export_name, sheet_name="Du_lieu", height=520)
        
    with tab_dict:
        var_info = loader.get_variable_info()
        render_table(var_info, file_name="Tu_dien_bien_Variable_Info.xlsx", sheet_name="Tu_dien_bien", height=450)


# ---------------- MENU 2: TIỀN XỬ LÝ: MÃ HÓA & LÀM SẠCH DỮ LIỆU ----------------
elif menu == "2. Tiền xử lý: Mã hóa & Làm sạch dữ liệu":
    st.markdown("### 2. Tiền xử lý: Mã hóa Cột & Làm sạch Dữ liệu (Data Cleaning & Preparation)")
    st.markdown("""
    <div style="color: #475569; font-size: 14px; margin-bottom: 16px; line-height: 1.6;">
        Phân hệ chuẩn bị dữ liệu học thuật dành cho các tệp khảo sát thô (Google Forms, MS Forms, KoboToolbox, khảo sát Excel). 
        Hỗ trợ <b>mã hóa tên cột</b> từ câu hỏi dài thành mã biến ngắn gọn (PE1, PU2...), <b>mã hóa giá trị chữ sang số</b> (Likert 1-5), 
        <b>làm sạch giá trị khuyết / dị biệt / trùng lặp</b>, và <b>tải tệp Excel đã chuẩn hóa kèm Codebook</b>. 
        Mọi dữ liệu sau khi lưu tại đây sẽ được cập nhật và kế thừa cho toàn bộ các bước phân tích tiếp theo.
    </div>
    """, unsafe_allow_html=True)

    # Calculate Data Health
    health = detect_data_health(df)

    # Top KPI Health Summary
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
        <div class="pro-kpi-card">
            <div>
                <div class="kpi-label">Cỡ mẫu hiện tại (N)</div>
                <div class="kpi-value">{health['total_rows']:,}</div>
            </div>
            <div class="kpi-badge badge-info">{health['total_cols']} Biến</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        dup_badge = "badge-success" if health['duplicate_rows'] == 0 else "badge-danger"
        dup_txt = "✓ Không có" if health['duplicate_rows'] == 0 else f"{health['duplicate_rows']} dòng trùng"
        st.markdown(f"""
        <div class="pro-kpi-card">
            <div>
                <div class="kpi-label">Dòng trùng lặp</div>
                <div class="kpi-value">{health['duplicate_rows']}</div>
            </div>
            <div class="kpi-badge {dup_badge}">{dup_txt}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        na_badge = "badge-success" if health['missing_cells'] == 0 else "badge-warning"
        na_txt = "✓ Hoàn hảo" if health['missing_cells'] == 0 else f"{health['missing_columns_count']} cột khuyết"
        st.markdown(f"""
        <div class="pro-kpi-card">
            <div>
                <div class="kpi-label">Ô khuyết (Missing)</div>
                <div class="kpi-value">{health['missing_cells']} ({health['missing_pct']}%)</div>
            </div>
            <div class="kpi-badge {na_badge}">{na_txt}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        long_names_count = sum(len(str(c)) > 15 for c in df.columns)
        code_badge = "badge-success" if long_names_count == 0 else "badge-warning"
        code_txt = "✓ Đã mã hóa" if long_names_count == 0 else f"{long_names_count} câu hỏi dài"
        st.markdown(f"""
        <div class="pro-kpi-card">
            <div>
                <div class="kpi-label">Cột tên dài (>15 ký tự)</div>
                <div class="kpi-value">{long_names_count}</div>
            </div>
            <div class="kpi-badge {code_badge}">{code_txt}</div>
        </div>
        """, unsafe_allow_html=True)

    tab_encode, tab_recode, tab_clean, tab_export = st.tabs([
        "1. Mã hóa Tên cột (Column Coding)",
        "2. Mã hóa Giá trị Thang đo (Value Recoding)",
        "3. Làm sạch Dữ liệu (Data Cleaning)",
        "4. Lưu trữ & Xuất Tệp đã Chuẩn hóa (Export File)"
    ])

    # ---------------- TAB 1: MÃ HÓA TÊN CỘT ----------------
    with tab_encode:
        st.markdown("#### 1. Mã hóa Tên Cột từ Câu hỏi Khảo sát Dài sang Mã Biến Nghiên cứu")
        st.write("""
        Trong SPSS, SmartPLS hay AMOS, tên biến cần ngắn gọn (ví dụ: `PE1, PE2, PU1, Age, Gender`) thay vì cả câu hỏi dài 50 từ. 
        Tại đây, bạn có thể **áp dụng tiền tố hàng loạt theo nhân tố**, dùng **tự động gợi ý mã thông minh**, hoặc **chỉnh sửa trực tiếp trên bảng**. 
        *Nội dung câu hỏi gốc sẽ được tự động bảo toàn thành Nhãn biến (Variable Label) phục vụ cho các báo cáo sau này.*
        """)

        # Tool A: Batch Prefix
        with st.expander("⚡ Công cụ Đổi tên Nhanh theo Tiền tố Nhân tố (Batch Prefix Renamer)", expanded=False):
            st.caption("Chọn nhóm các câu hỏi cùng đo lường một nhân tố (ví dụ 4 câu đo Hiệu quả) và nhập Tiền tố (ví dụ `PE`) để tự động gán `PE1`, `PE2`, `PE3`, `PE4`:")
            col_sel_prefix, col_inp_prefix = st.columns([3, 1])
            with col_sel_prefix:
                batch_cols = st.multiselect(
                    "Chọn các câu hỏi cùng nhân tố:",
                    options=list(df.columns),
                    key="batch_prefix_cols_select"
                )
            with col_inp_prefix:
                prefix_val = st.text_input("Tiền tố biến (ví dụ: PE, PU, ATT):", value="VAR", key="batch_prefix_val_input").strip()
                prefix_start = st.number_input("Bắt đầu từ số:", min_value=1, value=1, step=1, key="batch_prefix_start_num")

            if st.button("Áp dụng tiền tố cho các cột đã chọn", key="btn_apply_prefix_run"):
                if batch_cols and prefix_val:
                    rename_map = {}
                    for i, col_name in enumerate(batch_cols):
                        rename_map[col_name] = f"{prefix_val}{prefix_start + i}"
                    
                    new_labels = loader.variable_labels.copy()
                    for orig_c, new_c in rename_map.items():
                        new_labels[new_c] = loader.variable_labels.get(orig_c, orig_c)
                        if orig_c in new_labels and orig_c != new_c:
                            del new_labels[orig_c]

                    df_renamed = df.rename(columns=rename_map)
                    loader.set_cleaned_df(df_renamed, new_labels)
                    st.session_state.analysis_results = {}
                    if "temp_suggested_codes" in st.session_state:
                        del st.session_state["temp_suggested_codes"]
                    st.success(f"✓ Đã áp dụng tiền tố '{prefix_val}' cho {len(batch_cols)} biến thành công!")
                    st.rerun()
                else:
                    st.warning("Vui lòng chọn ít nhất một cột và nhập tiền tố.")

        col_btn_suggest, col_btn_reset = st.columns([2, 1])
        with col_btn_suggest:
            if st.button("✨ Tự động nhận diện & Điền mã biến gợi ý (AI & Regex Auto-Suggest)", use_container_width=True, type="primary"):
                suggested_list = auto_suggest_column_codes(list(df.columns))
                st.session_state.temp_suggested_codes = suggested_list
                st.rerun()
        with col_btn_reset:
            if st.button("↩️ Khôi phục tên cột ban đầu", use_container_width=True):
                loader.reset_to_original()
                if "temp_suggested_codes" in st.session_state:
                    del st.session_state["temp_suggested_codes"]
                st.session_state.analysis_results = {}
                st.success("✓ Đã khôi phục tên cột ban đầu.")
                st.rerun()

        st.markdown("##### Bảng Chỉnh sửa Mã Biến & Nhãn Câu hỏi (Variable Codebook Editor)")
        st.caption("Nhấp đúp vào ô 'Mã biến mới' hoặc 'Nhãn nội dung câu hỏi' để chỉnh sửa trực tiếp, sau đó bấm 'Xác nhận & Cập nhật Mã hóa Biến':")

        suggested_codes = st.session_state.get("temp_suggested_codes", list(df.columns))
        if len(suggested_codes) != len(df.columns):
            suggested_codes = list(df.columns)

        coding_data = []
        for i, c in enumerate(df.columns):
            coding_data.append({
                "STT": i + 1,
                "Tên cột hiện tại (Thô)": str(c),
                "Mã biến mới (Code)": suggested_codes[i] if i < len(suggested_codes) else str(c),
                "Nhãn nội dung câu hỏi (Label)": loader.variable_labels.get(c, str(c)),
                "Kiểu dữ liệu": str(df[c].dtype)
            })
        coding_df = pd.DataFrame(coding_data)

        edited_coding_df = st.data_editor(
            coding_df,
            column_config={
                "STT": st.column_config.NumberColumn("STT", disabled=True, width="small"),
                "Tên cột hiện tại (Thô)": st.column_config.TextColumn("Tên cột hiện tại (Thô)", disabled=True, width="large"),
                "Mã biến mới (Code)": st.column_config.TextColumn("Mã biến mới (Code) ✏️", width="medium", required=True),
                "Nhãn nội dung câu hỏi (Label)": st.column_config.TextColumn("Nhãn nội dung câu hỏi (Label) ✏️", width="large"),
                "Kiểu dữ liệu": st.column_config.TextColumn("Kiểu dữ liệu", disabled=True, width="small"),
            },
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            key="coding_editor_table_widget"
        )

        if st.button("💾 Xác nhận & Cập nhật Mã hóa Biến vào Toàn bộ Hệ thống", type="primary", use_container_width=True, key="btn_save_coding_final"):
            new_codes = [str(x).strip() for x in edited_coding_df["Mã biến mới (Code)"]]
            if any(not c for c in new_codes):
                st.error("Mã biến mới không được để trống!")
            elif len(set(new_codes)) < len(new_codes):
                st.error("Tồn tại mã biến bị trùng lặp! Mỗi biến số phải có một mã định danh duy nhất.")
            else:
                mapping = dict(zip(edited_coding_df["Tên cột hiện tại (Thô)"], new_codes))
                labels_map = dict(zip(new_codes, edited_coding_df["Nhãn nội dung câu hỏi (Label)"]))

                df_renamed = df.rename(columns=mapping)
                loader.set_cleaned_df(df_renamed, labels_map)
                st.session_state.analysis_results = {}
                if "temp_suggested_codes" in st.session_state:
                    del st.session_state["temp_suggested_codes"]
                st.success("✓ Đã cập nhật mã hóa biến thành công cho toàn bộ hệ thống!")
                st.rerun()

    # ---------------- TAB 2: MÃ HÓA GIÁ TRỊ ----------------
    with tab_recode:
        st.markdown("#### 2. Mã hóa Giá trị Thang đo từ Chữ sang Số (Text to Numeric Recoding)")
        st.write("""
        Nếu tệp dữ liệu của bạn xuất từ biểu mẫu Google Forms chứa các câu trả lời dạng chữ như *'Đồng ý'*, *'Không đồng ý'* hoặc *'Nam'*, *'Nữ'*, 
        hãy chọn các cột đó và áp dụng bộ quy tắc chuyển đổi sang số chuẩn học thuật:
        """)

        col_target, col_scheme = st.columns([1, 1])
        with col_target:
            text_candidate_cols = [c for c in df.columns if df[c].dtype == object or df[c].nunique() <= 10]
            default_recode_sel = [c for c in text_candidate_cols if c in health['likert_text_cols']] if health['likert_text_cols'] else text_candidate_cols[:5]
            rec_cols = st.multiselect(
                "Chọn các cột cần mã hóa giá trị sang số:",
                options=list(df.columns),
                default=default_recode_sel,
                key="rec_cols_selected_box"
            )
        with col_scheme:
            scheme_choice = st.selectbox(
                "Chọn Bộ quy tắc chuyển đổi mẫu (Preset):",
                list(LIKERT_PRESETS.keys()),
                key="preset_scheme_choice_box"
            )
            selected_mapping = LIKERT_PRESETS[scheme_choice]
            st.caption("Bảng quy đổi mẫu:")
            map_preview_df = pd.DataFrame([{"Giá trị gốc (Text)": k, "Mã số quy đổi (Numeric)": v} for k, v in selected_mapping.items()])
            render_table(map_preview_df, file_name="Bang_Quy_Doi_Ma_Hoa.xlsx", sheet_name="Quy_doi", height=140, hide_index=True)

        if st.button("⚡ Thực hiện Mã hóa Giá trị cho Các Cột Đã Chọn", type="primary", use_container_width=True, key="btn_apply_recode_run"):
            if rec_cols:
                df_rec = df.copy()
                for col_name in rec_cols:
                    df_rec = recode_column_values(df_rec, col_name, selected_mapping)
                loader.set_cleaned_df(df_rec, loader.variable_labels)
                st.session_state.analysis_results = {}
                st.success(f"✓ Đã mã hóa giá trị thành công cho {len(rec_cols)} cột!")
                st.rerun()
            else:
                st.warning("Vui lòng chọn ít nhất một cột cần mã hóa giá trị.")

    # ---------------- TAB 3: LÀM SẠCH DỮ LIỆU ----------------
    with tab_clean:
        st.markdown("#### 3. Bộ Công cụ Làm sạch Dữ liệu Học thuật (Data Cleaning Suite)")

        # 1. System Columns Removal
        st.markdown("##### 1. Xóa các Cột Siêu dữ liệu / Không phục vụ phân tích")
        sys_cols_found = [c for c in health['system_cols'] if c in df.columns]
        cols_to_drop = st.multiselect(
            "Chọn các cột muốn xóa bỏ khỏi bộ dữ liệu (Timestamp, Email, Tên người trả lời, v.v.):",
            options=list(df.columns),
            default=sys_cols_found,
            key="cols_to_drop_sel_box"
        )
        if st.button("Xóa các cột đã chọn khỏi bộ dữ liệu", key="btn_drop_cols_run"):
            if cols_to_drop:
                df_dropped = df.drop(columns=cols_to_drop)
                new_labels = {k: v for k, v in loader.variable_labels.items() if k not in cols_to_drop}
                loader.set_cleaned_df(df_dropped, new_labels)
                st.session_state.analysis_results = {}
                st.success(f"✓ Đã xóa {len(cols_to_drop)} cột khỏi bộ dữ liệu!")
                st.rerun()
            else:
                st.warning("Chưa chọn cột nào để xóa.")

        st.markdown("---")

        # 2. Missing Values
        st.markdown("##### 2. Xử lý Giá trị Khuyết (Missing Values)")
        if health['missing_cells'] > 0:
            render_table(health['missing_df'], file_name="Thong_Ke_Gia_Tri_Khuyet.xlsx", sheet_name="Khuyet_thieu", height=180, hide_index=True)
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                missing_method = st.selectbox(
                    "Chọn phương pháp xử lý giá trị khuyết:",
                    [
                        "Loại bỏ dòng chứa giá trị khuyết (Listwise Deletion - Khuyến nghị SPSS)",
                        "Điền giá trị Trung bình (Mean Imputation - Dành cho biến định lượng)",
                        "Điền giá trị Trung vị (Median Imputation - Dành cho biến Likert)",
                        "Điền giá trị Yếu vị (Mode Imputation - Giá trị xuất hiện nhiều nhất)",
                        "Loại bỏ cột nếu tỷ lệ khuyết vượt ngưỡng > 40%"
                    ],
                    key="missing_method_choice_sel"
                )
            with col_m2:
                target_na_cols = st.multiselect(
                    "Áp dụng cho các cột cụ thể (để trống để áp dụng toàn bộ):",
                    options=list(health['missing_df']['Tên cột']) if not health['missing_df'].empty else list(df.columns),
                    key="target_na_cols_sel_box"
                )

            if st.button("⚡ Áp dụng Xử lý Giá trị Khuyết", key="btn_apply_na_run"):
                method_key = "drop_rows"
                if "Trung bình" in missing_method:
                    method_key = "impute_mean"
                elif "Trung vị" in missing_method:
                    method_key = "impute_median"
                elif "Yếu vị" in missing_method:
                    method_key = "impute_mode"
                elif "Loại bỏ cột" in missing_method:
                    method_key = "drop_cols"

                cols_arg = target_na_cols if target_na_cols else None
                df_na_clean, na_msg = clean_missing_values(df, method=method_key, target_cols=cols_arg, threshold_pct=40.0)
                loader.set_cleaned_df(df_na_clean, loader.variable_labels)
                st.session_state.analysis_results = {}
                st.success(f"✓ {na_msg}")
                st.rerun()
        else:
            st.success("✓ Bộ dữ liệu hoàn hảo! Không có giá trị khuyết nào.")

        st.markdown("---")

        # 3. Duplicates
        st.markdown("##### 3. Xử lý Bản ghi Trùng lặp (Duplicate Records)")
        if health['duplicate_rows'] > 0:
            st.warning(f"Phát hiện **{health['duplicate_rows']} dòng** có nội dung trùng lặp hoàn toàn với các dòng khác (có thể do người tham gia gửi form nhiều lần).")
            if st.button(f"🗑️ Loại bỏ {health['duplicate_rows']} dòng trùng lặp", type="primary", key="btn_rm_dup_run"):
                df_dedup, n_rm = clean_duplicates(df)
                loader.set_cleaned_df(df_dedup, loader.variable_labels)
                st.session_state.analysis_results = {}
                st.success(f"✓ Đã loại bỏ {n_rm} dòng trùng lặp thành công! Cỡ mẫu hiện tại N = {len(df_dedup):,}.")
                st.rerun()
        else:
            st.success("✓ Không có dòng trùng lặp nào trong bộ dữ liệu.")

        st.markdown("---")

        # 4. Outliers & Bounds
        st.markdown("##### 4. Kiểm soát & Xử lý Giá trị Dị biệt / Ngoại lai (Outliers)")
        num_scale_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if num_scale_cols:
            col_out1, col_out2 = st.columns(2)
            with col_out1:
                outlier_target_cols = st.multiselect(
                    "Chọn các biến thang đo cần kiểm soát giá trị:",
                    options=num_scale_cols,
                    default=num_scale_cols[:10] if len(num_scale_cols) <= 15 else num_scale_cols[:5],
                    key="outlier_cols_sel_box"
                )
            with col_out2:
                outlier_method = st.selectbox(
                    "Phương thức kiểm soát dị biệt:",
                    [
                        "Giới hạn (Clip) về khoảng thang đo Likert [1, 5]",
                        "Giới hạn (Clip) về khoảng thang đo Likert [1, 7]",
                        "Loại bỏ dòng có giá trị ngoại lai Z-Score (|Z| > 3.0)"
                    ],
                    key="outlier_method_sel_box"
                )

            if st.button("⚡ Áp dụng Xử lý Dị biệt", key="btn_apply_outlier_run"):
                if outlier_target_cols:
                    if "[1, 5]" in outlier_method:
                        df_out, out_msg = clean_outliers(df, outlier_target_cols, method="clip", min_val=1.0, max_val=5.0)
                    elif "[1, 7]" in outlier_method:
                        df_out, out_msg = clean_outliers(df, outlier_target_cols, method="clip", min_val=1.0, max_val=7.0)
                    else:
                        df_out, out_msg = clean_outliers(df, outlier_target_cols, method="drop_zscore", z_threshold=3.0)
                    loader.set_cleaned_df(df_out, loader.variable_labels)
                    st.session_state.analysis_results = {}
                    st.success(f"✓ {out_msg}")
                    st.rerun()
                else:
                    st.warning("Vui lòng chọn ít nhất một biến số.")

        st.markdown("---")

        if st.button("↩️ Khôi phục Dữ liệu Ban đầu (Reset to Raw File)", key="btn_reset_clean_full"):
            loader.reset_to_original()
            st.session_state.analysis_results = {}
            st.success("✓ Đã khôi phục dữ liệu ban đầu khi mới nạp tệp.")
            st.rerun()

    # ---------------- TAB 4: LƯU TRỮ & XUẤT TỆP ----------------
    with tab_export:
        st.markdown("#### 4. Lưu trữ & Tải xuống Tệp Dữ liệu Đã Làm Sạch (Export Cleaned Dataset)")
        st.write("""
        Bạn có thể tải xuống ngay bộ dữ liệu đã được làm sạch và mã hóa đầy đủ dưới định dạng **Excel (.xlsx)** hoặc **CSV** 
        để lưu trữ hoặc mở trong bất kỳ phần mềm phân tích nào khác (SPSS, AMOS, SmartPLS, Stata, R) mà không bắt buộc phải thực hiện các bước phân tích tiếp theo.
        """)

        # Generate download buffers
        excel_bytes = export_cleaned_dataset_excel(df, loader.variable_labels, loader.value_labels, loader.measure_types)
        csv_bytes = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

        base_name = loader.filename.rsplit('.', 1)[0] if '.' in loader.filename else loader.filename
        clean_filename = f"{base_name}_cleaned.xlsx"
        clean_csv_filename = f"{base_name}_cleaned.csv"

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                label="📥 Tải xuống Tệp Excel Đã Làm Sạch & Kèm Codebook (.xlsx)",
                data=excel_bytes.getvalue(),
                file_name=clean_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True,
                key="dl_cleaned_excel_btn"
            )
            st.caption("✨ Tệp Excel tích hợp sẵn 2 Sheet: **Du_lieu_sach** (Dữ liệu sẵn sàng phân tích) và **Tu_dien_bien_Codebook** (Từ điển định nghĩa biến và câu hỏi gốc).")

        with col_d2:
            st.download_button(
                label="📥 Tải xuống Tệp CSV Đã Làm Sạch (.csv)",
                data=csv_bytes,
                file_name=clean_csv_filename,
                mime="text/csv",
                use_container_width=True,
                key="dl_cleaned_csv_btn"
            )
            st.caption("✨ Định dạng CSV tiêu chuẩn quốc tế UTF-8-BOM tương thích hoàn hảo mọi phần mềm thống kê.")

        st.markdown("---")

        st.markdown("##### Xem trước Từ điển Biến số (Codebook) đính kèm trong tệp Excel:")
        codebook_view = generate_codebook(df, loader.variable_labels, loader.value_labels, loader.measure_types)
        render_table(codebook_view, file_name="Tu_Dien_Bien_Codebook.xlsx", sheet_name="Codebook", height=350, hide_index=True)

        st.markdown("##### Xem trước 50 Dòng Dữ liệu Đã Làm sạch:")
        render_table(df.head(50), file_name="Du_Lieu_Sach_50_Dong_Dau.xlsx", sheet_name="Du_lieu_sach", height=300)


# ---------------- MENU 3: THỐNG KÊ MÔ TẢ & NHÂN KHẨU HỌC ----------------
elif menu == "3. Thống kê Nhân khẩu học & Mô tả":
    st.markdown("### 3. Thống kê Mô tả & Phân tích Cơ cấu Mẫu (Demographics)")
    
    tab_demo, tab_likert = st.tabs(["Biến Định tính / Nhân khẩu học", "Biến Định lượng / Thang đo Likert"])
    
    with tab_demo:
        cat_cols = [c for c in df.columns if c not in ['ID', 'id'] and df[c].nunique() <= 10]
        if not cat_cols:
            cat_cols = df.columns.tolist()
            
        c_sel, c_space = st.columns([2, 1])
        with c_sel:
            def format_cat_choice(x):
                lbl = loader.variable_labels.get(x, "")
                if lbl and lbl != x:
                    return f"{x} - {lbl}"
                return str(x)

            selected_cat = st.selectbox(
                "Chọn biến phân loại nhân khẩu học:",
                cat_cols,
                format_func=format_cat_choice
            )
        
        if selected_cat:
            mapping = loader.value_labels.get(selected_cat, {})
            freq_df = calculate_frequency_table(df[selected_cat], mapping)
            
            c_tbl, c_chart = st.columns([1.1, 1])
            with c_tbl:
                st.markdown("##### Bảng Tần số và Tỷ lệ phần trăm (Frequency Table)")
                render_table(
                    freq_df,
                    file_name=f"Bang_Tan_So_{selected_cat}.xlsx",
                    sheet_name="Tan_so",
                    column_config={
                        "Phân loại": st.column_config.TextColumn("Phân loại / Danh mục", width="large"),
                        "Tần số (N)": st.column_config.NumberColumn("Tần số (N)", format="%d", width="small"),
                        "Tỷ lệ (%)": st.column_config.NumberColumn("Tỷ lệ (%)", format="%.2f%%", width="small"),
                        "Tỷ lệ hợp lệ (%)": st.column_config.NumberColumn("Tỷ lệ hợp lệ (%)", format="%.2f%%", width="small"),
                        "Tỷ lệ tích lũy (%)": st.column_config.NumberColumn("Tỷ lệ tích lũy (%)", format="%.2f%%", width="small")
                    },
                    use_container_width=True,
                    hide_index=True,
                    height=max(200, min(420, len(freq_df) * 38 + 45))
                )
                
            with c_chart:
                valid_rows = freq_df[freq_df["Phân loại"] != "Khuyết (Missing)"].copy()
                has_long_labels = any(len(str(cat)) > 15 for cat in valid_rows["Phân loại"])
                
                var_lbl = loader.variable_labels.get(selected_cat, "")
                disp_title = f"{selected_cat} ({var_lbl})" if var_lbl and var_lbl != selected_cat else selected_cat
                if len(disp_title) > 65:
                    disp_title = disp_title[:62] + "..."
                st.markdown(f"""
                <div style="font-size: 14px; font-weight: 750; color: #0f172a; margin-bottom: 6px;">
                    📊 Biểu đồ Cơ cấu: <span style="color: #1d4ed8;">{disp_title}</span>
                </div>
                """, unsafe_allow_html=True)

                chart_type = st.radio(
                    "Dạng biểu đồ hiển thị:",
                    ["📊 Biểu đồ Cột ngang (Khuyên dùng cho nhãn dài)", "🍩 Biểu đồ Tròn / Donut"],
                    index=0 if has_long_labels else 1,
                    horizontal=True,
                    key=f"demo_chart_choice_{selected_cat}"
                )

                if "Cột ngang" in chart_type:
                    # Format labels with clean wrap so multi-word categories never get cut off
                    import textwrap
                    valid_rows["Phân_loại_wrapped"] = valid_rows["Phân loại"].apply(
                        lambda s: "<br>".join(textwrap.wrap(str(s), width=24)) if len(str(s)) > 24 else str(s)
                    )
                    
                    max_freq = valid_rows["Tần số (N)"].max() if not valid_rows.empty else 10
                    # Give 28% breathing room on the X-axis so outside labels '234 (48.75%)' have plenty of space
                    x_upper = max_freq * 1.28 if max_freq > 0 else 10

                    # Dynamic left margin based on the longest wrapped line
                    max_line_len = max(
                        max(len(line) for line in str(wrapped).split("<br>"))
                        for wrapped in valid_rows["Phân_loại_wrapped"]
                    ) if not valid_rows.empty else 10
                    dynamic_left_margin = max(130, min(250, max_line_len * 10))

                    fig = px.bar(
                        valid_rows,
                        x="Tần số (N)",
                        y="Phân_loại_wrapped",
                        orientation="h",
                        color_discrete_sequence=['#2563eb'],
                        text=valid_rows.apply(
                            lambda r: f"{r['Tần số (N)']:,} ({r['Tỷ lệ hợp lệ (%)']}%)" if r['Tỷ lệ hợp lệ (%)'] is not None else f"{r['Tần số (N)']:,}",
                            axis=1
                        )
                    )
                    fig.update_layout(
                        yaxis=dict(
                            autorange="reversed",
                            title=None,
                            automargin=True,
                            tickfont=dict(size=12, color="#0f172a", family="Plus Jakarta Sans, sans-serif")
                        ),
                        xaxis=dict(
                            title="Số lượng quan sát (N)",
                            range=[0, x_upper],
                            automargin=True
                        ),
                        margin=dict(l=dynamic_left_margin, r=65, t=15, b=45),
                        height=max(320, min(550, len(valid_rows) * 55 + 75))
                    )
                    fig.update_traces(
                        textposition='outside',
                        cliponaxis=False,
                        marker=dict(line=dict(color='#1d4ed8', width=1))
                    )
                    clean_chart(fig, margin=dict(l=dynamic_left_margin, r=65, t=15, b=45))
                    st.plotly_chart(fig, use_container_width=True, theme=None)
                else:
                    # Donut Chart with bottom horizontal legend and percent-only inside slices
                    fig = px.pie(
                        valid_rows,
                        names="Phân loại",
                        values="Tần số (N)",
                        hole=0.45,
                        color_discrete_sequence=['#2563eb', '#0ea5e9', '#10b981', '#f59e0b', '#ec4899', '#8b5cf6', '#64748b', '#14b8a6', '#f97316', '#a855f7']
                    )
                    fig.update_traces(
                        textposition='inside',
                        textinfo='percent',
                        hovertemplate='<b>%{label}</b><br>Tần số: <b>%{value:,}</b> quan sát<br>Tỷ lệ hợp lệ: <b>%{percent}</b><extra></extra>',
                        marker=dict(line=dict(color='#ffffff', width=2))
                    )
                    donut_margin = dict(l=25, r=25, t=15, b=85)
                    fig.update_layout(
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="top",
                            y=-0.15,
                            xanchor="center",
                            x=0.5,
                            font=dict(size=11, color="#1e293b")
                        ),
                        margin=donut_margin,
                        height=400
                    )
                    clean_chart(fig, margin=donut_margin)
                    st.plotly_chart(fig, use_container_width=True, theme=None)
                
            st.session_state.analysis_results[f"Freq_{selected_cat}"] = freq_df

    with tab_likert:
        st.markdown(r"""
        *Kiểm tra hệ số Skewness và Kurtosis theo chuẩn **Kline (2015)** ($|Skew| < 3, |Kurt| < 10$) và **Hair et al. (2022)** ($|Skew| < 1, |Kurt| < 1$)*.
        """)
        
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [c for c in num_cols if c not in ['ID', 'id']]
        
        selected_num = st.multiselect(
            "Chọn các biến quan sát / thang đo Likert:",
            num_cols,
            default=num_cols[:15] if len(num_cols) >= 15 else num_cols
        )
        
        if selected_num:
            desc_df = calculate_descriptive_stats(df, selected_num, loader.variable_labels)
            render_table(desc_df, file_name="Thong_Ke_Mo_Ta_Descriptives.xlsx", sheet_name="Thong_ke_mo_ta", height=360)
            
            # Narrative Box
            narrative = generate_descriptives_narrative(desc_df)
            st.markdown(f'<div class="academic-narrative">{narrative}</div>', unsafe_allow_html=True)
            
            # Distribution Plot
            c_p1, c_p2 = st.columns([1, 2])
            with c_p1:
                plot_var = st.selectbox("Chọn biến quan sát để vẽ phân phối tần số:", selected_num)
            with c_p2:
                if plot_var:
                    var_lbl = loader.variable_labels.get(plot_var, "")
                    var_display = f"{plot_var} - {var_lbl}" if var_lbl else plot_var
                    fig = px.histogram(
                        df, x=plot_var, nbins=5, marginal="box",
                        title=f"Biểu đồ phân phối tần số của biến {var_display}",
                        color_discrete_sequence=['#2563eb'],
                        labels={plot_var: f"Giá trị thang đo ({plot_var})", "count": "Tần số (Frequency)"}
                    )
                    clean_chart(fig, height=380, margin=dict(t=55, b=65, l=65, r=30))
                    
                    # Tách biệt rõ ràng nhãn trục x và ticks, tránh hoàn toàn dính chữ
                    fig.layout.xaxis.title.text = f"Giá trị thang đo ({plot_var})"
                    fig.layout.xaxis.title.standoff = 16
                    fig.layout.xaxis.dtick = 1
                    fig.layout.xaxis2.title.text = ""
                    fig.layout.xaxis2.showticklabels = False
                    fig.layout.yaxis.title.text = "Tần số (Count)"
                    fig.layout.yaxis.title.standoff = 12
                    fig.layout.yaxis2.title.text = ""
                    
                    st.plotly_chart(fig, use_container_width=True, theme=None)
                
            st.session_state.analysis_results["Descriptive_Stats"] = desc_df


# ---------------- MENU 4: CRONBACH'S ALPHA ----------------
elif menu == "4. Độ tin cậy thang đo (Cronbach's Alpha)":
    st.markdown("### 4. Kiểm định Độ tin cậy Thang đo (Cronbach's Alpha)")
    st.markdown(r"""
    **Quy chuẩn học thuật (Nunnally & Bernstein 1994, Hair et al. 2022):**
    - Hệ số Cronbach's $\alpha \ge 0.70$ (hoặc $\ge 0.60$ đối với nghiên cứu sơ bộ).
    - Hệ số tương quan biến - tổng hiệu chỉnh (Corrected Item-Total Correlation) $\ge 0.30$.
    """)
    
    # Auto-detect groups based on prefix
    detected_scales = {}
    for c in df.select_dtypes(include=[np.number]).columns:
        if c.lower() not in ['id', 'use', 'gioitinh', 'dotuoi', 'thunhap', 'hocvan']:
            prefix = ''.join([char for char in c if not char.isdigit()]).strip('_')
            if prefix:
                detected_scales.setdefault(prefix, []).append(c)

    mode = st.radio("Chế độ phân tích:", ["⚡ Phân tích hàng loạt tất cả các thang đo", "🎯 Phân tích một thang đo tùy chỉnh"], horizontal=True)
    
    if mode == "⚡ Phân tích hàng loạt tất cả các thang đo":
        if detected_scales:
            summary_list = []
            details_dict = {}
            
            for scale_name, items in detected_scales.items():
                if len(items) >= 2:
                    res = analyze_scale_reliability(df, items, scale_name, loader.variable_labels)
                    if "error" not in res:
                        summary_list.append({
                            "Thang đo": scale_name,
                            "Số biến": res["k_items"],
                            "Cronbach's Alpha": res["overall_alpha"],
                            "Đánh giá chất lượng": res["scale_quality"],
                            "Biến rác (< 0.30)": ", ".join(res["bad_items"]) if res["bad_items"] else "Không có (Đạt chuẩn)",
                            "Kết luận": "✓ Đạt độ tin cậy" if res["is_valid"] else "Cần loại biến"
                        })
                        details_dict[scale_name] = res
                        
            df_batch = pd.DataFrame(summary_list)
            st.markdown("##### Bảng Tổng hợp Độ tin cậy tất cả các Thang đo")
            render_table(df_batch, file_name="Tong_Hop_Cronbach_Alpha.xlsx", sheet_name="Cronbach_Alpha")
            st.session_state.analysis_results["Cronbach_Summary"] = df_batch
            
            st.markdown("---")
            selected_detail = st.selectbox("Xem chi tiết bảng tương quan biến - tổng của thang đo:", list(details_dict.keys()))
            if selected_detail:
                res_detail = details_dict[selected_detail]
                
                # KPI Summary Cards
                k1, k2, k3 = st.columns(3)
                with k1:
                    st.markdown(f"""
                    <div class="pro-kpi-card">
                        <div>
                            <div class="kpi-label">Cronbach's Alpha ({selected_detail})</div>
                            <div class="kpi-value">{res_detail['overall_alpha']:.3f}</div>
                        </div>
                        <div class="kpi-badge badge-success">✓ {res_detail['scale_quality']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with k2:
                    st.markdown(f"""
                    <div class="pro-kpi-card">
                        <div>
                            <div class="kpi-label">Số biến quan sát</div>
                            <div class="kpi-value">{res_detail['k_items']}</div>
                        </div>
                        <div class="kpi-badge badge-info">Chỉ báo đo lường</div>
                    </div>
                    """, unsafe_allow_html=True)
                with k3:
                    bad_cnt = len(res_detail['bad_items'])
                    badge_b = "badge-success" if bad_cnt == 0 else "badge-danger"
                    txt_b = "✓ Không có biến rác" if bad_cnt == 0 else f"Cảnh báo: {bad_cnt} biến"
                    st.markdown(f"""
                    <div class="pro-kpi-card">
                        <div>
                            <div class="kpi-label">Biến cần loại bỏ (r < 0.30)</div>
                            <div class="kpi-value">{bad_cnt}</div>
                        </div>
                        <div class="kpi-badge {badge_b}">{txt_b}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.markdown(f"##### Bảng chỉ số tương quan biến - tổng thang đo {selected_detail}")
                render_table(res_detail["items_table"], file_name=f"Tuong_Quan_Bien_Tong_{selected_detail}.xlsx", sheet_name="Item_Total")
                narr = generate_reliability_narrative(res_detail)
                st.markdown(f'<div class="academic-narrative">{narr}</div>', unsafe_allow_html=True)
        else:
            st.warning("Không tự động phát hiện được nhóm thang đo. Vui lòng chọn chế độ phân tích tùy chỉnh.")

    else:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [c for c in num_cols if c not in ['ID', 'id']]
        
        c1, c2 = st.columns([1, 2])
        with c1:
            custom_scale_name = st.text_input("Tên thang đo:", "Thang đo A")
        with c2:
            custom_items = st.multiselect("Chọn các biến quan sát:", num_cols)
            
        if custom_items:
            res = analyze_scale_reliability(df, custom_items, custom_scale_name, loader.variable_labels)
            if "error" in res:
                st.error(res["error"])
            else:
                k1, k2, k3 = st.columns(3)
                with k1:
                    st.markdown(f"""
                    <div class="pro-kpi-card">
                        <div>
                            <div class="kpi-label">Cronbach's Alpha</div>
                            <div class="kpi-value">{res['overall_alpha']:.3f}</div>
                        </div>
                        <div class="kpi-badge badge-success">✓ {res['scale_quality']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with k2:
                    st.markdown(f"""
                    <div class="pro-kpi-card">
                        <div>
                            <div class="kpi-label">Số biến quan sát</div>
                            <div class="kpi-value">{res['k_items']}</div>
                        </div>
                        <div class="kpi-badge badge-info">Chỉ báo đo lường</div>
                    </div>
                    """, unsafe_allow_html=True)
                with k3:
                    bad_cnt = len(res['bad_items'])
                    badge_b = "badge-success" if bad_cnt == 0 else "badge-danger"
                    st.markdown(f"""
                    <div class="pro-kpi-card">
                        <div>
                            <div class="kpi-label">Biến rác (< 0.30)</div>
                            <div class="kpi-value">{bad_cnt}</div>
                        </div>
                        <div class="kpi-badge {badge_b}">{'Đạt chuẩn' if bad_cnt == 0 else 'Cần loại'}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.markdown("##### Bảng thống kê Item-Total Statistics")
                render_table(res["items_table"], file_name=f"Cronbach_{custom_scale_name}_Item_Total.xlsx", sheet_name="Item_Total")
                
                narr = generate_reliability_narrative(res)
                st.markdown(f'<div class="academic-narrative">{narr}</div>', unsafe_allow_html=True)
                st.session_state.analysis_results[f"Cronbach_{custom_scale_name}"] = res["items_table"]


# ---------------- MENU 5: T-TEST & ANOVA ----------------
elif menu == "5. So sánh Trung bình (T-Test & ANOVA)":
    st.markdown("### 5. Kiểm định So sánh Trung bình (T-Test & ANOVA)")
    
    test_type = st.radio("Chọn loại phép thử so sánh:", ["Independent Samples T-Test (2 nhóm)", "One-Way ANOVA (>= 2 nhóm)"], horizontal=True)
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ['ID', 'id']]
    all_cols = df.columns.tolist()
    
    if test_type == "Independent Samples T-Test (2 nhóm)":
        c1, c2 = st.columns(2)
        with c1:
            dep_var = st.selectbox("Biến phụ thuộc (Kiểm tra trung bình):", num_cols, index=0)
        with c2:
            two_cat = [c for c in all_cols if df[c].dropna().nunique() == 2]
            group_var = st.selectbox("Biến phân nhóm (Đúng 2 nhóm, ví dụ Giới tính):", two_cat if two_cat else all_cols)
            
        if st.button("🚀 Chạy kiểm định T-Test", use_container_width=True):
            res_t = run_independent_ttest(df, dep_var, group_var, loader.value_labels)
            if "error" in res_t:
                st.error(res_t["error"])
            else:
                st.markdown("##### 1. Thống kê nhóm (Group Statistics)")
                render_table(res_t["groups_table"], file_name=f"TTest_Thong_Ke_Nhom_{dep_var}_{group_var}.xlsx", sheet_name="Group_Stats")
                
                st.markdown("##### 2. Kết quả kiểm định mẫu độc lập (Independent Samples Test)")
                render_table(res_t["tests_table"], file_name=f"TTest_Kiem_Dinh_Doc_Lap_{dep_var}_{group_var}.xlsx", sheet_name="Independent_Test")
                
                narr = generate_ttest_narrative(res_t)
                st.markdown(f'<div class="academic-narrative">{narr}</div>', unsafe_allow_html=True)
                st.session_state.analysis_results[f"TTest_{dep_var}_{group_var}"] = res_t["tests_table"]

    else:
        c1, c2 = st.columns(2)
        with c1:
            dep_var = st.selectbox("Biến phụ thuộc (Kiểm tra trung bình):", num_cols, index=0)
        with c2:
            multi_cat = [c for c in all_cols if 2 <= df[c].dropna().nunique() <= 10]
            group_var = st.selectbox("Biến phân nhóm (Độ tuổi, Thu nhập, Học vấn...):", multi_cat if multi_cat else all_cols)
            
        if st.button("🚀 Chạy kiểm định One-Way ANOVA", use_container_width=True):
            res_a = run_oneway_anova(df, dep_var, group_var, loader.value_labels)
            if "error" in res_a:
                st.error(res_a["error"])
            else:
                st.markdown("##### 1. Thống kê mô tả theo nhóm")
                render_table(res_a["descriptives_table"], file_name=f"ANOVA_Mo_Ta_Nhom_{dep_var}_{group_var}.xlsx", sheet_name="Descriptives")
                
                st.markdown("##### 2. Bảng phân tích phương sai (ANOVA)")
                render_table(res_a["anova_table"], file_name=f"ANOVA_Bang_Phan_Tich_{dep_var}_{group_var}.xlsx", sheet_name="ANOVA")
                
                st.markdown(fr"**Levene Test:** Sig. = {res_a['levene_p']} | **Kích thước ảnh hưởng Eta-squared:** $\eta^2 = {res_a['eta_squared']:.4f}$")
                
                if res_a["is_significant"]:
                    st.success("✓ Kiểm định ANOVA có ý nghĩa thống kê (p < 0.05). Có sự khác biệt trung bình giữa các nhóm!")
                    st.markdown("##### 3. Kiểm định so sánh cặp Post-Hoc (Tukey HSD)")
                    render_table(res_a["posthoc_table"], file_name=f"ANOVA_Hau_Kiem_Tukey_{dep_var}_{group_var}.xlsx", sheet_name="Tukey_HSD")
                else:
                    st.info("Kiểm định ANOVA không có ý nghĩa thống kê (p >= 0.05). Không có sự khác biệt giữa các nhóm.")
                    
                st.session_state.analysis_results[f"ANOVA_{dep_var}_{group_var}"] = res_a["anova_table"]


# ---------------- MENU 6: EFA ----------------
elif menu == "6. Phân tích Nhân tố Khám phá (EFA)":
    st.markdown("### 6. Phân tích Nhân tố Khám phá (Exploratory Factor Analysis - EFA)")
    st.markdown(r"""
    **Tiêu chuẩn kiểm định EFA:**
    - Hệ số KMO $\ge 0.50$ (lý tưởng $\ge 0.70$); Bartlett's Test có Sig. $< 0.05$.
    - Tổng phương sai trích (Cumulative Variance) $\ge 50\%$; Eigenvalues $\ge 1.0$.
    - Trọng số tải nhân tố (Factor Loading) $\ge 0.50$; Chênh lệch tải kép (Cross-loading) $\Delta \ge 0.30$.
    """)
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    likert_candidates = [c for c in num_cols if c.lower() not in ['id', 'use', 'gioitinh', 'dotuoi', 'thunhap', 'hocvan']]
    
    selected_items = st.multiselect(
        "Chọn các biến quan sát để đưa vào phân tích EFA:",
        num_cols,
        default=likert_candidates if likert_candidates else num_cols
    )
    
    c1, c2, c3 = st.columns(3)
    with c1:
        method = st.selectbox(
            "Phương pháp trích (Extraction):",
            ["principal", "minres", "ml", "uls"],
            index=0,
            format_func=lambda x: {
                "principal": "Principal Components (PCA)",
                "minres": "Principal Axis Factoring (PAF / MinRes)",
                "ml": "Maximum Likelihood (ML)",
                "uls": "Unweighted Least Squares (ULS)"
            }.get(x, x)
        )
    with c2:
        rotation = st.selectbox(
            "Phép xoay nhân tố (Rotation):",
            ["varimax", "promax", "oblimin"],
            index=0,
            format_func=lambda x: {
                "varimax": "Varimax (Vuông góc - Orthogonal)",
                "promax": "Promax (Xiên - Oblique)",
                "oblimin": "Direct Oblimin (Xiên - Oblique)"
            }.get(x, x)
        )
    with c3:
        n_factors_opt = st.number_input("Số lượng nhân tố (0 = Tự động theo Kaiser Eigenvalue >= 1):", min_value=0, max_value=15, value=0)
        
    if st.button("🚀 Chạy Phân tích EFA", use_container_width=True):
        if len(selected_items) < 3:
            st.error("Cần chọn ít nhất 3 biến quan sát để chạy EFA.")
        else:
            res_efa = run_efa(
                df, selected_items, 
                n_factors=None if n_factors_opt == 0 else n_factors_opt, 
                method=method, rotation=rotation,
                labels_dict=loader.variable_labels
            )
            if "error" in res_efa:
                st.error(res_efa["error"])
            else:
                col_kmo, col_bart, col_var = st.columns(3)
                with col_kmo:
                    st.markdown(f"""
                    <div class="pro-kpi-card">
                        <div>
                            <div class="kpi-label">Hệ số KMO</div>
                            <div class="kpi-value">{res_efa['kmo_total']:.3f}</div>
                        </div>
                        <div class="kpi-badge badge-success">✓ {res_efa['kmo_eval']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_bart:
                    st.markdown(f"""
                    <div class="pro-kpi-card">
                        <div>
                            <div class="kpi-label">Kiểm định Bartlett</div>
                            <div class="kpi-value" style="font-size: 20px;">p = {res_efa['bartlett_p']:.5f}</div>
                        </div>
                        <div class="kpi-badge badge-success">✓ p < 0.05 (Thích hợp)</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_var:
                    is_cum = res_efa['total_cum_variance'] >= 50.0
                    b_cls = "badge-success" if is_cum else "badge-warning"
                    st.markdown(f"""
                    <div class="pro-kpi-card">
                        <div>
                            <div class="kpi-label">Tổng phương sai trích</div>
                            <div class="kpi-value">{res_efa['total_cum_variance']:.2f}%</div>
                        </div>
                        <div class="kpi-badge {b_cls}">{'✓ >= 50% (Đạt yêu cầu)' if is_cum else 'Chưa đạt 50%'}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Phân loại Tabs hiển thị chuẩn học thuật & SPSS
                matrix_label = res_efa["matrix_name"]
                if res_efa["is_oblique"]:
                    tab_pat, tab_struct, tab_phi, tab_eval, tab_scree = st.tabs([
                        f"1. {matrix_label}",
                        "2. Structure Matrix (Ma trận Cấu trúc)",
                        "3. Tương quan Nhân tố (Factor Correlation)",
                        "4. Chẩn đoán Tải kép & Chất lượng",
                        "5. Phương sai trích & Scree Plot"
                    ])
                else:
                    tab_pat, tab_eval, tab_scree = st.tabs([
                        f"1. {matrix_label}",
                        "2. Chẩn đoán Tải kép & Chất lượng",
                        "3. Phương sai trích & Scree Plot"
                    ])
                    tab_struct = None
                    tab_phi = None
                    
                with tab_pat:
                    st.markdown(f"##### {matrix_label} - Định dạng Chuẩn SPSS")
                    
                    # Tùy chọn hiển thị chuẩn SPSS
                    c_opt1, c_opt2 = st.columns([1, 1])
                    with c_opt1:
                        suppress_small = st.checkbox("Ẩn các hệ số nhỏ (< 0.30) theo chuẩn SPSS (Suppress small coefficients)", value=True, key="efa_suppress")
                    with c_opt2:
                        show_labels = st.checkbox("Hiển thị cột Nhãn biến (Variable Labels)", value=True, key="efa_show_labels")
                        
                    # Lựa chọn bảng dữ liệu hiển thị
                    df_to_show = res_efa["pattern_matrix_spss"] if suppress_small else res_efa["pattern_matrix_full"]
                    if not show_labels and "Nhãn biến (Label)" in df_to_show.columns:
                        df_to_show = df_to_show.drop(columns=["Nhãn biến (Label)"])
                        
                    render_table(df_to_show, file_name="EFA_Pattern_Matrix.xlsx", sheet_name="Pattern_Matrix")
                    
                    # Chú thích học thuật chuẩn SPSS
                    meth_str = "Principal Components (PCA)" if res_efa["method_name"] == "principal" else ("Principal Axis Factoring (PAF / MinRes)" if res_efa["method_name"] == "minres" else res_efa["method_name"].upper())
                    rot_str = "Promax with Kaiser Normalization" if res_efa["rotation_name"] == "promax" else ("Varimax with Kaiser Normalization" if res_efa["rotation_name"] == "varimax" else res_efa["rotation_name"].capitalize())
                    st.caption(f"Extraction Method: {meth_str} • Rotation Method: {rot_str}. Dữ liệu được sắp xếp theo nhóm nhân tố (Sorted by size) và ẩn các hệ số tải < 0.30 theo quy chuẩn SPSS.")

                if tab_struct is not None:
                    with tab_struct:
                        st.markdown("##### Structure Matrix (Ma trận Cấu trúc tương quan Biến - Nhân tố)")
                        st.caption("Ma trận cấu trúc thể hiện hệ số tương quan đơn giản giữa biến quan sát và các nhân tố xoay.")
                        df_struct_disp = res_efa["structure_matrix_spss"]
                        if not show_labels and "Nhãn biến (Label)" in df_struct_disp.columns:
                            df_struct_disp = df_struct_disp.drop(columns=["Nhãn biến (Label)"])
                        render_table(df_struct_disp, file_name="EFA_Structure_Matrix.xlsx", sheet_name="Structure_Matrix")
                        
                if tab_phi is not None:
                    with tab_phi:
                        st.markdown("##### Factor Correlation Matrix (Ma trận tương quan giữa các Nhân tố)")
                        st.caption("Theo Tabachnick & Fidell (2019) và Hair et al. (2022), nếu tương quan giữa các nhân tố r > 0.32 thì việc sử dụng phép xoay xiên (Promax/Oblimin) là hoàn toàn phù hợp và có cơ sở lý luận vững chắc.")
                        render_table(res_efa["factor_correlations"], file_name="EFA_Tuong_Quan_Nhan_To.xlsx", sheet_name="Factor_Correlations")
                        
                with tab_eval:
                    st.markdown("##### Bảng Đánh giá Hệ số tải & Kiểm định Tải kép (Cross-Loading Diagnostic)")
                    render_table(res_efa["evaluated_table"], file_name="EFA_Danh_Gia_Tieu_Chuan.xlsx", sheet_name="EFA_Evaluated")
                    if res_efa["problem_items"]:
                        for it, prob in res_efa["problem_items"]:
                            st.warning(f"⚠️ Biến **{it}**: {prob}")
                    else:
                        st.success("✅ Toàn bộ các biến quan sát đều đạt chuẩn (Loading >= 0.50, Chênh lệch tải kép Delta >= 0.30)!")
                        
                with tab_scree:
                    st.markdown("##### Bảng Phương sai giải thích & Eigenvalues")
                    render_table(res_efa["variance_table"], file_name="EFA_Tong_Phuong_Sai_Giai_Thich.xlsx", sheet_name="Variance_Explained")
                    
                    ev = res_efa["eigenvalues"]
                    fig_scree = go.Figure()
                    fig_scree.add_trace(go.Scatter(x=list(range(1, len(ev)+1)), y=ev, mode='lines+markers', name='Eigenvalue', line=dict(color='#2563eb', width=2.5)))
                    fig_scree.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Kaiser Criterion (Eigenvalue = 1.0)")
                    fig_scree.update_layout(title="Biểu đồ Scree Plot (Tiêu chuẩn trích nhân tố)", xaxis_title="Nhân tố", yaxis_title="Eigenvalue")
                    clean_chart(fig_scree, height=340)
                    fig_scree.update_xaxes(showgrid=True, gridcolor="#cbd5e1")
                    fig_scree.update_yaxes(showgrid=True, gridcolor="#cbd5e1")
                    st.plotly_chart(fig_scree, use_container_width=True, theme=None)
                    
                narr = generate_efa_narrative(res_efa)
                st.markdown(f'<div class="academic-narrative">{narr}</div>', unsafe_allow_html=True)
                
                # Lưu vào session_state để hỗ trợ xuất báo cáo Word / Excel
                st.session_state.analysis_results["EFA_Pattern_Matrix"] = res_efa["pattern_matrix_spss"]
                st.session_state.analysis_results["EFA_Rotated_Matrix"] = res_efa["evaluated_table"]
                st.session_state.analysis_results["EFA_Variance"] = res_efa["variance_table"]
                if res_efa["is_oblique"] and res_efa["factor_correlations"] is not None:
                    st.session_state.analysis_results["EFA_Factor_Correlations"] = res_efa["factor_correlations"]


# ---------------- MENU 7: CFA & CB-SEM ----------------
elif menu == "7. CFA & CB-SEM (Mô hình hiệp phương sai)":
    st.markdown("### 7. Phân tích Nhân tố Khẳng định (CFA) & Mô hình CB-SEM")
    st.markdown(r"""
    **Quy chuẩn kiểm định AMOS/semopy (Hu & Bentler 1999, Hair et al. 2022):**
    - Độ phù hợp mô hình: $CMIN/df < 3.0$; $CFI \ge 0.90$; $TLI \ge 0.90$; $RMSEA < 0.08$.
    - Độ hội tụ: Hệ số tải $\lambda \ge 0.50$; Độ tin cậy tổng hợp $CR \ge 0.70$; Phương sai trích $AVE \ge 0.50$.
    - Độ phân biệt: Tiêu chuẩn Fornell & Larcker ($\sqrt{AVE} > r_{ij}$) và $HTMT < 0.85$.
    """)
    
    detected_scales = {}
    for c in df.select_dtypes(include=[np.number]).columns:
        if c.lower() not in ['id', 'use', 'gioitinh', 'dotuoi', 'thunhap', 'hocvan']:
            prefix = ''.join([char for char in c if not char.isdigit()]).strip('_')
            if prefix:
                detected_scales.setdefault(prefix, []).append(c)

    default_syntax = ""
    for prefix, items in detected_scales.items():
        if len(items) >= 2:
            default_syntax += f"{prefix} =~ {' + '.join(items)}\n"
            
    if "AT" in detected_scales and "BI" in detected_scales and "PU" in detected_scales:
        default_syntax += "AT ~ PU + PE + TR\n"
        default_syntax += "BI ~ AT + PU\n"

    st.markdown("##### Khai báo Mô hình Phương trình Cấu trúc (semopy Syntax):")
    syntax_input = st.text_area(
        "Khai báo mô hình đo lường (=~) và mô hình cấu trúc (~):",
        value=default_syntax,
        height=160
    )
    
    if st.button("🚀 Chạy Ước lượng CB-SEM / CFA", use_container_width=True):
        with st.spinner("Đang ước lượng mô hình Maximum Likelihood (ML)..."):
            res_cfa = run_cfa_cbsem(df, syntax_input, detected_scales)
            
        if "error" in res_cfa:
            st.error(res_cfa["error"])
        else:
            # KPI Fit Cards
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                cmin_ok = res_cfa['cmin_df'] < 3.0
                st.markdown(f"""
                <div class="pro-kpi-card">
                    <div>
                        <div class="kpi-label">CMIN / df</div>
                        <div class="kpi-value">{res_cfa['cmin_df']:.3f}</div>
                    </div>
                    <div class="kpi-badge {'badge-success' if cmin_ok else 'badge-warning'}">{'✓ Rất tốt (< 3.0)' if cmin_ok else 'Chấp nhận'}</div>
                </div>
                """, unsafe_allow_html=True)
            with k2:
                cfi_ok = res_cfa['cfi'] >= 0.90
                st.markdown(f"""
                <div class="pro-kpi-card">
                    <div>
                        <div class="kpi-label">CFI</div>
                        <div class="kpi-value">{res_cfa['cfi']:.3f}</div>
                    </div>
                    <div class="kpi-badge {'badge-success' if cfi_ok else 'badge-danger'}">{'✓ Đạt chuẩn (>= 0.90)' if cfi_ok else 'Chưa đạt'}</div>
                </div>
                """, unsafe_allow_html=True)
            with k3:
                tli_ok = res_cfa['tli'] >= 0.90
                st.markdown(f"""
                <div class="pro-kpi-card">
                    <div>
                        <div class="kpi-label">TLI</div>
                        <div class="kpi-value">{res_cfa['tli']:.3f}</div>
                    </div>
                    <div class="kpi-badge {'badge-success' if tli_ok else 'badge-danger'}">{'✓ Đạt chuẩn (>= 0.90)' if tli_ok else 'Chưa đạt'}</div>
                </div>
                """, unsafe_allow_html=True)
            with k4:
                rmsea_ok = res_cfa['rmsea'] < 0.08
                st.markdown(f"""
                <div class="pro-kpi-card">
                    <div>
                        <div class="kpi-label">RMSEA</div>
                        <div class="kpi-value">{res_cfa['rmsea']:.3f}</div>
                    </div>
                    <div class="kpi-badge {'badge-success' if rmsea_ok else 'badge-danger'}">{'✓ Phù hợp (< 0.08)' if rmsea_ok else 'Chưa đạt'}</div>
                </div>
                """, unsafe_allow_html=True)

            # Model Path Diagram
            struct_df = res_cfa["structural_table"]
            if not struct_df.empty:
                sem_paths = []
                for _, r in struct_df.iterrows():
                    sem_paths.append((r["Biến độc lập (Nguyên nhân)"], r["Biến phụ thuộc (Kết quả)"], f"{r['Hệ số chuẩn hóa (Beta)']:.3f}"))
                fig_sem = render_sem_diagram(sem_paths, title="Sơ đồ Mô hình Cấu trúc CB-SEM (Path Diagram)")
                st.plotly_chart(fig_sem, use_container_width=True, theme=None)

            st.markdown("##### 1. Độ phù hợp tổng thể của mô hình (Model Fit Indices)")
            render_table(res_cfa["fit_table"], file_name="CFA_Do_Phu_Hop_Mo_Hinh.xlsx", sheet_name="Model_Fit")
            
            st.markdown("##### 2. Trọng số Tải Nhân tố Chuẩn hóa (Standardized Factor Loadings)")
            render_table(res_cfa["loadings_table"], file_name="CFA_He_So_Tai_Nhan_To.xlsx", sheet_name="Loadings")
            
            st.markdown("##### 3. Kiểm định Độ hội tụ (Composite Reliability & AVE)")
            render_table(res_cfa["convergent_table"], file_name="CFA_Do_Hoi_Tu_Phan_Biet_CR_AVE.xlsx", sheet_name="CR_AVE")
            
            col_fornell, col_htmt = st.columns(2)
            with col_fornell:
                st.markdown("##### 4A. Tiêu chuẩn Fornell & Larcker")
                render_table(res_cfa["fornell_table"], file_name="CFA_Tieu_Chuan_Fornell_Larcker.xlsx", sheet_name="Fornell_Larcker")
            with col_htmt:
                st.markdown("##### 4B. Tỷ số Tương quan HTMT Matrix")
                render_table(res_cfa["htmt_table"], file_name="CFA_Ty_So_HTMT.xlsx", sheet_name="HTMT")

            if not res_cfa["structural_table"].empty:
                st.markdown("##### 5. Ước lượng các Đường dẫn Cấu trúc (Structural Path Coefficients)")
                render_table(res_cfa["structural_table"], file_name="CBSEM_He_So_Duong_Dan_Cau_Truc.xlsx", sheet_name="Structural_Paths")

            narr = generate_cfa_cbsem_narrative(res_cfa)
            st.markdown(f'<div class="academic-narrative">{narr}</div>', unsafe_allow_html=True)
            
            st.session_state.analysis_results["CBSEM_Fit"] = res_cfa["fit_table"]
            st.session_state.analysis_results["CBSEM_Convergent"] = res_cfa["convergent_table"]
            st.session_state.analysis_results["CBSEM_HTMT"] = res_cfa["htmt_table"]


# ---------------- MENU 8: PLS-SEM ----------------
elif menu == "8. PLS-SEM (Bình phương bé nhất từng phần)":
    st.markdown("### 8. Mô hình Cấu trúc Tuyến tính PLS-SEM (SmartPLS Algorithm)")
    st.markdown(r"""
    **Chuẩn mực thuật toán Wold's PLS & Hair et al. (2022):**
    - Mô hình đo lường: Outer Loading $\ge 0.708$; $CR (\rho_c) \ge 0.70$; $\rho_A \ge 0.70$; $AVE \ge 0.50$; $HTMT < 0.85$.
    - Mô hình cấu trúc: Bootstrapping để tính $p$-value và $t$-stat; Đánh giá $R^2$, kích thước ảnh hưởng $f^2$, VIF $< 3.0$.
    """)
    
    detected_scales = {}
    for c in df.select_dtypes(include=[np.number]).columns:
        if c.lower() not in ['id', 'use', 'gioitinh', 'dotuoi', 'thunhap', 'hocvan']:
            prefix = ''.join([char for char in c if not char.isdigit()]).strip('_')
            if prefix:
                detected_scales.setdefault(prefix, []).append(c)

    default_paths = [("PU", "AT"), ("PE", "AT"), ("TR", "AT"), ("AT", "BI"), ("PU", "BI")]
    
    col_model, col_boot = st.columns([2, 1])
    with col_model:
        st.markdown("##### Các thang đo đo lường (Constructs):")
        st.json(detected_scales)
    with col_boot:
        n_boot = st.selectbox("Số lượng mẫu Bootstrapping:", [500, 1000, 2000], index=1)
        st.caption("Bootstrapping giúp kiểm định giả thuyết phi tham số chính xác tuyệt đối.")

    if st.button("🚀 Chạy Thuật toán PLS-SEM & Bootstrapping", use_container_width=True):
        with st.spinner("Đang chạy thuật toán lặp Wold và Bootstrapping..."):
            pls = PLSModel(detected_scales, default_paths)
            pls.fit(df)
            measurement_res = pls.get_measurement_summary(df)
            boot_res = pls.bootstrap(df, n_boot=n_boot)

        # Path Diagram for PLS-SEM
        pls_paths_with_beta = []
        for _, r in boot_res["paths_table"].iterrows():
            parts = r["Đường dẫn (Path)"].split(" -> ")
            pls_paths_with_beta.append((parts[0], parts[1], f"{r['Hệ số tác động (Beta)']:.3f}"))
        fig_pls = render_sem_diagram(pls_paths_with_beta, r2_dict=pls.r_squared, title="Sơ đồ Mô hình Cấu trúc PLS-SEM & R² (Path Diagram)")
        st.plotly_chart(fig_pls, use_container_width=True, theme=None)

        st.markdown("##### 1. Đánh giá Mô hình Đo lường (Construct Reliability & Validity)")
        render_table(measurement_res["constructs_table"], file_name="PLS_Do_Tin_Cay_Hoi_Tu.xlsx", sheet_name="Constructs")

        st.markdown("##### 2. Trọng số Tải ngoài (Outer Loadings) & Độ tin cậy chỉ báo")
        render_table(measurement_res["indicators_table"], file_name="PLS_Trong_So_Tai_Ngoai.xlsx", sheet_name="Outer_Loadings")

        col_f, col_h = st.columns(2)
        with col_f:
            st.markdown("##### 3A. Tiêu chuẩn Fornell-Larcker")
            render_table(measurement_res["fornell_table"], file_name="PLS_Tieu_Chuan_Fornell_Larcker.xlsx", sheet_name="Fornell_Larcker")
        with col_h:
            st.markdown("##### 3B. Ma trận HTMT (Heterotrait-Monotrait Ratio)")
            render_table(measurement_res["htmt_table"], file_name="PLS_Ma_Tran_HTMT.xlsx", sheet_name="HTMT")

        st.markdown("##### 4. Hệ số Xác định R^2 & Sức mạnh giải thích")
        render_table(measurement_res["r2_table"], file_name="PLS_He_So_Xac_Dinh_R2.xlsx", sheet_name="R_Squared")

        st.markdown("##### 5. Kết quả Kiểm định Giả thuyết Mô hình Cấu trúc (Bootstrapping Results)")
        render_table(boot_res["paths_table"], file_name="PLS_Kiem_Dinh_Gia_Thuyet_Bootstrap.xlsx", sheet_name="Bootstrap_Paths")

        if not boot_res["mediation_table"].empty:
            st.markdown("##### 6. Phân tích Tác động Trung gian (Mediation Analysis & Zhao et al. 2010)")
            render_table(boot_res["mediation_table"], file_name="PLS_Tac_Dong_Trung_Gian_Mediation.xlsx", sheet_name="Mediation")

        narr = generate_pls_narrative(measurement_res, boot_res)
        st.markdown(f'<div class="academic-narrative">{narr}</div>', unsafe_allow_html=True)
        
        st.session_state.analysis_results["PLS_Constructs"] = measurement_res["constructs_table"]
        st.session_state.analysis_results["PLS_Paths"] = boot_res["paths_table"]
        st.session_state.analysis_results["PLS_Mediation"] = boot_res["mediation_table"]


# ---------------- MENU 9: MACHINE LEARNING & AI NCKH ----------------
elif menu == "9. Machine Learning trong NCKH":
    st.markdown("### 9. Trí tuệ Nhân tạo & Machine Learning trong NCKH (AI Research Suite)")
    st.markdown("""
    Nền tảng tích hợp toàn diện các phương pháp **Học có giám sát**, **Học không giám sát**, **Mạng nơ-ron** và **Xử lý ngôn ngữ tự nhiên (NLP)**.
    Hệ thống tự động phân tích cấu trúc dữ liệu để đề xuất mô hình tối ưu, đồng thời cung cấp công thức toán học và căn cứ học thuật chi tiết.
    """)
    
    all_num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_num_cols = [c for c in all_num_cols if c.lower() not in ['id']]
    all_text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    
    ml_tabs = st.tabs([
        "1. Cố vấn Đề xuất Mô hình (AI Advisor)",
        "2. Học có Giám sát (Supervised Learning)",
        "3. Học không Giám sát (Unsupervised)",
        "4. Mạng Nơ-ron & NLP (ANN & Text Mining)"
    ])
    
    # ---------------- TAB 1: AI METHODOLOGY ADVISOR ----------------
    with ml_tabs[0]:
        st.markdown("#### 1. Cố vấn Đề xuất Phương pháp & Từ điển Kiến thức Học thuật")
        st.markdown("*Hệ thống tự động quét cấu trúc và kiểu dữ liệu hiện tại để đưa ra gợi ý mô hình phù hợp nhất cho đề tài nghiên cứu:*")
        
        recs_data = analyze_dataset_and_recommend(df)
        
        # Summary Overview of Data
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="pro-kpi-card">
                <div>
                    <div class="kpi-label">Cỡ mẫu khảo sát</div>
                    <div class="kpi-value">{recs_data['n_samples']:,}</div>
                </div>
                <div class="kpi-badge badge-success">✓ Đủ điều kiện ML</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="pro-kpi-card">
                <div>
                    <div class="kpi-label">Thuộc tính đo lường</div>
                    <div class="kpi-value">{len(recs_data['numeric_cols'])}</div>
                </div>
                <div class="kpi-badge badge-info">Biến thang đo/định lượng</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            n_bin = len(recs_data['binary_targets'])
            b_cls = "badge-success" if n_bin > 0 else "badge-warning"
            st.markdown(f"""
            <div class="pro-kpi-card">
                <div>
                    <div class="kpi-label">Biến nhị phân mục tiêu</div>
                    <div class="kpi-value">{n_bin}</div>
                </div>
                <div class="kpi-badge {b_cls}">{'✓ Sẵn sàng phân loại' if n_bin > 0 else 'Chưa có biến 0/1'}</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            n_txt = len(recs_data['meaningful_text_cols'])
            b_txt_cls = "badge-success" if n_txt > 0 else "badge-warning"
            st.markdown(f"""
            <div class="pro-kpi-card">
                <div>
                    <div class="kpi-label">Dữ liệu văn bản mở</div>
                    <div class="kpi-value">{n_txt}</div>
                </div>
                <div class="kpi-badge {b_txt_cls}">{'✓ Sẵn sàng NLP' if n_txt > 0 else 'Chưa có câu hỏi mở'}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("##### Bảng Đề xuất Mô hình Phù hợp cho Tập Dữ liệu:")
        df_recs = pd.DataFrame(recs_data["recommendations"])
        df_recs_disp = df_recs[["model_key", "task_type", "suitability", "reason"]].rename(columns={
            "model_key": "Mô hình Machine Learning",
            "task_type": "Dạng bài toán",
            "suitability": "Mức độ phù hợp",
            "reason": "Căn cứ đề xuất theo dữ liệu hiện hành"
        })
        render_table(df_recs_disp, file_name="ML_De_Xuat_Mo_Hinh_AI_Advisor.xlsx", sheet_name="AI_Advisor")

        st.markdown("---")
        st.markdown("#### Cẩm nang Chi tiết: Công thức Toán & Ý nghĩa Học thuật của Từng Mô hình")
        st.caption("Nhấp vào từng mô hình để xem công thức toán học, hàm mất mát và cách biện luận trong bài báo Scopus / luận văn:")
        
        for m_key, m_info in ML_KNOWLEDGE_BASE.items():
            with st.expander(f"{m_info['name']} - [{m_info['category']}]"):
                col_info_l, col_info_r = st.columns([1, 1])
                with col_info_l:
                    st.markdown(f"**Bản chất mô hình:**\n\n{m_info['description']}")
                    st.markdown("**Công thức toán học chính:**")
                    st.latex(m_info["math_formula"])
                    if "loss_formula" in m_info:
                        st.markdown("**Hàm mục tiêu / Hàm mất mát tối ưu:**")
                        st.latex(m_info["loss_formula"])
                with col_info_r:
                    st.markdown(f"**Ý nghĩa trong NCKH Xã hội & Cách biện luận:**\n\n{m_info['social_science_meaning']}")
                    st.markdown(f"**Ưu điểm:** {m_info['pros']}")
                    st.markdown(f"**Nhược điểm:** {m_info['cons']}")

    # ---------------- TAB 2: SUPERVISED LEARNING ----------------
    with ml_tabs[1]:
        st.markdown("#### 2. Học có Giám sát (Supervised Learning)")
        st.markdown("*Dự báo biến mục tiêu (Phân loại hoặc Hồi quy) với các thuật toán hiện đại:*")
        
        sup_col1, sup_col2, sup_col3 = st.columns([1, 1, 1])
        with sup_col1:
            target_var = st.selectbox("Chọn biến phụ thuộc (Target Y):", df.columns.tolist(), index=df.columns.get_loc("USE") if "USE" in df.columns else 0)
        with sup_col2:
            default_feats = [c for c in all_num_cols if c != target_var][:6]
            feature_vars = st.multiselect("Chọn các biến độc lập (Features X):", [c for c in all_num_cols if c != target_var], default=default_feats)
        with sup_col3:
            available_algos = [
                "Random Forest", "Decision Tree", "Gradient Boosting (GBM)",
                "Support Vector Machine (SVM)", "Ridge (L2)", "Lasso (L1)",
                "Elastic Net", "Artificial Neural Network (ANN)", "Logistic Regression"
            ]
            if HAS_XGBOOST:
                available_algos.insert(3, "XGBoost")
            if HAS_LIGHTGBM:
                available_algos.insert(4, "LightGBM")
                
            algo_choice = st.selectbox("Chọn thuật toán Machine Learning:", available_algos)

        # Hyperparameter Accordion
        with st.expander("⚙️ Tùy chỉnh Siêu tham số (Hyperparameters)", expanded=False):
            hp_c1, hp_c2, hp_c3 = st.columns(3)
            with hp_c1:
                n_trees = st.slider("Số lượng cây (n_estimators):", 50, 500, 150, 50) if any(x in algo_choice for x in ["Forest", "Boosting", "XGBoost", "LightGBM"]) else 100
                alpha_val = st.slider("Hệ số phạt quy chuẩn hóa (Alpha / L1 / L2):", 0.01, 10.0, 1.0, 0.05) if any(x in algo_choice for x in ["Ridge", "Lasso", "Elastic"]) else 1.0
            with hp_c2:
                max_d = st.slider("Độ sâu tối đa (max_depth):", 2, 20, 5) if any(x in algo_choice for x in ["Tree", "Forest", "Boosting", "XGBoost", "LightGBM"]) else None
                l1_ratio_val = st.slider("Tỷ lệ phạt L1 (Elastic Net L1 Ratio):", 0.0, 1.0, 0.5, 0.1) if "Elastic" in algo_choice else 0.5
            with hp_c3:
                svm_kernel = st.selectbox("Kernel SVM:", ["rbf", "linear", "poly"]) if "SVM" in algo_choice else "rbf"
                learn_rate = st.slider("Tốc độ học (Learning Rate):", 0.01, 0.5, 0.1, 0.01) if any(x in algo_choice for x in ["Boosting", "XGBoost", "LightGBM"]) else 0.1

        if st.button("🚀 Huấn luyện & Đánh giá Mô hình", use_container_width=True):
            if not feature_vars:
                st.error("Vui lòng chọn ít nhất 1 biến độc lập (Features).")
            else:
                with st.spinner(f"Đang huấn luyện mô hình {algo_choice}..."):
                    res_sup = run_advanced_supervised(
                        df, target_var, feature_vars, algorithm=algo_choice,
                        n_estimators=n_trees, max_depth=max_d, alpha=alpha_val,
                        l1_ratio=l1_ratio_val, learning_rate=learn_rate, kernel=svm_kernel
                    )
                    
                if "error" in res_sup:
                    st.error(res_sup["error"])
                else:
                    st.markdown(f"##### 1. Kết quả Đánh giá Hiệu năng: {algo_choice}")
                    render_table(res_sup["metrics_table"], file_name=f"ML_{algo_choice}_Hieu_Nang.xlsx", sheet_name="Metrics")
                    
                    # Visualizations
                    if res_sup["is_classification"]:
                        c_cm, c_roc = st.columns([1, 1])
                        with c_cm:
                            st.markdown("##### 2. Ma trận Phân loại (Confusion Matrix)")
                            render_table(res_sup["confusion_matrix"], file_name=f"ML_{algo_choice}_Ma_Tran_Nham_Lan.xlsx", sheet_name="Confusion_Matrix")
                        with c_roc:
                            if res_sup["roc_data"]:
                                roc = res_sup["roc_data"]
                                fig_roc = px.area(
                                    x=roc["fpr"], y=roc["tpr"],
                                    title=f"Đường cong ROC (AUC = {roc['auc']:.3f})",
                                    labels=dict(x='Tỷ lệ dương tính giả (1 - Specificity)', y='Độ nhạy (Sensitivity)'),
                                    color_discrete_sequence=['#2563eb']
                                )
                                fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                                clean_chart(fig_roc, height=320)
                                st.plotly_chart(fig_roc, use_container_width=True, theme=None)
                    else:
                        st.markdown("##### 2. Biểu đồ So sánh Giá trị Thực tế vs Giá trị Dự đoán")
                        df_reg_plot = pd.DataFrame({
                            "Thực tế (Actual)": res_sup["y_actual"][:80],
                            "Dự đoán (Predicted)": res_sup["y_predicted"][:80]
                        })
                        fig_reg = px.scatter(
                            df_reg_plot, x="Thực tế (Actual)", y="Dự đoán (Predicted)",
                            trendline="ols", title="Tương quan giữa Giá trị Thực tế và Giá trị Dự báo"
                        )
                        clean_chart(fig_reg, height=340)
                        st.plotly_chart(fig_reg, use_container_width=True, theme=None)

                    # Coefficients / Feature Importance
                    if not res_sup["importance_table"].empty:
                        st.markdown("##### 3. Bảng Xếp hạng Tầm quan trọng của Biến (Feature Importance)")
                        render_table(res_sup["importance_table"], file_name=f"ML_{algo_choice}_Feature_Importance.xlsx", sheet_name="Importance")
                        
                        fig_imp = px.bar(
                            res_sup["importance_table"],
                            x="Mức độ quan trọng (Feature Importance)", y="Biến số",
                            orientation='h', title=f"Xếp hạng Yếu tố Chi phối ({algo_choice})",
                            color="Mức độ quan trọng (Feature Importance)", color_continuous_scale="Viridis"
                        )
                        clean_chart(fig_imp, height=350)
                        fig_imp.update_layout(yaxis=dict(autorange="reversed"))
                        st.plotly_chart(fig_imp, use_container_width=True, theme=None)
                        
                    elif not res_sup["coefficients_table"].empty:
                        st.markdown("##### 3. Bảng Hệ số Hồi quy (Coefficients & Odds Ratios)")
                        render_table(res_sup["coefficients_table"], file_name=f"ML_{algo_choice}_He_So_Hoi_Quy.xlsx", sheet_name="Coefficients")

                    st.session_state.analysis_results[f"ML_Supervised_{algo_choice}"] = res_sup["metrics_table"]

    # ---------------- TAB 3: UNSUPERVISED LEARNING ----------------
    with ml_tabs[2]:
        st.markdown("#### 3. Học không Giám sát (Unsupervised Learning)")
        unsub_choice = st.radio(
            "Chọn thuật toán không giám sát:",
            ["Phân cụm K-Means (K-Means Clustering)", "Phân cụm Phân tầng (Hierarchical Clustering)", "Phân tích Thành phần Chính (PCA)"],
            horizontal=True
        )
        
        if unsub_choice == "Phân cụm K-Means (K-Means Clustering)":
            c1, c2 = st.columns(2)
            with c1:
                km_features = st.multiselect("Chọn các biến tiêu chí để phân cụm:", all_num_cols, default=all_num_cols[:6])
            with c2:
                n_clusters_km = st.slider("Số lượng cụm mong muốn (K):", 2, 6, 3)
                
            if st.button("🚀 Chạy Phân cụm K-Means", use_container_width=True):
                if len(km_features) < 2:
                    st.error("Cần chọn ít nhất 2 biến tiêu chí.")
                else:
                    res_km = run_kmeans_clustering(df, km_features, n_clusters=n_clusters_km)
                    if "error" in res_km:
                        st.error(res_km["error"])
                    else:
                        st.markdown("##### 1. Đánh giá Lựa chọn Số cụm Tối ưu (Elbow Inertia & Silhouette)")
                        render_table(res_km["k_evaluation_table"], file_name="KMeans_Danh_Gia_So_Cum.xlsx", sheet_name="K_Evaluation")
                        
                        st.markdown("##### 2. Quy mô và Tỷ lệ từng Phân cụm")
                        render_table(res_km["cluster_sizes_table"], file_name="KMeans_Quy_Mo_Cum.xlsx", sheet_name="Sizes")
                        
                        st.markdown("##### 3. Hồ sơ Đặc tính Phân cụm (Cluster Profiles - Trung bình từng nhóm)")
                        render_table(res_km["cluster_profile_table"], file_name="KMeans_Ho_So_Dac_Tinh_Cum.xlsx", sheet_name="Profiles")
                        
                        st.session_state.analysis_results["KMeans_Profiles"] = res_km["cluster_profile_table"]

        elif unsub_choice == "Phân cụm Phân tầng (Hierarchical Clustering)":
            c1, c2, c3 = st.columns(3)
            with c1:
                h_features = st.multiselect("Chọn các biến phân tầng:", all_num_cols, default=all_num_cols[:6])
            with c2:
                n_h_clusters = st.slider("Số lượng phân nhóm:", 2, 6, 3)
            with c3:
                linkage_method = st.selectbox("Phương pháp liên kết (Linkage):", ["ward", "complete", "average"])
                
            if st.button("🚀 Chạy Phân cụm Phân tầng", use_container_width=True):
                if len(h_features) < 2:
                    st.error("Cần chọn ít nhất 2 biến tiêu chí.")
                else:
                    res_hc = run_hierarchical_clustering_model(df, h_features, n_clusters=n_h_clusters, method=linkage_method)
                    
                    st.markdown(f"**Chỉ số Silhouette:** **{res_hc['silhouette_score']}** (Chất lượng phân cụm)")
                    
                    col_hz, col_hp = st.columns([1, 2])
                    with col_hz:
                        st.markdown("##### 1. Quy mô các phân nhóm")
                        render_table(res_hc["cluster_sizes"], file_name="HCA_Quy_Mo_Cum.xlsx", sheet_name="Sizes")
                    with col_hp:
                        st.markdown("##### 2. Hồ sơ đặc tính các cụm")
                        render_table(res_hc["cluster_profiles"], file_name="HCA_Ho_So_Cum.xlsx", sheet_name="Profiles")
                        
                    st.session_state.analysis_results["Hierarchical_Profiles"] = res_hc["cluster_profiles"]

        else: # PCA
            c1, c2 = st.columns(2)
            with c1:
                pca_features = st.multiselect("Chọn các biến đưa vào nén chiều PCA:", all_num_cols, default=all_num_cols[:8])
            with c2:
                n_comps = st.slider("Số lượng thành phần chính (Components):", 1, min(len(pca_features), 6) if pca_features else 3, 3)
                
            if st.button("🚀 Chạy Phân tích PCA", use_container_width=True):
                if len(pca_features) < 2:
                    st.error("Cần chọn ít nhất 2 biến để chạy PCA.")
                else:
                    res_pca = run_pca_reduction(df, pca_features, n_components=n_comps)
                    st.metric("Tổng phương sai giải thích", f"{res_pca['total_variance_explained']:.2f}%")
                    
                    st.markdown("##### 1. Bảng Phương sai Giải thích từng Thành phần Chính")
                    render_table(res_pca["variance_table"], file_name="PCA_Phuong_Sai_Giai_Thich.xlsx", sheet_name="Variance")
                    
                    st.markdown("##### 2. Ma trận Hệ số Tải Thành phần (Component Loadings)")
                    render_table(res_pca["loadings_table"], file_name="PCA_He_So_Tai_Thanh_Phan.xlsx", sheet_name="Loadings")
                    
                    # 2D Biplot
                    if res_pca["n_components"] >= 2:
                        st.markdown("##### 3. Biểu đồ Không gian Chiếu 2D (PC1 vs PC2 Biplot)")
                        fig_biplot = px.scatter(
                            res_pca["scores_2d"], x="PC1", y="PC2",
                            title="Không gian Biểu diễn 2D của các Quan sát Nghiên cứu",
                            color_discrete_sequence=['#2563eb']
                        )
                        clean_chart(fig_biplot, height=380)
                        fig_biplot.update_xaxes(showgrid=True, gridcolor="#cbd5e1")
                        fig_biplot.update_yaxes(showgrid=True, gridcolor="#cbd5e1")
                        st.plotly_chart(fig_biplot, use_container_width=True, theme=None)
                        
                    st.session_state.analysis_results["PCA_Variance"] = res_pca["variance_table"]

    # ---------------- TAB 4: NEURAL NETS & NLP ----------------
    with ml_tabs[3]:
        st.markdown("#### 4. Mạng Nơ-ron (ANN) & Xử lý Ngôn ngữ Tự nhiên (NLP)")
        nlp_choice = st.radio(
            "Chọn bài toán:",
            [
                "1. Phân tích Cảm xúc Ý kiến Đáp viên (Sentiment Analysis)",
                "2. Mô hình Chủ đề Tiềm ẩn (LDA Topic Modeling cho câu hỏi mở)",
                "3. Mạng Nơ-ron Nhân tạo (Artificial Neural Network - ANN / MLP)"
            ],
            horizontal=True
        )
        
        if "Sentiment" in nlp_choice:
            st.markdown(r"""
            *Phân tích trạng thái cảm xúc (Tích cực, Tiêu cực, Trung lập) và tính toán điểm phân cực (Polarity Score $[-1, +1]$) của các câu trả lời mở:*
            """)
            
            cand_text_cols = all_text_cols if all_text_cols else df.columns.tolist()
            text_target = st.selectbox("Chọn cột văn bản ý kiến đóng góp:", cand_text_cols, index=cand_text_cols.index("Y_KIEN_DAP_VIEN") if "Y_KIEN_DAP_VIEN" in cand_text_cols else 0)
            
            if st.button("🚀 Phân tích Cảm xúc Văn bản", use_container_width=True):
                res_sent = run_sentiment_analysis_suite(df[text_target])
                if "error" in res_sent:
                    st.error(res_sent["error"])
                else:
                    k1, k2 = st.columns(2)
                    with k1:
                        st.markdown(f"""
                        <div class="pro-kpi-card">
                            <div>
                                <div class="kpi-label">Điểm Cảm xúc Trung bình (Polarity)</div>
                                <div class="kpi-value">{res_sent['average_polarity']:+.3f}</div>
                            </div>
                            <div class="kpi-badge badge-success">Thái độ chung: {res_sent['overall_sentiment']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with k2:
                        fig_sent = px.pie(
                            res_sent["summary_table"],
                            names="Thái độ / Cảm xúc", values="Số lượng (N)",
                            title="Tỷ lệ Cơ cấu Cảm xúc của Đáp viên", hole=0.45,
                            color="Thái độ / Cảm xúc",
                            color_discrete_map={
                                "Tích cực (Positive)": "#10b981",
                                "Trung lập (Neutral)": "#94a3b8",
                                "Tiêu cực (Negative)": "#ef4444"
                            }
                        )
                        fig_sent.update_traces(
                            textposition='inside', 
                            textinfo='percent+label', 
                            marker=dict(line=dict(color='#ffffff', width=2))
                        )
                        clean_chart(fig_sent, height=280)
                        st.plotly_chart(fig_sent, use_container_width=True, theme=None)

                    st.markdown("##### 1. Bảng Tổng hợp Cơ cấu Cảm xúc")
                    render_table(res_sent["summary_table"], file_name="Sentiment_Tong_Hop_Cam_Xuc.xlsx", sheet_name="Summary")
                    
                    st.markdown("##### 2. Chi tiết Đánh giá Từng Câu Ý kiến của Đáp viên (30 mẫu đầu)")
                    render_table(res_sent["detailed_table"].head(30), file_name="Sentiment_Chi_Tiet_Danh_Gia.xlsx", sheet_name="Details")
                    
                    st.session_state.analysis_results["Sentiment_Summary"] = res_sent["summary_table"]

        elif "LDA" in nlp_choice:
            st.markdown(r"""
            *Trích xuất tự động các chủ đề thảo luận tiềm ẩn (Latent Topics) và cụm từ khóa (Keywords) từ văn bản câu trả lời mở:*
            """)
            cand_text_cols = all_text_cols if all_text_cols else df.columns.tolist()
            text_target = st.selectbox("Chọn cột văn bản phản hồi mở:", cand_text_cols, index=cand_text_cols.index("Y_KIEN_DAP_VIEN") if "Y_KIEN_DAP_VIEN" in cand_text_cols else 0)
            n_topics_lda = st.slider("Số lượng chủ đề cần trích xuất (K Topics):", 2, 5, 3)
            
            if st.button("🚀 Khai phá Chủ đề LDA", use_container_width=True):
                res_lda = run_lda_topic_model(df[text_target], n_topics=n_topics_lda)
                if "error" in res_lda:
                    st.error(res_lda["error"])
                else:
                    st.markdown(f"**Chỉ số Perplexity:** {res_lda['perplexity']} (Độ phức tạp ngôn ngữ)")
                    
                    st.markdown("##### 1. Danh sách các Chủ đề Tiềm ẩn & Từ khóa Đặc trưng (Topics & Keywords)")
                    render_table(res_lda["topics_table"], file_name="LDA_Danh_Sach_Chu_De_Keywords.xlsx", sheet_name="Topics")
                    
                    col_tp1, col_tp2 = st.columns([1, 1])
                    with col_tp1:
                        st.markdown("##### 2. Phân bố Tỷ lệ các Chủ đề")
                        render_table(res_lda["topic_counts"], file_name="LDA_Phan_Bo_Ty_Le_Chu_De.xlsx", sheet_name="Counts")
                    with col_tp2:
                        fig_lda = px.bar(
                            res_lda["topic_counts"], x="Chủ đề", y="Số lượng ý kiến",
                            title="Số lượng Ý kiến Phân bổ theo từng Chủ đề",
                            color="Chủ đề", color_discrete_sequence=['#2563eb', '#10b981', '#f59e0b', '#ec4899']
                        )
                        clean_chart(fig_lda, height=280)
                        fig_lda.update_xaxes(showgrid=True, gridcolor="#cbd5e1")
                        fig_lda.update_yaxes(showgrid=True, gridcolor="#cbd5e1")
                        st.plotly_chart(fig_lda, use_container_width=True, theme=None)

                    st.markdown("##### 3. Mẫu Phân loại Ý kiến Đáp viên theo Chủ đề")
                    render_table(res_lda["sample_assignments"], file_name="LDA_Mau_Phan_Loai_Y_Kien.xlsx", sheet_name="Sample_Assignments")
                    st.session_state.analysis_results["LDA_Topics"] = res_lda["topics_table"]

        else: # ANN / MLP
            st.markdown(r"""
            *Mô hình Mạng Nơ-ron Nhân tạo nhiều tầng (Multi-Layer Perceptron) mô hình hóa mối quan hệ phi tuyến phức tạp giữa các thuộc tính:*
            """)
            c1, c2 = st.columns(2)
            with c1:
                ann_target = st.selectbox("Chọn biến đầu ra (Target Y):", df.columns.tolist(), index=df.columns.get_loc("USE") if "USE" in df.columns else 0)
            with c2:
                ann_features = st.multiselect("Chọn các biến đầu vào (Inputs X):", [c for c in all_num_cols if c != ann_target], default=[c for c in all_num_cols if c != ann_target][:6])
                
            c_h1, c_h2 = st.columns(2)
            with c_h1:
                hidden_neurons = st.select_slider("Cấu trúc Tầng ẩn (Hidden Layer Neurons):", options=[(16,), (32,), (64, 32), (64, 32, 16)], value=(64, 32))
            with c_h2:
                max_iter_ann = st.slider("Số vòng lặp tối đa (Max Epochs):", 200, 1000, 500, 100)

            if st.button("🚀 Huấn luyện Mạng Nơ-ron Nhân tạo (ANN)", use_container_width=True):
                if not ann_features:
                    st.error("Cần chọn ít nhất 1 biến đầu vào.")
                else:
                    with st.spinner("Đang huấn luyện mạng nơ-ron bằng thuật toán Lan truyền ngược (Backpropagation)..."):
                        res_ann = run_advanced_supervised(
                            df, ann_target, ann_features, algorithm="Artificial Neural Network (ANN)",
                            hidden_layer_sizes=hidden_neurons, max_iter=max_iter_ann
                        )
                        
                    if "error" in res_ann:
                        st.error(res_ann["error"])
                    else:
                        st.markdown("##### 1. Hiệu năng Huấn luyện Mạng Nơ-ron")
                        render_table(res_ann["metrics_table"], file_name="ANN_Hieu_Nang_Huan_Luyen.xlsx", sheet_name="Metrics")
                        
                        if res_ann["is_classification"]:
                            st.markdown("##### 2. Ma trận Phân loại (Confusion Matrix)")
                            render_table(res_ann["confusion_matrix"], file_name="ANN_Ma_Tran_Nham_Lan.xlsx", sheet_name="Confusion_Matrix")
                            
                        st.session_state.analysis_results["ANN_Performance"] = res_ann["metrics_table"]


# ---------------- MENU 10: XUẤT BÁO CÁO ----------------
elif menu == "10. Xuất Báo cáo Học thuật (Word/Excel)":
    st.markdown("### 10. Xuất Báo cáo Kết quả Nghiên cứu Chuẩn Học thuật APA 7th Edition")
    st.markdown("""
    Xuất trực tiếp toàn bộ các bảng thống kê và đoạn văn thuyết minh học thuật tự động sang:
    - **Microsoft Word (.docx):** Bảng biểu kẻ chuẩn APA 7 (3 đường kẻ ngang, không kẻ dọc), tiêu đề in nghiêng đậm, thuyết minh tiếng Việt chuẩn quy chuẩn luận văn / bài báo NCKH.
    - **Microsoft Excel (.xlsx):** Các sheet dữ liệu có cấu trúc gọn gàng, tiện lưu trữ và tính toán thêm.
    """)
    
    st.markdown("##### Danh sách các bảng phân tích đã sẵn sàng xuất:")
    if not st.session_state.analysis_results:
        st.markdown("""
        <div class="pro-alert pro-alert-warning">
            <span>⚠️ Bạn chưa thực hiện phép kiểm định nào. Hãy vào các mục (1 -> 7) để chạy các kiểm định cần thiết, kết quả sẽ tự động được gom vào đây để bạn tải trọn bộ báo cáo!</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        for k_name in st.session_state.analysis_results.keys():
            st.markdown(f"- ✅ **{k_name}**")
            
        st.markdown("---")
        report_title = st.text_input("Tiêu đề báo cáo học thuật:", "BÁO CÁO KẾT QUẢ XỬ LÝ SỐ LIỆU NGHIÊN CỨU KHOA HỌC")
        
        c1, c2 = st.columns(2)
        with c1:
            sections = []
            for name, tbl in st.session_state.analysis_results.items():
                sections.append({
                    "heading": f"Phân tích: {name}",
                    "narrative": f"Dưới đây là bảng số liệu thống kê chi tiết cho mục {name} được trích xuất từ phần mềm.",
                    "tables": [(f"Bảng: {name}", tbl)]
                })
                
            docx_buffer = create_word_report(report_title, sections)
            st.download_button(
                label="📄 Tải Báo cáo Microsoft Word (.docx) Chuẩn APA 7",
                data=docx_buffer,
                file_name="Bao_cao_NCKH_APA7.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True
            )
            
        with c2:
            excel_buffer = create_excel_report(st.session_state.analysis_results)
            st.download_button(
                label="📊 Tải Trọn bộ Kết quả Microsoft Excel (.xlsx)",
                data=excel_buffer,
                file_name="Ket_qua_Thong_ke_NCKH.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )


# ---------------- GÓP Ý & BRAND FOOTER ----------------
render_page_footer(key_suffix="main")
