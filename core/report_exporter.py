"""
Report Exporter Module
Exports all statistical results and narrative interpretations to:
1. Microsoft Word (.docx) with APA 7th edition table formatting
2. Microsoft Excel (.xlsx) with multiple organized analysis sheets
"""

import io
import re
import mimetypes
try:
    if hasattr(mimetypes, "MimeTypes") and hasattr(mimetypes.MimeTypes, "read_windows_registry"):
        mimetypes.MimeTypes.read_windows_registry = lambda self, *args, **kwargs: None
    mimetypes.init()
except Exception:
    pass

import pandas as pd
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


def _set_cell_border(cell, **kwargs):
    """
    Sets individual cell borders in Word XML.
    kwargs: top, bottom, left, right with values like {"sz": 12, "val": "single", "color": "000000"}
    """
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = tcPr.first_child_found_in("w:tcBorders")
    if tcBorders is None:
        tcBorders = OxmlElement('w:tcBorders')
        tcPr.append(tcBorders)
    
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        edge_data = kwargs.get(edge)
        if edge_data:
            tag = 'w:{}'.format(edge)
            element = tcBorders.find(qn(tag))
            if element is None:
                element = OxmlElement(tag)
                tcBorders.append(element)
            for key, val in edge_data.items():
                element.set(qn('w:{}'.format(key)), str(val))


def create_word_report(title: str, sections: list) -> io.BytesIO:
    """
    Creates a professional academic Word document in APA 7th style:
    sections: list of dicts:
    [
        {
            "heading": "1. Thống kê mô tả và độ tin cậy Cronbach's Alpha",
            "narrative": "Đoạn văn thuyết minh học thuật...",
            "tables": [("Bảng 1: Kết quả Cronbach's Alpha", df_table)]
        },
        ...
    ]
    """
    doc = docx.Document()
    
    # Page setup: Standard A4, 1 inch margins
    sections_doc = doc.sections
    for s in sections_doc:
        s.top_margin = Inches(1)
        s.bottom_margin = Inches(1)
        s.left_margin = Inches(1)
        s.right_margin = Inches(1)
        
    # Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = p_title.add_run(title)
    run_title.font.name = 'Times New Roman'
    run_title.font.size = Pt(18)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(26, 54, 93) # Deep academic navy
    
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_sub = p_sub.add_run("Báo cáo xuất tự động theo chuẩn quy chuẩn học thuật APA 7th Edition")
    run_sub.font.name = 'Times New Roman'
    run_sub.font.size = Pt(11)
    run_sub.font.italic = True
    
    doc.add_paragraph() # Spacing
    
    for sec in sections:
        # Heading
        heading_text = sec.get("heading", "")
        if heading_text:
            h = doc.add_heading(heading_text, level=1)
            for r in h.runs:
                r.font.name = 'Times New Roman'
                r.font.size = Pt(14)
                r.font.bold = True
                r.font.color.rgb = RGBColor(30, 41, 59)
                
        # Narrative
        narrative = sec.get("narrative", "")
        if narrative:
            for p_text in narrative.split("\n\n"):
                if p_text.strip():
                    p = doc.add_paragraph()
                    p.paragraph_format.line_spacing = 1.2
                    p.paragraph_format.space_after = Pt(6)
                    # Clean markdown bold markers
                    clean_text = p_text.replace("**", "").replace("*", "")
                    r = p.add_run(clean_text)
                    r.font.name = 'Times New Roman'
                    r.font.size = Pt(11)

        # Tables
        tables = sec.get("tables", [])
        for tbl_title, df in tables:
            if df is None or df.empty:
                continue
                
            # Table Title in APA format
            p_tbl = doc.add_paragraph()
            p_tbl.paragraph_format.space_before = Pt(8)
            p_tbl.paragraph_format.space_after = Pt(2)
            r_tbl_title = p_tbl.add_run(tbl_title)
            r_tbl_title.font.name = 'Times New Roman'
            r_tbl_title.font.size = Pt(11)
            r_tbl_title.font.bold = True
            
            # Word table
            n_rows, n_cols = df.shape
            table = doc.add_table(rows=n_rows + 1, cols=n_cols)
            table.alignment = WD_TABLE_ALIGNMENT.CENTER
            
            # Header Row
            for col_idx, col_name in enumerate(df.columns):
                cell = table.cell(0, col_idx)
                cell.text = str(col_name)
                for prg in cell.paragraphs:
                    prg.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for r in prg.runs:
                        r.font.name = 'Times New Roman'
                        r.font.size = Pt(10)
                        r.font.bold = True
                _set_cell_border(cell, top={"sz": 8, "val": "single", "color": "000000"},
                                      bottom={"sz": 8, "val": "single", "color": "000000"},
                                      left={"val": "nil"}, right={"val": "nil"})
                
            # Data Rows
            for row_idx in range(n_rows):
                for col_idx in range(n_cols):
                    cell = table.cell(row_idx + 1, col_idx)
                    val = df.iloc[row_idx, col_idx]
                    cell.text = "" if pd.isna(val) else str(val)
                    for prg in cell.paragraphs:
                        # Left align strings, right align numbers
                        if isinstance(val, (int, float)):
                            prg.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                        else:
                            prg.alignment = WD_ALIGN_PARAGRAPH.LEFT
                        for r in prg.runs:
                            r.font.name = 'Times New Roman'
                            r.font.size = Pt(9.5)
                            
                    # Bottom border on last row only (APA style)
                    if row_idx == n_rows - 1:
                        _set_cell_border(cell, bottom={"sz": 8, "val": "single", "color": "000000"},
                                              left={"val": "nil"}, right={"val": "nil"})
                    else:
                        _set_cell_border(cell, left={"val": "nil"}, right={"val": "nil"}, top={"val": "nil"}, bottom={"val": "nil"})

            doc.add_paragraph() # spacing after table
            
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer


def create_excel_report(sheets_dict: dict) -> io.BytesIO:
    """
    Exports multiple dataframes to Excel sheets.
    sheets_dict: {"Thống kê mô tả": df_desc, "Cronbach Alpha": df_alpha, ...}
    """
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        for sheet_name, df in sheets_dict.items():
            if df is not None and not df.empty:
                # Truncate sheet name to 31 chars (Excel limit)
                safe_name = sheet_name[:30]
                df.to_excel(writer, sheet_name=safe_name, index=False)
                
    buffer.seek(0)
    return buffer


def df_to_styled_excel(df_or_styler, sheet_name: str = "Data") -> bytes:
    """
    Chuyển đổi DataFrame hoặc Styler sang tệp Excel (.xlsx) với định dạng bảng biểu
    chuẩn nghiên cứu khoa học: Header xanh đậm, chữ trắng, kẻ viền ô, căn chỉnh độ rộng cột tự động.
    """
    df = getattr(df_or_styler, 'data', df_or_styler)
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception:
            return b""

    if df.empty:
        return b""

    # Xử lý Index nếu không phải RangeIndex thông thường
    if not isinstance(df.index, pd.RangeIndex):
        df_export = df.reset_index()
        if 'index' in df_export.columns and (df.index.name is None or str(df.index.name).strip() in ('', 'None')):
            first_col_name = "Chỉ số / Biến"
            df_export = df_export.rename(columns={'index': first_col_name})
    else:
        df_export = df.copy()

    safe_sheet = re.sub(r'[\\/*?:\[\]]', '_', str(sheet_name))[:30] or "Data"
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_export.to_excel(writer, sheet_name=safe_sheet, index=False)
        try:
            from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
            from openpyxl.utils import get_column_letter

            wb = writer.book
            ws = wb[safe_sheet]
            ws.views.sheetView[0].showGridLines = True

            header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
            header_fill = PatternFill(start_color="1E3A8A", end_color="1E3A8A", fill_type="solid")
            thin_border = Border(
                left=Side(style='thin', color='CBD5E1'),
                right=Side(style='thin', color='CBD5E1'),
                top=Side(style='thin', color='CBD5E1'),
                bottom=Side(style='thin', color='CBD5E1')
            )
            data_font = Font(name="Calibri", size=10, color="0F172A")

            # Style header row
            for cell in ws[1]:
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

            # Style data cells
            for row in ws.iter_rows(min_row=2):
                for cell in row:
                    cell.font = data_font
                    cell.border = thin_border
                    if isinstance(cell.value, (int, float)):
                        cell.alignment = Alignment(horizontal="right", vertical="center")
                    else:
                        cell.alignment = Alignment(horizontal="left", vertical="center")

            # Auto column width
            for col in ws.columns:
                col_letter = get_column_letter(col[0].column)
                max_len = max(len(str(cell.value or '')) for cell in col[:100])
                ws.column_dimensions[col_letter].width = min(max(max_len + 3, 12), 50)
        except Exception:
            pass

    output.seek(0)
    return output.getvalue()
