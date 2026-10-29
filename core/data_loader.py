"""
Data Loader Module for Social Science Research (NCKH)
Supports Excel (.xlsx, .xls), CSV, and SPSS (.sav, .zsav) formats.
Preserves variable labels, value labels, and measurement scales.
"""

import io
import mimetypes
# Patch mimetypes for Windows registry sandbox safety
mimetypes.init = lambda *args, **kwargs: None
mimetypes.inited = True
if hasattr(mimetypes, "MimeTypes"):
    mimetypes.MimeTypes.read_windows_registry = lambda *args, **kwargs: None

import pandas as pd
import numpy as np
import pyreadstat


class DataLoader:
    def __init__(self):
        self.df = None
        self.raw_df = None
        self.original_df = None
        self.meta = None
        self.variable_labels = {}       # {col: "Label"}
        self.original_variable_labels = {}
        self.value_labels = {}          # {col: {1: "Nam", 2: "Nu"}}
        self.measure_types = {}         # {col: "scale"|"nominal"|"ordinal"}
        self.filename = ""
        self.file_type = ""

    def load_file(self, file_bytes: bytes, filename: str) -> pd.DataFrame:
        """
        Load data from bytes buffer with automatic format detection.
        Supports .sav, .zsav, .xlsx, .xls, .csv, and auto-detects by file signature
        even if the filename is missing an extension.
        """
        self.filename = filename
        ext = filename.lower().split('.')[-1] if '.' in filename else ''
        
        # Auto-detect if extension is missing or unrecognized
        if ext not in ['sav', 'zsav', 'xlsx', 'xls', 'csv']:
            if file_bytes.startswith(b"$FL2") or file_bytes.startswith(b"$FL3"):
                ext = 'sav'
            elif file_bytes.startswith(b"PK\x03\x04"):
                ext = 'xlsx'
            elif file_bytes.startswith(b"\xd0\xcf\x11\xe0"):
                ext = 'xls'
            else:
                # Fallback to csv
                ext = 'csv'

        self.file_type = ext

        if ext in ['sav', 'zsav']:
            try:
                # pyreadstat needs a file path or bytes buffer
                df, meta = pyreadstat.read_sav(io.BytesIO(file_bytes))
            except Exception:
                # Fallback: maybe it's an excel or csv misclassified
                try:
                    df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=0)
                    meta = None
                    ext = 'xlsx'
                    self.file_type = ext
                except Exception as e_inner:
                    raise ValueError(f"Không thể đọc tệp SPSS: {str(e_inner)}")

            df = df.dropna(how='all')
            df.columns = [str(c).strip() for c in df.columns]
            self.raw_df = df.copy()
            self.df = df
            self.meta = meta
            if meta:
                self.variable_labels = meta.column_labels if meta.column_labels else {}
                if meta.column_names and meta.column_labels:
                    self.variable_labels = dict(zip(meta.column_names, meta.column_labels))
                self.value_labels = meta.variable_value_labels if meta.variable_value_labels else {}
                self.measure_types = meta.variable_measure if meta.variable_measure else {}
            else:
                self.variable_labels = {c: str(c) for c in df.columns}
                self.value_labels = {}
                self.measure_types = {c: self._infer_measure_type(df[c]) for c in df.columns}
            
        elif ext in ['xlsx', 'xls']:
            try:
                excel_file = pd.ExcelFile(io.BytesIO(file_bytes))
                # Read first sheet
                df = pd.read_excel(excel_file, sheet_name=0)
            except Exception as e:
                # Try CSV fallback
                try:
                    df = pd.read_csv(io.BytesIO(file_bytes))
                    ext = 'csv'
                    self.file_type = ext
                except Exception:
                    raise ValueError(f"Không thể đọc tệp Excel: {str(e)}")

            # Drop completely empty rows and empty columns
            df = df.dropna(how='all').dropna(axis=1, how='all')
            df.columns = [str(c).strip() for c in df.columns]

            # Auto-clean numeric columns formatted as string
            for c in df.columns:
                if df[c].dtype == object:
                    try:
                        converted = pd.to_numeric(df[c].astype(str).str.replace(',', '.'), errors='coerce')
                        valid_orig = df[c].dropna()
                        if len(valid_orig) > 0 and (converted.notna().sum() / len(valid_orig)) >= 0.95:
                            df[c] = converted
                    except Exception:
                        pass

            self.raw_df = df.copy()
            self.df = df
            self.meta = None
            self.variable_labels = {c: str(c) for c in df.columns}
            self.value_labels = {}
            self.measure_types = {c: self._infer_measure_type(df[c]) for c in df.columns}
            
        elif ext == 'csv':
            # Try utf-8 then utf-8-sig or latin1
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), encoding='utf-8')
            except Exception:
                try:
                    df = pd.read_csv(io.BytesIO(file_bytes), encoding='utf-8-sig')
                except Exception:
                    df = pd.read_csv(io.BytesIO(file_bytes), encoding='latin1')

            df = df.dropna(how='all').dropna(axis=1, how='all')
            df.columns = [str(c).strip() for c in df.columns]

            for c in df.columns:
                if df[c].dtype == object:
                    try:
                        converted = pd.to_numeric(df[c].astype(str).str.replace(',', '.'), errors='coerce')
                        valid_orig = df[c].dropna()
                        if len(valid_orig) > 0 and (converted.notna().sum() / len(valid_orig)) >= 0.95:
                            df[c] = converted
                    except Exception:
                        pass

            self.raw_df = df.copy()
            self.df = df
            self.meta = None
            self.variable_labels = {c: str(c) for c in df.columns}
            self.value_labels = {}
            self.measure_types = {c: self._infer_measure_type(df[c]) for c in df.columns}
        else:
            raise ValueError(f"Định dạng tệp .{ext} không được hỗ trợ. Vui lòng tải lên file .sav, .xlsx, .xls hoặc .csv.")

        self.original_df = self.df.copy()
        self.original_variable_labels = self.variable_labels.copy()

        return self.df

    def reset_to_original(self) -> pd.DataFrame:
        """Khôi phục dữ liệu và nhãn về nguyên bản khi vừa nạp tệp."""
        if self.original_df is not None:
            self.df = self.original_df.copy()
            self.raw_df = self.original_df.copy()
            self.variable_labels = self.original_variable_labels.copy()
            self.measure_types = {c: self._infer_measure_type(self.df[c]) for c in self.df.columns}
        return self.df

    def set_cleaned_df(self, new_df: pd.DataFrame, new_variable_labels: dict = None, new_measure_types: dict = None):
        """Cập nhật dữ liệu đã làm sạch vào DataLoader để phục vụ toàn bộ quy trình tiếp theo."""
        self.df = new_df
        self.raw_df = new_df.copy()
        if new_variable_labels is not None:
            self.variable_labels = new_variable_labels
        if new_measure_types is not None:
            self.measure_types = new_measure_types
        else:
            self.measure_types = {c: self._infer_measure_type(self.df[c]) for c in self.df.columns}

    def _infer_measure_type(self, series: pd.Series) -> str:
        """Infer measure scale if not provided by SPSS metadata."""
        if pd.api.types.is_numeric_dtype(series):
            unique_count = series.nunique()
            if unique_count <= 7:
                return "ordinal"
            return "scale"
        return "nominal"

    def get_variable_info(self) -> pd.DataFrame:
        """
        Returns a DataFrame summarizing all variables:
        Column Name, Label, Inferred Type, Missing Values, Sample Values.
        """
        if self.df is None:
            return pd.DataFrame()

        rows = []
        for col in self.df.columns:
            series = self.df[col]
            label = self.variable_labels.get(col, col)
            val_labels = self.value_labels.get(col, {})
            measure = self.measure_types.get(col, self._infer_measure_type(series))
            missing_count = series.isna().sum()
            missing_pct = (missing_count / len(series)) * 100 if len(series) > 0 else 0

            # Formatted sample
            unique_vals = series.dropna().unique()
            if len(val_labels) > 0:
                sample_str = ", ".join([f"{k}={v}" for k, v in list(val_labels.items())[:3]])
                if len(val_labels) > 3:
                    sample_str += "..."
            else:
                sample_str = ", ".join([str(x) for x in unique_vals[:3]])
                if len(unique_vals) > 3:
                    sample_str += "..."

            rows.append({
                "Tên biến": col,
                "Nhãn biến (Label)": label,
                "Loại thang đo": measure.capitalize(),
                "Số dòng khuyết": missing_count,
                "Tỷ lệ khuyết (%)": f"{missing_pct:.1f}%",
                "Mẫu giá trị / Nhãn": sample_str
            })

        return pd.DataFrame(rows)

    def get_labeled_series(self, col: str) -> pd.Series:
        """
        Returns series with numeric codes replaced by their SPSS value labels (if available),
        useful for demographics charts and frequency tables.
        """
        series = self.df[col].copy()
        if col in self.value_labels and self.value_labels[col]:
            mapping = self.value_labels[col]
            return series.map(mapping).fillna(series.astype(str))
        return series

    def handle_missing(self, method: str = "listwise", cols: list = None) -> pd.DataFrame:
        """
        Handle missing values:
        - 'listwise': drop rows with any missing value in target columns
        - 'mean': impute with column mean (numeric only)
        - 'median': impute with column median (numeric only)
        """
        target_df = self.df if cols is None else self.df[cols]
        if method == "listwise":
            return target_df.dropna()
        elif method == "mean":
            res = target_df.copy()
            for c in res.select_dtypes(include=[np.number]).columns:
                res[c] = res[c].fillna(res[c].mean())
            return res
        elif method == "median":
            res = target_df.copy()
            for c in res.select_dtypes(include=[np.number]).columns:
                res[c] = res[c].fillna(res[c].median())
            return res
        return target_df
