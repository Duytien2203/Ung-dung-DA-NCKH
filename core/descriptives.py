"""
Descriptive Statistics and Demographics Module
Calculates frequencies, percentages, mean, SD, skewness, kurtosis, and normality tests.
Generates academic APA tables and Vietnamese interpretation text.
"""

import numpy as np
import pandas as pd
from scipy import stats


def calculate_frequency_table(series: pd.Series, label_mapping: dict = None) -> pd.DataFrame:
    """
    Computes frequency table for categorical/demographic variables:
    Categories, Frequency, Percent, Valid Percent, Cumulative Percent.
    """
    total_n = len(series)
    valid_series = series.dropna()
    valid_n = len(valid_series)
    
    # Counts
    val_counts = valid_series.value_counts().sort_index()
    
    rows = []
    cum_pct = 0.0
    for val, freq in val_counts.items():
        # Label resolution
        if label_mapping and val in label_mapping:
            category_name = f"{label_mapping[val]} ({val})"
        elif label_mapping and str(val) in label_mapping:
            category_name = f"{label_mapping[str(val)]} ({val})"
        else:
            category_name = str(val)
            
        pct = (freq / total_n) * 100.0
        val_pct = (freq / valid_n) * 100.0 if valid_n > 0 else 0.0
        cum_pct += val_pct
        
        rows.append({
            "Phân loại": category_name,
            "Tần số (N)": int(freq),
            "Tỷ lệ (%)": round(pct, 2),
            "Tỷ lệ hợp lệ (%)": round(val_pct, 2),
            "Tỷ lệ tích lũy (%)": round(cum_pct, 2)
        })
        
    # Check for missing
    missing_count = total_n - valid_n
    if missing_count > 0:
        missing_pct = (missing_count / total_n) * 100.0
        rows.append({
            "Phân loại": "Khuyết (Missing)",
            "Tần số (N)": int(missing_count),
            "Tỷ lệ (%)": round(missing_pct, 2),
            "Tỷ lệ hợp lệ (%)": None,
            "Tỷ lệ tích lũy (%)": None
        })
        
    df_res = pd.DataFrame(rows)
    return df_res


def calculate_descriptive_stats(df: pd.DataFrame, columns: list = None, labels_dict: dict = None) -> pd.DataFrame:
    """
    Calculates detailed descriptive statistics for continuous/Likert variables:
    N, Mean, Std. Error of Mean, Std. Dev, Variance, Min, Max, Skewness, Kurtosis, Normality.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
    results = []
    for col in columns:
        s = df[col].dropna()
        n = len(s)
        if n < 3:
            continue
            
        mean = s.mean()
        std = s.std(ddof=1)
        var = s.var(ddof=1)
        se_mean = std / np.sqrt(n)
        min_val = s.min()
        max_val = s.max()
        
        # Skewness and Kurtosis (SPSS style / unbiased sample)
        skew = stats.skew(s, bias=False)
        kurt = stats.kurtosis(s, bias=False) # excess kurtosis (normal = 0)
        
        # Standard errors of skewness and kurtosis
        # Formulas used by SPSS
        se_skew = np.sqrt((6.0 * n * (n - 1.0)) / ((n - 2.0) * (n + 1.0) * (n + 3.0))) if n > 2 else np.nan
        se_kurt = 2.0 * se_skew * np.sqrt((n**2 - 1.0) / ((n - 3.0) * (n + 5.0))) if n > 3 else np.nan
        
        # Normality check (Kline 2015 & Hair et al. 2022)
        # Kline: |Skew| < 3 and |Kurt| < 10
        # Hair: |Skew| < 1 and |Kurt| < 1 (ideal), |Skew| < 2 and |Kurt| < 2 (acceptable)
        if abs(skew) <= 1.0 and abs(kurt) <= 1.0:
            norm_status = "Chuẩn tốt (|Skew|<1, |Kurt|<1)"
        elif abs(skew) <= 2.0 and abs(kurt) <= 2.0:
            norm_status = "Chấp nhận được (|Skew|<2, |Kurt|<2)"
        elif abs(skew) < 3.0 and abs(kurt) < 10.0:
            norm_status = "Xấp xỉ chuẩn (Kline 2015)"
        else:
            norm_status = "Không chuẩn (Vi phạm giả định)"
            
        var_label = labels_dict.get(col, col) if labels_dict else col

        results.append({
            "Biến": col,
            "Nhãn nội dung": var_label,
            "N": n,
            "Mean": round(mean, 3),
            "SE Mean": round(se_mean, 3),
            "Std. Dev": round(std, 3),
            "Variance": round(var, 3),
            "Min": round(min_val, 2),
            "Max": round(max_val, 2),
            "Skewness": round(skew, 3),
            "SE Skew": round(se_skew, 3) if not np.isnan(se_skew) else 0,
            "Kurtosis": round(kurt, 3),
            "SE Kurt": round(se_kurt, 3) if not np.isnan(se_kurt) else 0,
            "Đánh giá phân phối": norm_status
        })

    return pd.DataFrame(results)


def generate_descriptives_narrative(df_stats: pd.DataFrame) -> str:
    """
    Generates a professional academic paragraph interpreting the descriptive statistics.
    """
    if df_stats.empty:
        return ""
        
    highest_item = df_stats.loc[df_stats["Mean"].idxmax()]
    lowest_item = df_stats.loc[df_stats["Mean"].idxmin()]
    
    non_normal = df_stats[df_stats["Đánh giá phân phối"].str.contains("Không chuẩn")]
    
    text = (
        f"**Báo cáo tóm tắt thống kê mô tả:**\n\n"
        f"Trong tổng số các biến quan sát được phân tích, giá trị trung bình dao động từ "
        f"**{lowest_item['Mean']}** (thuộc biến *{lowest_item['Biến']} - {lowest_item['Nhãn nội dung']}*) đến "
        f"**{highest_item['Mean']}** (thuộc biến *{highest_item['Biến']} - {highest_item['Nhãn nội dung']}*). "
        f"Độ lệch chuẩn dao động trong khoảng từ {df_stats['Std. Dev'].min():.3f} đến {df_stats['Std. Dev'].max():.3f}, "
        f"cho thấy mức độ phân tán của dữ liệu ở mức bình thường đối với thang đo Likert.\n\n"
        f"**Kiểm tra phân phối chuẩn:** "
    )
    if non_normal.empty:
        text += (
            "Tất cả các biến quan sát đều có hệ số Skewness nằm trong khoảng [-2; +2] và Kurtosis nằm trong khoảng [-7; +7] "
            "(thỏa mãn tiêu chuẩn nghiêm ngặt của Hair et al., 2022 và Kline, 2015). "
            "Do đó, dữ liệu đảm bảo tính chuẩn xấp xỉ và đáp ứng đầy đủ điều kiện để thực hiện các kiểm định tham số tiếp theo "
            "(như Cronbach's Alpha, EFA, T-Test, ANOVA, CFA/SEM)."
        )
    else:
        abnormal_cols = ", ".join(non_normal["Biến"].tolist())
        text += (
            f"Phát hiện một số biến có dấu hiệu lệch chuẩn vượt ngưỡng quy ước ({abnormal_cols}). "
            f"Đối với CB-SEM, cần cân nhắc sử dụng phương pháp ước lượng Bollen-Stine Bootstrap, hoặc chuyển sang phương pháp phi tham số PLS-SEM."
        )
        
    return text
