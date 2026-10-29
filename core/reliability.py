"""
Scale Reliability Module (Cronbach's Alpha)
Computes overall Alpha, Item-Total Correlations, Alpha if Item Deleted.
Provides academic thresholds (Nunnally & Bernstein 1994, Hair et al. 2022)
and auto-detection of poor items.
"""

import numpy as np
import pandas as pd


def compute_cronbach_alpha(items_df: pd.DataFrame) -> float:
    """Computes basic Cronbach's Alpha for a dataframe of items."""
    k = items_df.shape[1]
    if k <= 1:
        return 0.0
    item_vars = items_df.var(axis=0, ddof=1)
    total_var = items_df.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return 0.0
    alpha = (k / (k - 1.0)) * (1.0 - (item_vars.sum() / total_var))
    return float(alpha)


def analyze_scale_reliability(df: pd.DataFrame, items: list, scale_name: str = "Thang đo", labels_dict: dict = None) -> dict:
    """
    Performs full reliability analysis on the selected scale items:
    - Overall Cronbach's Alpha
    - Scale Mean if Item Deleted
    - Scale Variance if Item Deleted
    - Corrected Item-Total Correlation
    - Cronbach's Alpha if Item Deleted
    - Recommendations / Action Flags
    """
    data = df[items].dropna().copy()
    n_samples, k_items = data.shape
    
    if k_items < 2:
        return {
            "error": "Cần ít nhất 2 biến quan sát để tính độ tin cậy thang đo Cronbach's Alpha."
        }
        
    overall_alpha = compute_cronbach_alpha(data)
    total_score = data.sum(axis=1)
    
    rows = []
    bad_items = []
    
    for col in items:
        # Scale with item removed
        remaining_items = [c for c in items if c != col]
        sub_df = data[remaining_items]
        sub_total = sub_df.sum(axis=1)
        
        # Mean if deleted
        mean_deleted = sub_total.mean()
        # Variance if deleted
        var_deleted = sub_total.var(ddof=1)
        
        # Corrected Item-Total Correlation
        item_series = data[col]
        # Correlation between item and sum of remaining items
        cov_matrix = np.cov(item_series, sub_total)
        cov_val = cov_matrix[0, 1]
        std_item = item_series.std(ddof=1)
        std_sub = sub_total.std(ddof=1)
        if std_item > 0 and std_sub > 0:
            item_total_corr = cov_val / (std_item * std_sub)
        else:
            item_total_corr = 0.0
            
        # Alpha if deleted
        alpha_deleted = compute_cronbach_alpha(sub_df)
        
        # Academic rule: item_total_corr >= 0.30
        is_acceptable = item_total_corr >= 0.30
        causes_higher_alpha = alpha_deleted > (overall_alpha + 0.02)
        
        if not is_acceptable:
            status = "Loại bỏ (Tương quan < 0.30)"
            bad_items.append(col)
        elif causes_higher_alpha and overall_alpha < 0.8:
            status = "Cân nhắc loại (Alpha tăng đáng kể)"
        else:
            status = "Đạt tiêu chuẩn (Giữ lại)"
            
        var_label = labels_dict.get(col, col) if labels_dict else col

        rows.append({
            "Biến quan sát": col,
            "Nội dung / Nhãn": var_label,
            "Trung bình nếu loại biến": round(mean_deleted, 3),
            "Phương sai nếu loại biến": round(var_deleted, 3),
            "Tương quan biến - tổng": round(item_total_corr, 3),
            "Alpha nếu loại biến": round(alpha_deleted, 3),
            "Đánh giá & Khuyến nghị": status
        })
        
    df_items = pd.DataFrame(rows)
    
    # Scale level assessment
    if overall_alpha >= 0.9:
        scale_quality = "Rất tốt (Excellent, >= 0.90)"
    elif overall_alpha >= 0.8:
        scale_quality = "Tốt (Good, 0.80 - 0.89)"
    elif overall_alpha >= 0.7:
        scale_quality = "Chấp nhận được (Acceptable, 0.70 - 0.79)"
    elif overall_alpha >= 0.6:
        scale_quality = "Khả dụng trong nghiên cứu sơ bộ (Questionable, 0.60 - 0.69)"
    else:
        scale_quality = "Không đạt độ tin cậy (< 0.60, Unacceptable)"
        
    return {
        "scale_name": scale_name,
        "n_samples": n_samples,
        "k_items": k_items,
        "overall_alpha": round(overall_alpha, 3),
        "scale_quality": scale_quality,
        "items_table": df_items,
        "bad_items": bad_items,
        "is_valid": overall_alpha >= 0.7 and len(bad_items) == 0
    }


def generate_reliability_narrative(res: dict) -> str:
    """Generates APA-style narrative for scale reliability results."""
    if "error" in res:
        return res["error"]
        
    scale = res["scale_name"]
    alpha = res["overall_alpha"]
    k = res["k_items"]
    n = res["n_samples"]
    quality = res["scale_quality"]
    bad = res["bad_items"]
    
    txt = (
        f"**Kết quả kiểm định độ tin cậy thang đo {scale}:**\n\n"
        f"Kiểm định Cronbach's Alpha cho thang đo **{scale}** gồm {k} biến quan sát (cỡ mẫu N = {n}) "
        f"cho kết quả hệ số Cronbach's Alpha tổng thể đạt **{alpha:.3f}**, "
        f"được đánh giá ở mức **{quality}** theo tiêu chuẩn của Nunnally & Bernstein (1994) và Hair et al. (2022).\n\n"
    )
    
    if len(bad) == 0:
        txt += (
            "Xem xét chỉ số tương quan biến - tổng (Corrected Item-Total Correlation), tất cả các biến quan sát "
            f"đều có giá trị lớn hơn 0.30 (dao động từ {res['items_table']['Tương quan biến - tổng'].min():.3f} đến "
            f"{res['items_table']['Tương quan biến - tổng'].max():.3f}). "
            "Đồng thời, không có biến nào mà khi loại bỏ sẽ làm tăng đáng kể độ tin cậy chung của thang đo. "
            f"Do đó, toàn bộ {k} biến quan sát của thang đo {scale} được giữ lại để đưa vào bước phân tích nhân tố EFA tiếp theo."
        )
    else:
        bad_str = ", ".join(bad)
        txt += (
            f"**Cảnh báo:** Có {len(bad)} biến quan sát ({bad_str}) có hệ số tương quan biến - tổng nhỏ hơn 0.30 "
            "không đáp ứng tiêu chuẩn loại biến rác trong nghiên cứu khoa học. "
            "Khuyến nghị nghiên cứu viên nên loại bỏ các biến này và chạy lại kiểm định độ tin cậy thang đo."
        )
    return txt
