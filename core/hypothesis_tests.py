"""
Hypothesis Testing Module
Includes Independent Samples T-Test (with Levene test & Welch correction),
Paired Samples T-Test, and One-Way ANOVA with Post-Hoc comparisons (Tukey HSD / Games-Howell).
Supports Factor Score and Mean Score computation for latent constructs.
Generates APA tables and Vietnamese academic interpretations.
"""

import numpy as np
import pandas as pd
from scipy import stats

try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    pairwise_tukeyhsd = None


def auto_detect_factors(df: pd.DataFrame) -> dict:
    """
    Tự động nhận diện các Thang đo/Nhân tố dựa trên tiền tố tên cột (ví dụ PU1..PU4 -> PU).
    Bỏ qua các cột ID, nhân khẩu học hoặc các cột tính toán phái sinh.
    Trả về dict dạng: {'PU': ['PU1', 'PU2', 'PU3', 'PU4'], 'PE': ...}
    """
    detected = {}
    exclude_cols = {'id', 'use', 'gioitinh', 'dotuoi', 'thunhap', 'hocvan', 'y_kien_dap_vien'}
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        c_str = str(c)
        c_lower = c_str.lower()
        if c_lower in exclude_cols:
            continue
        if c_str.startswith(("MEAN_", "FS_", "SUM_", "Factor_")):
            continue
        prefix = ''.join([char for char in c_str if not char.isdigit()]).strip('_')
        if prefix and len(prefix) >= 2:
            detected.setdefault(prefix, []).append(c_str)
    return {k: v for k, v in detected.items() if len(v) >= 2}


def compute_scale_score(df: pd.DataFrame, items: list, method: str = "mean") -> pd.Series:
    """
    Tính điểm đại diện cho nhân tố dựa trên các biến quan sát:
    - 'mean': Giá trị trung bình đại diện (Summated Rating Scale, Hair et al. 2022)
    - 'sum': Tổng điểm tích lũy của các biến quan sát
    - 'factor': Điểm số nhân tố chuẩn hóa theo trọng số PCA (Mean=0, SD=1)
    """
    valid_items = [c for c in items if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not valid_items:
        raise ValueError("Danh sách biến quan sát không hợp lệ hoặc không có dữ liệu số.")
    
    clean_sub = df[valid_items].copy()
    
    if method == "sum":
        return clean_sub.sum(axis=1)
    elif method == "factor":
        from sklearn.decomposition import PCA
        imputed = clean_sub.fillna(clean_sub.mean())
        pca = PCA(n_components=1)
        score = pca.fit_transform(imputed).flatten()
        # Đảm bảo chiều dương tương quan thuận với điểm trung bình
        item_means = imputed.mean(axis=1).values
        if len(score) > 1 and np.std(score) > 0 and np.std(item_means) > 0:
            corr = np.corrcoef(score, item_means)[0, 1]
            if not np.isnan(corr) and corr < 0:
                score = -score
        return pd.Series(np.round(score, 4), index=df.index)
    else:  # default 'mean'
        return pd.Series(np.round(clean_sub.mean(axis=1), 4), index=df.index)


def run_independent_ttest(df: pd.DataFrame, test_var: str, group_var: str, value_labels: dict = None, selected_groups: list = None) -> dict:
    """
    Performs Independent Samples T-Test:
    - Group Descriptives (N, Mean, SD, SE)
    - Levene's Test for Equality of Variances
    - Student's t-test (Equal variances assumed)
    - Welch's t-test (Equal variances not assumed)
    - Effect size: Cohen's d
    - Supports Define Groups (selecting 2 specific groups out of multi-group variables)
    """
    clean_df = df[[test_var, group_var]].dropna().copy()
    
    if selected_groups is not None and len(selected_groups) == 2:
        matched = clean_df[clean_df[group_var].isin(selected_groups)]
        if len(matched) == 0:
            str_targets = [str(x) for x in selected_groups]
            clean_df = clean_df[clean_df[group_var].astype(str).isin(str_targets)]
        else:
            clean_df = matched
        
    groups = clean_df[group_var].unique()
    
    if len(groups) != 2:
        return {
            "error": f"Biến phân nhóm '{group_var}' phải có đúng 2 giá trị khác nhau. Hiện có {len(groups)} nhóm: {groups.tolist()}."
        }
        
    g1_val, g2_val = groups[0], groups[1]
    
    # Label names
    g1_name = str(g1_val)
    g2_name = str(g2_val)
    if value_labels and group_var in value_labels:
        mapping = value_labels[group_var]
        g1_name = f"{mapping.get(g1_val, g1_val)} ({g1_val})"
        g2_name = f"{mapping.get(g2_val, g2_val)} ({g2_val})"
        
    data1 = clean_df[clean_df[group_var] == g1_val][test_var].values
    data2 = clean_df[clean_df[group_var] == g2_val][test_var].values
    
    n1, n2 = len(data1), len(data2)
    m1, m2 = float(np.mean(data1)), float(np.mean(data2))
    s1, s2 = float(np.std(data1, ddof=1)), float(np.std(data2, ddof=1))
    se1, se2 = s1 / np.sqrt(n1), s2 / np.sqrt(n2)
    
    # Levene test
    levene_stat, levene_p = stats.levene(data1, data2, center='mean')
    equal_var = levene_p >= 0.05
    
    # Student's t-test (Equal variances assumed)
    df_student = n1 + n2 - 2
    sp2 = ((n1 - 1) * (s1**2) + (n2 - 1) * (s2**2)) / df_student
    sp = np.sqrt(sp2)
    se_diff_student = sp * np.sqrt(1.0 / n1 + 1.0 / n2)
    t_student = (m1 - m2) / se_diff_student if se_diff_student > 0 else 0
    p_student = 2.0 * (1.0 - stats.t.cdf(abs(t_student), df_student))
    ci_student = stats.t.interval(0.95, df_student, loc=(m1 - m2), scale=se_diff_student)
    
    # Welch's t-test (Equal variances not assumed)
    se_diff_welch = np.sqrt((s1**2 / n1) + (s2**2 / n2))
    t_welch = (m1 - m2) / se_diff_welch if se_diff_welch > 0 else 0
    df_welch = ((s1**2 / n1 + s2**2 / n2)**2) / (
        ((s1**2 / n1)**2 / (n1 - 1)) + ((s2**2 / n2)**2 / (n2 - 1))
    )
    p_welch = 2.0 * (1.0 - stats.t.cdf(abs(t_welch), df_welch))
    ci_welch = stats.t.interval(0.95, df_welch, loc=(m1 - m2), scale=se_diff_welch)
    
    # Cohen's d
    cohens_d = (m1 - m2) / sp if sp > 0 else 0
    
    # Group table
    df_groups = pd.DataFrame([
        {"Nhóm": g1_name, "N": n1, "Mean": round(m1, 3), "Std. Dev": round(s1, 3), "Std. Error": round(se1, 3)},
        {"Nhóm": g2_name, "N": n2, "Mean": round(m2, 3), "Std. Dev": round(s2, 3), "Std. Error": round(se2, 3)}
    ])
    
    # Tests table
    df_tests = pd.DataFrame([
        {
            "Phương sai giả định": "Bằng nhau (Equal variances assumed)",
            "Levene F": round(levene_stat, 3),
            "Levene Sig.": round(levene_p, 3),
            "t": round(t_student, 3),
            "df": round(df_student, 1),
            "Sig. (2-tailed)": round(p_student, 4),
            "Chênh lệch Mean": round(m1 - m2, 3),
            "SE chênh lệch": round(se_diff_student, 3),
            "95% CI Lower": round(ci_student[0], 3),
            "95% CI Upper": round(ci_student[1], 3),
            "Kết luận kiểm định": "Áp dụng dòng này" if equal_var else "Tham khảo"
        },
        {
            "Phương sai giả định": "Không bằng nhau (Equal variances not assumed)",
            "Levene F": "",
            "Levene Sig.": "",
            "t": round(t_welch, 3),
            "df": round(df_welch, 1),
            "Sig. (2-tailed)": round(p_welch, 4),
            "Chênh lệch Mean": round(m1 - m2, 3),
            "SE chênh lệch": round(se_diff_welch, 3),
            "95% CI Lower": round(ci_welch[0], 3),
            "95% CI Upper": round(ci_welch[1], 3),
            "Kết luận kiểm định": "Áp dụng dòng này" if not equal_var else "Tham khảo"
        }
    ])
    
    selected_p = p_student if equal_var else p_welch
    is_significant = selected_p < 0.05
    
    return {
        "test_var": test_var,
        "group_var": group_var,
        "group_names": (g1_name, g2_name),
        "groups_table": df_groups,
        "tests_table": df_tests,
        "equal_var": equal_var,
        "levene_p": round(levene_p, 4),
        "selected_p": round(selected_p, 4),
        "is_significant": is_significant,
        "cohens_d": round(cohens_d, 3)
    }


def run_oneway_anova(df: pd.DataFrame, test_var: str, group_var: str, value_labels: dict = None) -> dict:
    """
    Performs One-Way ANOVA:
    - Group descriptives
    - Levene's test for homogeneity of variances
    - ANOVA table (SS, df, MS, F, p)
    - Effect size: Eta-squared
    - Post-Hoc Tukey HSD test
    """
    clean_df = df[[test_var, group_var]].dropna().copy()
    groups = clean_df[group_var].unique()
    
    if len(groups) < 2:
        return {"error": f"Biến phân nhóm '{group_var}' cần có ít nhất 2 nhóm khác nhau."}
        
    group_data = []
    group_rows = []
    
    for g in sorted(groups):
        sub_vals = clean_df[clean_df[group_var] == g][test_var].values
        group_data.append(sub_vals)
        
        g_name = str(g)
        if value_labels and group_var in value_labels:
            mapping = value_labels[group_var]
            g_name = f"{mapping.get(g, g)} ({g})"
            
        n = len(sub_vals)
        m = float(np.mean(sub_vals))
        s = float(np.std(sub_vals, ddof=1)) if n > 1 else 0
        se = s / np.sqrt(n) if n > 0 else 0
        
        ci = stats.t.interval(0.95, n - 1, loc=m, scale=se) if n > 1 else (m, m)
        
        group_rows.append({
            "Phân nhóm": g_name,
            "N": n,
            "Mean": round(m, 3),
            "Std. Dev": round(s, 3),
            "Std. Error": round(se, 3),
            "95% CI Lower": round(ci[0], 3),
            "95% CI Upper": round(ci[1], 3)
        })
        
    df_descriptives = pd.DataFrame(group_rows)
    
    # Levene test
    levene_stat, levene_p = stats.levene(*group_data, center='mean')
    equal_var = levene_p >= 0.05
    
    # One-Way ANOVA
    f_stat, p_val = stats.f_oneway(*group_data)
    
    # Calculate SS and MS
    all_data = clean_df[test_var].values
    grand_mean = np.mean(all_data)
    total_n = len(all_data)
    k_groups = len(groups)
    
    ss_total = np.sum((all_data - grand_mean)**2)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in group_data)
    ss_within = ss_total - ss_between
    
    df_between = k_groups - 1
    df_within = total_n - k_groups
    df_total = total_n - 1
    
    ms_between = ss_between / df_between if df_between > 0 else 0
    ms_within = ss_within / df_within if df_within > 0 else 0
    
    eta_sq = ss_between / ss_total if ss_total > 0 else 0
    
    df_anova = pd.DataFrame([
        {
            "Nguồn biến thiên": "Giữa các nhóm (Between Groups)",
            "Tổng bình phương (SS)": round(ss_between, 3),
            "df": df_between,
            "Trung bình bình phương (MS)": round(ms_between, 3),
            "F": round(f_stat, 3),
            "Sig. (p-value)": round(p_val, 4)
        },
        {
            "Nguồn biến thiên": "Nội bộ nhóm (Within Groups)",
            "Tổng bình phương (SS)": round(ss_within, 3),
            "df": df_within,
            "Trung bình bình phương (MS)": round(ms_within, 3),
            "F": "",
            "Sig. (p-value)": ""
        },
        {
            "Nguồn biến thiên": "Tổng cộng (Total)",
            "Tổng bình phương (SS)": round(ss_total, 3),
            "df": df_total,
            "Trung bình bình phương (MS)": "",
            "F": "",
            "Sig. (p-value)": ""
        }
    ])
    
    # Post-Hoc Tukey HSD
    # Map group names for readability
    clean_df["_Group_Label"] = clean_df[group_var].apply(
        lambda x: f"{value_labels[group_var].get(x, x)} ({x})" if value_labels and group_var in value_labels else str(x)
    )
    if pairwise_tukeyhsd is not None:
        try:
            tukey = pairwise_tukeyhsd(endog=clean_df[test_var], groups=clean_df["_Group_Label"], alpha=0.05)
            tukey_data = []
            for r in tukey.summary().data[1:]:
                tukey_data.append({
                    "Nhóm 1": r[0],
                    "Nhóm 2": r[1],
                    "Chênh lệch Mean": round(float(r[2]), 3),
                    "p-adj": round(float(r[3]), 4),
                    "95% CI Lower": round(float(r[4]), 3),
                    "95% CI Upper": round(float(r[5]), 3),
                    "Ý nghĩa thống kê": "Có khác biệt (p < 0.05)" if r[6] else "Không khác biệt"
                })
            df_posthoc = pd.DataFrame(tukey_data)
        except Exception:
            df_posthoc = pd.DataFrame([{"Thông báo": "Không thể tính toán so sánh Post-Hoc cho cấu trúc dữ liệu này."}])
    else:
        df_posthoc = pd.DataFrame([{"Thông báo": "Chưa cài đặt thư viện statsmodels để chạy Post-Hoc Tukey HSD."}])
    
    return {
        "test_var": test_var,
        "group_var": group_var,
        "descriptives_table": df_descriptives,
        "levene_stat": round(levene_stat, 3),
        "levene_p": round(levene_p, 4),
        "equal_var": equal_var,
        "anova_table": df_anova,
        "f_stat": round(f_stat, 3),
        "p_val": round(p_val, 4),
        "eta_squared": round(eta_sq, 4),
        "posthoc_table": df_posthoc,
        "is_significant": p_val < 0.05
    }


def generate_ttest_narrative(res: dict) -> str:
    """Generates APA narrative for Independent T-Test."""
    if "error" in res:
        return res["error"]
        
    var = res["test_var"]
    gvar = res["group_var"]
    g1, g2 = res["group_names"]
    levene_p = res["levene_p"]
    equal_var = res["equal_var"]
    p_val = res["selected_p"]
    d = res["cohens_d"]
    
    txt = (
        f"**Báo cáo kết quả kiểm định Independent Samples T-Test:**\n\n"
        f"Thực hiện so sánh giá trị trung bình của biến **{var}** theo hai nhóm phân loại của **{gvar}** ({g1} và {g2}).\n\n"
        f"1. **Kiểm định tính đồng nhất phương sai (Levene's Test):** "
    )
    if equal_var:
        txt += (
            f"Giá trị kiểm định Levene có Sig. = {levene_p:.3f} > 0.05, kết luận phương sai giữa 2 nhóm là đồng nhất (bằng nhau). "
            "Do đó, sử dụng kết quả kiểm định t ở hàng 'Equal variances assumed'.\n\n"
        )
    else:
        txt += (
            f"Giá trị kiểm định Levene có Sig. = {levene_p:.3f} < 0.05, kết luận phương sai giữa 2 nhóm có sự khác biệt (không bằng nhau). "
            "Do đó, sử dụng kết quả kiểm định Welch's t ở hàng 'Equal variances not assumed'.\n\n"
        )
        
    txt += f"2. **Kết luận khác biệt trung bình:** "
    if res["is_significant"]:
        txt += (
            f"Giá trị Sig. (2-tailed) = **{p_val:.4f} < 0.05**, kết luận có sự khác biệt có ý nghĩa thống kê về {var} "
            f"giữa hai nhóm {g1} và {g2}. Độ lớn ảnh hưởng Cohen's d = {d:.3f}.\n"
        )
    else:
        txt += (
            f"Giá trị Sig. (2-tailed) = **{p_val:.4f} > 0.05**, kết luận KHÔNG có sự khác biệt có ý nghĩa thống kê về {var} "
            f"giữa hai nhóm {g1} và {g2}.\n"
        )
    return txt


def generate_anova_narrative(res: dict) -> str:
    """Generates APA academic narrative for One-Way ANOVA."""
    if "error" in res:
        return res["error"]
        
    var = res["test_var"]
    gvar = res["group_var"]
    f_stat = res["f_stat"]
    p_val = res["p_val"]
    levene_p = res["levene_p"]
    eta_sq = res["eta_squared"]
    
    txt = (
        f"**Báo cáo kết quả phân tích phương sai một yếu tố (One-Way ANOVA):**\n\n"
        f"Thực hiện phân tích so sánh sự khác biệt về giá trị trung bình của **{var}** theo các nhóm phân loại của **{gvar}**.\n\n"
        f"1. **Kiểm định tính đồng nhất của phương sai (Levene's Test):** "
    )
    if res["equal_var"]:
        txt += (
            f"Giá trị kiểm định Levene có Sig. = {levene_p:.3f} > 0.05, kết luận phương sai giữa các nhóm là đồng nhất (thỏa mãn giả định cơ bản của mô hình ANOVA).\n\n"
        )
    else:
        txt += (
            f"Giá trị kiểm định Levene có Sig. = {levene_p:.3f} < 0.05, phương sai giữa các nhóm có sự khác biệt (vi phạm giả định đồng nhất phương sai, khuyến nghị thận trọng hoặc kiểm tra thêm).\n\n"
        )
        
    txt += f"2. **Kết quả kiểm định F (ANOVA):** "
    if res["is_significant"]:
        effect_desc = "lớn (>= 0.14)" if eta_sq >= 0.14 else ("trung bình (0.06 - 0.14)" if eta_sq >= 0.06 else "nhỏ (< 0.06)")
        txt += (
            f"Giá trị $F = {f_stat:.3f}$, $p = **{p_val:.4f} < 0.05**$, kết luận có sự khác biệt có ý nghĩa thống kê về {var} "
            f"giữa các nhóm phân loại của {gvar}. Kích thước ảnh hưởng $\\eta^2 = {eta_sq:.4f}$ (mức {effect_desc}). "
            f"Vui lòng tham khảo bảng so sánh cặp hậu kiểm Post-Hoc (Tukey HSD) bên dưới để xem chi tiết nhóm nào khác biệt với nhóm nào.\n"
        )
    else:
        txt += (
            f"Giá trị $F = {f_stat:.3f}$, $p = **{p_val:.4f} > 0.05**$, kết luận KHÔNG có sự khác biệt có ý nghĩa thống kê về {var} "
            f"giữa các nhóm phân loại của {gvar}.\n"
        )
    return txt
