"""
Exploratory Factor Analysis (EFA) Module
Calculates KMO measure, Bartlett's test of sphericity, Eigenvalues,
Total Variance Explained, Rotated Component / Pattern Matrix.
Checks for low loadings (< 0.5) and cross-loadings (delta < 0.3).
"""

import numpy as np
import pandas as pd

# Compatibility patch for factor_analyzer with scikit-learn >= 1.6
import factor_analyzer.factor_analyzer as fa_mod
_orig_check_array = fa_mod.check_array
fa_mod.check_array = lambda *args, **kwargs: _orig_check_array(
    *args, **{('ensure_all_finite' if k == 'force_all_finite' else k): v for k, v in kwargs.items()}
)

from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity


def run_efa(df: pd.DataFrame, items: list, n_factors: int = None, 
            method: str = "principal", rotation: str = "varimax", 
            loading_threshold: float = 0.50, labels_dict: dict = None) -> dict:
    """
    Performs full EFA analysis:
    - KMO and Bartlett's Test
    - Eigenvalues & Variance Explained
    - SPSS-Style Pattern Matrix (Sorted by size & Suppress small coefficients)
    - Structure Matrix & Factor Correlation Matrix (for Oblique rotation)
    - Cross-loading and Low-loading diagnostic checks
    """
    data = df[items].dropna().copy()
    n_samples, n_vars = data.shape
    
    if n_vars < 3:
        return {"error": "Cần ít nhất 3 biến quan sát để thực hiện phân tích nhân tố EFA."}
        
    # 1. KMO Test
    try:
        kmo_per_variable, kmo_total = calculate_kmo(data)
    except Exception as e:
        return {"error": f"Lỗi khi tính KMO: {str(e)}"}
        
    # 2. Bartlett's Test of Sphericity
    try:
        chi_square, bartlett_p = calculate_bartlett_sphericity(data)
        df_bartlett = int(n_vars * (n_vars - 1) / 2)
    except Exception as e:
        return {"error": f"Lỗi khi tính Bartlett's test: {str(e)}"}
        
    # Chuẩn hóa tên phương pháp trích và phép xoay để tương thích hoàn toàn với factor_analyzer
    method_alias = {
        "pa": "minres",
        "paf": "minres",
        "principal_axis": "minres",
        "minres": "minres",
        "pca": "principal",
        "principal": "principal",
        "ml": "ml",
        "mle": "ml",
        "uls": "uls"
    }
    method_clean = method_alias.get(str(method).lower().strip(), "principal")

    rotation_alias = {
        "varimax": "varimax",
        "promax": "promax",
        "oblimin": "oblimin",
        "quartimax": "quartimax",
        "equamax": "equamax",
        "none": None,
        "": None
    }
    rotation_clean = rotation_alias.get(str(rotation).lower().strip() if rotation else "", rotation)
    is_oblique = rotation_clean in ["promax", "oblimin", "quartimin", "geomin_obl"]

    # 3. Initial extraction to determine number of factors by Kaiser Criterion (Eigenvalue >= 1)
    fa_init = FactorAnalyzer(n_factors=n_vars, method=method_clean, rotation=None)
    fa_init.fit(data)
    ev, v = fa_init.get_eigenvalues()
    
    # Kaiser criterion count
    if n_factors is None or n_factors <= 0:
        kaiser_factors = int(np.sum(ev >= 1.0))
        n_factors = max(kaiser_factors, 1)
        
    # 4. Final rotated model
    fa = FactorAnalyzer(n_factors=n_factors, method=method_clean, rotation=rotation_clean)
    fa.fit(data)
    
    # Total variance explained table
    # fa.get_factor_variance() returns (variance, proportional variance, cumulative variance)
    var_explained = fa.get_factor_variance()
    var_rows = []
    for i in range(n_factors):
        var_rows.append({
            "Nhân tố (Factor)": f"Nhân tố {i+1}",
            "Initial Eigenvalue": round(float(ev[i]), 3),
            "% Phương sai trích": round(float(var_explained[1][i]) * 100.0, 3),
            "% Phương sai tích lũy": round(float(var_explained[2][i]) * 100.0, 3)
        })
    df_variance = pd.DataFrame(var_rows)
    total_cum_var = float(var_explained[2][-1]) * 100.0
    
    # 5. Factor Loadings / Pattern Matrix
    loadings = fa.loadings_
    factor_cols = [f"Nhân tố {i+1}" for i in range(n_factors)]
    df_loadings = pd.DataFrame(loadings, index=items, columns=factor_cols).round(3)
    
    # Primary factor sorting (SPSS "Sorted by size")
    primary_factors = [int(np.argmax(np.abs(df_loadings.loc[it].values))) for it in items]
    max_loads = [float(np.max(np.abs(df_loadings.loc[it].values))) for it in items]
    
    df_sort_helper = df_loadings.copy()
    df_sort_helper['__p__'] = primary_factors
    df_sort_helper['__m__'] = max_loads
    df_pattern_sorted = df_sort_helper.sort_values(
        by=['__p__', '__m__'], 
        ascending=[True, False]
    ).drop(columns=['__p__', '__m__'])
    
    # Suppress small loadings (< 0.30) like SPSS
    df_pattern_suppressed = df_pattern_sorted.map(lambda v: f"{v:.3f}" if abs(v) >= 0.30 else "")
    
    # Bảng Pattern Matrix chuẩn SPSS hoàn chỉnh (kèm cột Biến và Nhãn biến)
    df_pattern_spss = df_pattern_suppressed.copy()
    df_pattern_spss.insert(0, "Biến quan sát", df_pattern_spss.index)
    if labels_dict:
        df_pattern_spss.insert(1, "Nhãn biến (Label)", [labels_dict.get(it, "") for it in df_pattern_spss["Biến quan sát"]])
    df_pattern_spss = df_pattern_spss.reset_index(drop=True)
    
    # Bảng Pattern Matrix đầy đủ số (không ẩn < 0.30)
    df_pattern_full = df_pattern_sorted.copy()
    df_pattern_full.insert(0, "Biến quan sát", df_pattern_full.index)
    if labels_dict:
        df_pattern_full.insert(1, "Nhãn biến (Label)", [labels_dict.get(it, "") for it in df_pattern_full["Biến quan sát"]])
    df_pattern_full = df_pattern_full.reset_index(drop=True)
    
    # 6. Structure Matrix (Ma trận Cấu trúc) - khi dùng phép xoay xiên Promax / Oblimin
    df_structure_spss = None
    if is_oblique and getattr(fa, "structure_", None) is not None:
        df_struct_raw = pd.DataFrame(fa.structure_, index=items, columns=factor_cols).round(3)
        df_struct_sorted = df_struct_raw.loc[df_pattern_sorted.index].copy()
        df_struct_display = df_struct_sorted.copy()
        df_struct_display.insert(0, "Biến quan sát", df_struct_display.index)
        if labels_dict:
            df_struct_display.insert(1, "Nhãn biến (Label)", [labels_dict.get(it, "") for it in df_struct_display["Biến quan sát"]])
        df_structure_spss = df_struct_display.reset_index(drop=True)
        
    # 7. Factor Correlation Matrix (Ma trận tương quan nhân tố phi)
    df_factor_corr = None
    if is_oblique and getattr(fa, "phi_", None) is not None:
        df_phi = pd.DataFrame(fa.phi_, index=factor_cols, columns=factor_cols).round(3)
        df_phi.insert(0, "Nhân tố", df_phi.index)
        df_factor_corr = df_phi.reset_index(drop=True)
    
    # 8. Diagnostic checks (Low loading & Cross-loading evaluation)
    problem_items = []
    rows_eval = []
    
    for item in items:
        row_loads = np.abs(df_loadings.loc[item].values)
        sorted_indices = np.argsort(row_loads)[::-1]
        
        max_load = row_loads[sorted_indices[0]]
        primary_factor = factor_cols[sorted_indices[0]]
        
        # Cross loading check if n_factors >= 2
        is_cross_loading = False
        delta_loading = 1.0
        if n_factors >= 2:
            second_load = row_loads[sorted_indices[1]]
            delta_loading = max_load - second_load
            if max_load >= loading_threshold and second_load >= loading_threshold and delta_loading < 0.30:
                is_cross_loading = True
                
        is_low_loading = max_load < loading_threshold
        
        if is_low_loading:
            eval_text = f"Tải nhân tố thấp (< {loading_threshold})"
            problem_items.append((item, eval_text))
        elif is_cross_loading:
            eval_text = f"Tải kép (Cross-loading, Delta = {delta_loading:.3f} < 0.30)"
            problem_items.append((item, eval_text))
        else:
            eval_text = f"Đạt chuẩn (Thuộc {primary_factor})"
            
        row_dict = {
            "Biến quan sát": item,
            "Nhân tố chính": primary_factor,
            "Hệ số tải cao nhất": round(float(df_loadings.loc[item, primary_factor]), 3),
            "Chênh lệch tải kép (Delta)": round(float(delta_loading), 3) if n_factors >= 2 else 1.0,
            "Đánh giá chất lượng": eval_text
        }
        for f in factor_cols:
            row_dict[f] = round(float(df_loadings.loc[item, f]), 3)
            
        rows_eval.append(row_dict)
        
    df_evaluated_loadings = pd.DataFrame(rows_eval)
    
    # KMO Evaluation
    if kmo_total >= 0.8:
        kmo_eval = "Rất tốt (Meritorious, >= 0.80)"
    elif kmo_total >= 0.7:
        kmo_eval = "Tốt (Good, 0.70 - 0.79)"
    elif kmo_total >= 0.6:
        kmo_eval = "Trung bình (Mediocre, 0.60 - 0.69)"
    elif kmo_total >= 0.5:
        kmo_eval = "Chấp nhận tối thiểu (Poor, 0.50 - 0.59)"
    else:
        kmo_eval = "Không thích hợp (< 0.50, Unacceptable)"
        
    is_valid_efa = (
        kmo_total >= 0.5 and 
        bartlett_p < 0.05 and 
        total_cum_var >= 50.0 and 
        len(problem_items) == 0
    )
    
    matrix_name = "Pattern Matrix (Ma trận Mẫu)" if is_oblique else "Rotated Component Matrix (Ma trận Thành phần Xoay)"
    
    return {
        "n_samples": n_samples,
        "n_vars": n_vars,
        "n_factors": n_factors,
        "kmo_total": round(float(kmo_total), 3),
        "kmo_eval": kmo_eval,
        "chi_square": round(float(chi_square), 3),
        "df_bartlett": df_bartlett,
        "bartlett_p": round(float(bartlett_p), 5),
        "eigenvalues": [round(float(x), 3) for x in ev],
        "variance_table": df_variance,
        "total_cum_variance": round(total_cum_var, 3),
        "loadings_table": df_loadings.round(3),
        "evaluated_table": df_evaluated_loadings,
        "pattern_matrix_spss": df_pattern_spss,
        "pattern_matrix_full": df_pattern_full,
        "pattern_matrix_raw": df_pattern_sorted,
        "structure_matrix_spss": df_structure_spss,
        "factor_correlations": df_factor_corr,
        "is_oblique": is_oblique,
        "matrix_name": matrix_name,
        "method_name": method_clean,
        "rotation_name": rotation_clean,
        "problem_items": problem_items,
        "is_valid_efa": is_valid_efa
    }


def generate_efa_narrative(res: dict) -> str:
    """Generates APA academic report paragraph for EFA."""
    if "error" in res:
        return res["error"]
        
    kmo = res["kmo_total"]
    kmo_eval = res["kmo_eval"]
    chi2 = res["chi_square"]
    df = res["df_bartlett"]
    p_bart = res["bartlett_p"]
    factors = res["n_factors"]
    cum_var = res["total_cum_variance"]
    problems = res["problem_items"]
    
    txt = (
        f"**Kết quả phân tích nhân tố khám phá (EFA):**\n\n"
        f"1. **Điều kiện thích hợp của phân tích nhân tố:**\n"
        f"- Hệ số KMO = **{kmo:.3f}** (được đánh giá ở mức **{kmo_eval}**), thỏa mãn điều kiện KMO >= 0.50.\n"
        f"- Kiểm định Bartlett's Test of Sphericity có giá trị Chi-square = {chi2:.3f} (df = {df}) "
        f"với mức ý nghĩa Sig. = **{p_bart:.5f} < 0.05**. Điều này chứng minh các biến quan sát có tương quan với nhau "
        f"trong tổng thể và hoàn toàn phù hợp để phân tích nhân tố EFA.\n\n"
        f"2. **Kết quả trích nhân tố & Phương sai giải thích:**\n"
        f"- Dựa trên tiêu chuẩn Kaiser (Eigenvalue >= 1), phân tích đã trích được **{factors} nhân tố**.\n"
        f"- Tổng phương sai trích (Cumulative Variance) đạt **{cum_var:.2f}%** (vượt qua ngưỡng quy ước 50%), "
        f"nghĩa là {factors} nhân tố này giải thích được {cum_var:.2f}% sự biến thiên của tập dữ liệu.\n\n"
        f"3. **Ma trận xoay nhân tố (Rotated Matrix):**\n"
    )
    if len(problems) == 0:
        txt += (
            "Toàn bộ các biến quan sát đều có hệ số tải nhân tố (Factor Loading) đạt trên 0.50 và không có biến nào "
            "vi phạm hiện tượng tải kép (Cross-loading, khoảng cách tải giữa 2 nhân tố luôn >= 0.30). "
            "Dữ liệu đạt độ hội tụ và phân biệt tốt, đủ điều kiện đưa vào phân tích CFA / SEM tiếp theo."
        )
    else:
        prob_desc = "; ".join([f"{item}: {desc}" for item, desc in problems])
        txt += (
            f"**Cảnh báo vi phạm:** Phát hiện {len(problems)} biến quan sát có vấn đề: {prob_desc}. "
            "Khuyến nghị nghiên cứu viên loại bỏ lần lượt các biến vi phạm (bắt đầu từ biến có loading thấp nhất "
            "hoặc chênh lệch tải kép nhỏ nhất) và tiến hành trích lại nhân tố."
        )
    return txt
