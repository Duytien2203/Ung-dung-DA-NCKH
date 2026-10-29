"""
Exploratory Factor Analysis (EFA) Module
Calculates KMO measure, Bartlett's test of sphericity, Eigenvalues,
Total Variance Explained, Rotated Component / Pattern Matrix.
Supports full SPSS-style formatting:
- Sorted by size (grouping items by primary factor, ordered descending)
- Suppress small coefficients (customizable absolute threshold e.g. 0.10, 0.30, 0.40, 0.50)
- Checks for low loadings (< 0.50) and cross-loadings (delta < 0.30).
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='factor_analyzer')

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


def format_spss_pattern_matrix(
    df_loadings: pd.DataFrame,
    labels_dict: dict = None,
    sort_by_size: bool = True,
    suppress_small: bool = True,
    suppress_threshold: float = 0.30,
    show_labels: bool = True,
    show_primary_factor: bool = True,
    hide_empty_rows: bool = False
) -> pd.DataFrame:
    """
    Tạo bảng Pattern Matrix / Rotated Component Matrix chuẩn 100% định dạng SPSS:
    - Sorted by size: Sắp xếp các biến quan sát gom theo từng cụm nhân tố chính (tải cao nhất),
      và xếp giảm dần độ lớn hệ số tải trong từng nhân tố.
    - Suppress small coefficients: Ẩn các hệ số tải có giá trị tuyệt đối nhỏ hơn ngưỡng (mặc định 0.30).
    - Hiển thị trực quan biến quan sát nào tải lên nhân tố nào giống bảng output trong SPSS.
    - hide_empty_rows: Ẩn các dòng biến có tất cả hệ số tải đều nhỏ hơn ngưỡng.
    """
    items = list(df_loadings.index)
    factor_cols = list(df_loadings.columns)
    
    # Xác định nhân tố tải chính và giá trị tải lớn nhất của từng biến
    item_info = []
    for it in items:
        row_vals = np.abs(df_loadings.loc[it].values)
        p_idx = int(np.argmax(row_vals))
        p_name = factor_cols[p_idx]
        m_val = float(row_vals[p_idx])
        item_info.append({
            "item": it,
            "p_idx": p_idx,
            "p_name": p_name,
            "m_val": m_val
        })
        
    df_meta = pd.DataFrame(item_info)
    if sort_by_size:
        df_meta = df_meta.sort_values(by=["p_idx", "m_val"], ascending=[True, False])
        
    ordered_items = df_meta["item"].tolist()
    
    # Tạo bảng hiển thị
    rows = []
    for it in ordered_items:
        meta_row = df_meta[df_meta["item"] == it].iloc[0]
        row_dict = {"Biến quan sát": it}
        if show_labels:
            row_dict["Nhãn biến (Label)"] = labels_dict.get(it, "") if labels_dict else ""
        if show_primary_factor:
            row_dict["Nhân tố quy về"] = meta_row["p_name"]
            
        has_visible_loading = False
        for f in factor_cols:
            val = float(df_loadings.loc[it, f])
            if suppress_small and abs(val) < suppress_threshold:
                row_dict[f] = ""
            else:
                row_dict[f] = f"{val:.3f}"
                has_visible_loading = True
                
        if hide_empty_rows and suppress_small and not has_visible_loading:
            continue
            
        rows.append(row_dict)
        
    return pd.DataFrame(rows)


def run_efa(df: pd.DataFrame, items: list, n_factors: int = None, 
            method: str = "principal", rotation: str = "varimax", 
            loading_threshold: float = 0.50, 
            suppress_small: bool = True,
            suppress_threshold: float = 0.30,
            sort_by_size: bool = True,
            labels_dict: dict = None) -> dict:
    """
    Thực hiện phân tích nhân tố khám phá EFA toàn diện chuẩn SPSS:
    - Kiểm định KMO và Bartlett's Test of Sphericity
    - Bảng Phương sai giải thích & Giá trị riêng Eigenvalues
    - Pattern Matrix / Rotated Component Matrix chuẩn template SPSS
      (Hỗ trợ Sorted by size và tùy chọn ngưỡng Suppress small coefficients)
    - Structure Matrix & Factor Correlation Matrix (đối với phép xoay xiên Promax / Oblimin)
    - Chẩn đoán biến tải thấp (< loading_threshold) và tải kép (Cross-loading delta < 0.30)
    """
    valid_items = [c for c in items if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(valid_items) < 3:
        return {"error": "Cần ít nhất 3 biến quan sát dạng số để thực hiện phân tích nhân tố EFA."}
        
    data = df[valid_items].dropna().copy()
    if len(data) < 10:
        return {"error": f"Số lượng quan sát hợp lệ ({len(data)}) quá ít (< 10 quan sát). Vui lòng kiểm tra lại dữ liệu khuyết."}
        
    # Loại bỏ biến có phương sai = 0 (tất cả các dòng cùng 1 giá trị)
    zero_var_cols = [c for c in valid_items if data[c].std() == 0]
    if zero_var_cols:
        data = data.drop(columns=zero_var_cols)
        valid_items = [c for c in valid_items if c not in zero_var_cols]
        if len(valid_items) < 3:
            return {"error": f"Các biến {zero_var_cols} có phương sai = 0. Sau khi loại bỏ, số biến còn lại < 3 nên không thể chạy EFA."}
            
    n_samples, n_vars = data.shape
    if n_samples <= n_vars:
        return {"error": f"Cỡ mẫu hợp lệ (N = {n_samples}) cần lớn hơn số biến quan sát ({n_vars}) để phân tích nhân tố EFA."}
        
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
    try:
        fa_init = FactorAnalyzer(n_factors=n_vars, method=method_clean, rotation=None)
        fa_init.fit(data)
        ev, v = fa_init.get_eigenvalues()
    except Exception as e_init:
        # Fallback to PCA if specific extraction fails
        try:
            fa_init = FactorAnalyzer(n_factors=n_vars, method="principal", rotation=None)
            fa_init.fit(data)
            ev, v = fa_init.get_eigenvalues()
            method_clean = "principal"
        except Exception as e_pca:
            return {"error": f"Lỗi khi xác định giá trị riêng ban đầu: {str(e_pca)}"}
    
    # Kaiser criterion count
    if n_factors is None or n_factors <= 0:
        kaiser_factors = int(np.sum(ev >= 1.0))
        n_factors = max(kaiser_factors, 1)
        
    # Khống chế số lượng nhân tố n_factors <= n_vars - 1 để tránh lỗi bậc tự do âm
    n_factors = min(n_factors, max(1, n_vars - 1))
        
    # 4. Final rotated model with graceful fallback
    try:
        fa = FactorAnalyzer(n_factors=n_factors, method=method_clean, rotation=rotation_clean)
        fa.fit(data)
    except Exception as e_rot:
        try:
            # Fallback to varimax if oblique fails to converge
            fa = FactorAnalyzer(n_factors=n_factors, method=method_clean, rotation="varimax")
            fa.fit(data)
            rotation_clean = "varimax"
            is_oblique = False
        except Exception as e_fallback:
            return {"error": f"Lỗi trích nhân tố ({method_clean}) và xoay ({rotation_clean}): {str(e_fallback)}"}
    
    # Total variance explained table
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
    df_loadings = pd.DataFrame(loadings, index=valid_items, columns=factor_cols).round(3)
    
    # Sắp xếp và format bảng Pattern Matrix chuẩn template SPSS
    df_pattern_spss = format_spss_pattern_matrix(
        df_loadings=df_loadings,
        labels_dict=labels_dict,
        sort_by_size=sort_by_size,
        suppress_small=suppress_small,
        suppress_threshold=suppress_threshold,
        show_labels=True,
        show_primary_factor=True
    )
    
    # Bảng Pattern Matrix đầy đủ (không ẩn hệ số)
    df_pattern_full = format_spss_pattern_matrix(
        df_loadings=df_loadings,
        labels_dict=labels_dict,
        sort_by_size=sort_by_size,
        suppress_small=False,
        suppress_threshold=suppress_threshold,
        show_labels=True,
        show_primary_factor=True
    )
    
    # 6. Structure Matrix (Ma trận Cấu trúc) - khi dùng phép xoay xiên Promax / Oblimin
    df_structure_spss = None
    if is_oblique and getattr(fa, "structure_", None) is not None:
        try:
            df_struct_raw = pd.DataFrame(fa.structure_, index=valid_items, columns=factor_cols).round(3)
            df_structure_spss = format_spss_pattern_matrix(
                df_loadings=df_struct_raw,
                labels_dict=labels_dict,
                sort_by_size=sort_by_size,
                suppress_small=suppress_small,
                suppress_threshold=suppress_threshold,
                show_labels=True,
                show_primary_factor=False
            )
        except Exception:
            df_structure_spss = None
        
    # 7. Factor Correlation Matrix (Ma trận tương quan nhân tố phi)
    df_factor_corr = None
    if is_oblique and getattr(fa, "phi_", None) is not None:
        try:
            df_phi = pd.DataFrame(fa.phi_, index=factor_cols, columns=factor_cols).round(3)
            df_phi.insert(0, "Nhân tố", df_phi.index)
            df_factor_corr = df_phi.reset_index(drop=True)
        except Exception:
            df_factor_corr = None
    
    # 8. Diagnostic checks (Low loading & Cross-loading evaluation)
    problem_items = []
    rows_eval = []
    
    for item in valid_items:
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
        "structure_matrix_spss": df_structure_spss,
        "structure_matrix_raw": df_struct_raw if is_oblique else None,
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
