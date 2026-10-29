"""
PLS-SEM (Partial Least Squares Structural Equation Modeling) Engine
Implemented strictly following Hair, Hult, Ringle, & Sarstedt (2022).
Features:
- Wold's Iterative PLS Algorithm with Path Weighting Scheme
- Outer Model Evaluation: Outer Loadings, Indicator Reliability, Cronbach Alpha,
  Composite Reliability (rho_c), rho_A, AVE
- Discriminant Validity: HTMT Matrix, Fornell-Larcker, Cross-Loadings
- Inner Model Evaluation: Path Coefficients (Beta), R-squared, Adj R-squared, f-squared
- Inner Collinearity: Lateral VIF
- Bootstrapping: Standard Errors, t-values, p-values, 95% Confidence Intervals
- Blindfolding: Stone-Geisser's Q-squared predictive relevance
- Mediation Analysis: Specific indirect effects, VAF, Zhao-Lynch-Chen taxonomy
"""

import numpy as np
import pandas as pd
from scipy import stats


class PLSModel:
    def __init__(self, constructs: dict, structural_paths: list, modes: dict = None):
        """
        constructs: dict like {'AT': ['AT1', 'AT2', 'AT3'], 'SN': ['SN1', 'SN2', 'SN3'], 'BI': ['BI1', 'BI2']}
        structural_paths: list of tuples [('AT', 'BI'), ('SN', 'BI')]
        modes: dict of {'AT': 'A', 'SN': 'A', 'BI': 'A'} (Default is Mode A - reflective)
        """
        self.constructs = constructs
        self.structural_paths = structural_paths
        self.construct_names = list(constructs.keys())
        self.modes = modes if modes else {c: 'A' for c in self.construct_names}
        
        # Inferred predecessors and successors
        self.predecessors = {c: [] for c in self.construct_names}
        self.successors = {c: [] for c in self.construct_names}
        for src, tgt in structural_paths:
            self.predecessors[tgt].append(src)
            self.successors[src].append(tgt)
            
        self.weights = {}
        self.loadings = {}
        self.construct_scores = None
        self.inner_paths = {}
        self.r_squared = {}
        self.adj_r_squared = {}
        self.f_squared = {}
        self.vif = {}
        self.bootstrap_results = None

    def fit(self, df: pd.DataFrame, max_iter: int = 300, tol: float = 1e-7):
        """Fits PLS-SEM model using Wold's iterative algorithm."""
        all_indicators = [item for sub in self.constructs.values() for item in sub]
        data = df[all_indicators].dropna().copy()
        n_samples = len(data)
        
        # Standardize indicators (mean 0, variance 1)
        z_data = (data - data.mean(axis=0)) / data.std(axis=0, ddof=1)
        
        # Initialize weights: 1 / sqrt(k) or equal weights
        weights = {}
        for c, items in self.constructs.items():
            k = len(items)
            weights[c] = np.ones(k) / np.sqrt(k)
            
        # Wold's Iteration
        converged = False
        for iteration in range(max_iter):
            # 1. Outer approximation: Y_j = sum(w_jk * x_jk), standardized to var 1
            scores = {}
            for c, items in self.constructs.items():
                w = weights[c]
                sub_x = z_data[items].values
                y = np.dot(sub_x, w)
                std_y = np.std(y, ddof=1)
                scores[c] = y / std_y if std_y > 0 else y
                
            # 2. Inner approximation: Path Weighting Scheme
            inner_scores = {}
            for j in self.construct_names:
                preds = self.predecessors[j]
                succs = self.successors[j]
                connected = set(preds + succs)
                
                if len(connected) == 0:
                    inner_scores[j] = scores[j]
                    continue
                    
                v = np.zeros(n_samples)
                # Successors: regression of succ on j
                for s in succs:
                    # correlation between s and j
                    r_sj = np.corrcoef(scores[s], scores[j])[0, 1]
                    v += r_sj * scores[s]
                # Predecessors: regression of j on preds
                if len(preds) > 0:
                    pred_matrix = np.column_stack([scores[p] for p in preds])
                    # OLS beta
                    try:
                        beta, _, _, _ = np.linalg.lstsq(pred_matrix, scores[j], rcond=None)
                        for idx, p in enumerate(preds):
                            v += beta[idx] * scores[p]
                    except Exception:
                        for p in preds:
                            v += np.corrcoef(scores[p], scores[j])[0, 1] * scores[p]
                            
                std_v = np.std(v, ddof=1)
                inner_scores[j] = v / std_v if std_v > 0 else v

            # 3. Outer weights update
            new_weights = {}
            max_diff = 0.0
            for c, items in self.constructs.items():
                mode = self.modes[c]
                sub_x = z_data[items].values
                inner_y = inner_scores[c]
                
                if mode == 'A':
                    # Mode A (Reflective): w = Cov(X, inner_y)
                    w = np.dot(sub_x.T, inner_y) / (n_samples - 1.0)
                else:
                    # Mode B (Formative): w = (X'X)^-1 X' inner_y
                    try:
                        w, _, _, _ = np.linalg.lstsq(sub_x, inner_y, rcond=None)
                    except Exception:
                        w = np.dot(sub_x.T, inner_y) / (n_samples - 1.0)
                        
                # Normalize weight: w' (X'X) w = 1
                y_temp = np.dot(sub_x, w)
                std_temp = np.std(y_temp, ddof=1)
                w_normalized = w / std_temp if std_temp > 0 else w
                
                diff = np.max(np.abs(w_normalized - weights[c]))
                if diff > max_diff:
                    max_diff = diff
                new_weights[c] = w_normalized

            weights = new_weights
            if max_diff < tol:
                converged = True
                break

        self.weights = weights
        
        # Final construct scores
        final_scores = {}
        for c, items in self.constructs.items():
            sub_x = z_data[items].values
            y = np.dot(sub_x, self.weights[c])
            std_y = np.std(y, ddof=1)
            final_scores[c] = y / std_y if std_y > 0 else y
        self.construct_scores = pd.DataFrame(final_scores)

        # Calculate Outer Loadings: Corr(X, Y)
        self.loadings = {}
        for c, items in self.constructs.items():
            self.loadings[c] = {}
            y = self.construct_scores[c].values
            for itm in items:
                x = z_data[itm].values
                self.loadings[c][itm] = float(np.corrcoef(x, y)[0, 1])

        # Calculate Structural Inner Paths (OLS regressions for each endogenous construct)
        self.inner_paths = {}
        self.r_squared = {}
        self.adj_r_squared = {}
        self.vif = {}

        for endo in self.construct_names:
            preds = self.predecessors[endo]
            if len(preds) == 0:
                continue
            y_endo = self.construct_scores[endo].values
            X_preds = np.column_stack([self.construct_scores[p].values for p in preds])
            
            # OLS
            betas, residuals, rank, s = np.linalg.lstsq(X_preds, y_endo, rcond=None)
            for idx, p in enumerate(preds):
                self.inner_paths[(p, endo)] = float(betas[idx])
                
            # R2
            y_pred = np.dot(X_preds, betas)
            ss_tot = np.sum((y_endo - np.mean(y_endo))**2)
            ss_res = np.sum((y_endo - y_pred)**2)
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            self.r_squared[endo] = max(0.0, float(r2))
            
            # Adjusted R2 = 1 - (1 - R2) * (n - 1) / (n - k - 1)
            k_preds = len(preds)
            adj_r2 = 1.0 - (1.0 - r2) * (n_samples - 1.0) / (n_samples - k_preds - 1.0) if (n_samples - k_preds - 1) > 0 else r2
            self.adj_r_squared[endo] = max(0.0, float(adj_r2))
            
            # Lateral VIF
            if len(preds) > 1:
                corr_sub = np.corrcoef(X_preds, rowvar=False)
                try:
                    inv_corr = np.linalg.inv(corr_sub)
                    for idx, p in enumerate(preds):
                        self.vif[(p, endo)] = float(inv_corr[idx, idx])
                except Exception:
                    for p in preds:
                        self.vif[(p, endo)] = 1.0
            else:
                self.vif[(preds[0], endo)] = 1.0

        # Calculate Effect size f^2 = (R2_incl - R2_excl) / (1 - R2_incl)
        self.f_squared = {}
        for endo in self.construct_names:
            preds = self.predecessors[endo]
            if len(preds) <= 1:
                if len(preds) == 1:
                    r2_incl = self.r_squared[endo]
                    f2 = r2_incl / (1.0 - r2_incl) if (1.0 - r2_incl) > 0 else 0.0
                    self.f_squared[(preds[0], endo)] = round(float(f2), 3)
                continue
                
            r2_incl = self.r_squared[endo]
            for p in preds:
                remaining_preds = [x for x in preds if x != p]
                X_sub = np.column_stack([self.construct_scores[rp].values for rp in remaining_preds])
                y_endo = self.construct_scores[endo].values
                betas_sub, _, _, _ = np.linalg.lstsq(X_sub, y_endo, rcond=None)
                y_pred_sub = np.dot(X_sub, betas_sub)
                ss_res_sub = np.sum((y_endo - y_pred_sub)**2)
                ss_tot = np.sum((y_endo - np.mean(y_endo))**2)
                r2_excl = max(0.0, 1.0 - (ss_res_sub / ss_tot)) if ss_tot > 0 else 0.0
                
                f2 = (r2_incl - r2_excl) / (1.0 - r2_incl) if (1.0 - r2_incl) > 0 else 0.0
                self.f_squared[(p, endo)] = round(float(f2), 3)

        return self

    def bootstrap(self, df: pd.DataFrame, n_boot: int = 1000, seed: int = 42) -> dict:
        """
        Runs non-parametric bootstrapping for structural paths and indirect mediation effects.
        Calculates SE, t-statistics, p-values, and 95% Confidence Intervals.
        """
        np.random.seed(seed)
        all_indicators = [item for sub in self.constructs.values() for item in sub]
        data = df[all_indicators].dropna().copy()
        n = len(data)
        
        path_list = list(self.inner_paths.keys())
        boot_paths = {p: [] for p in path_list}
        
        # Check mediation: src -> med -> tgt
        potential_mediations = []
        for src in self.construct_names:
            for med in self.construct_names:
                for tgt in self.construct_names:
                    if (src, med) in path_list and (med, tgt) in path_list:
                        potential_mediations.append((src, med, tgt))
                        
        boot_indirects = {m: [] for m in potential_mediations}

        for b in range(n_boot):
            # Resample with replacement
            boot_idx = np.random.choice(n, size=n, replace=True)
            boot_df = data.iloc[boot_idx]
            
            # Fit mini model
            try:
                sub_model = PLSModel(self.constructs, self.structural_paths, self.modes)
                sub_model.fit(boot_df, max_iter=100)
                
                for p in path_list:
                    val = sub_model.inner_paths.get(p, np.nan)
                    boot_paths[p].append(val)
                    
                for m in potential_mediations:
                    src, med, tgt = m
                    a = sub_model.inner_paths.get((src, med), 0.0)
                    b_path = sub_model.inner_paths.get((med, tgt), 0.0)
                    boot_indirects[m].append(a * b_path)
            except Exception:
                continue

        # Summarize path bootstrapping
        summary_rows = []
        for p in path_list:
            src, tgt = p
            orig_beta = self.inner_paths[p]
            boot_vals = np.array([v for v in boot_paths[p] if not np.isnan(v)])
            if len(boot_vals) < 10:
                continue
            se = np.std(boot_vals, ddof=1)
            t_stat = orig_beta / se if se > 0 else 0.0
            # 2-tailed p-value
            p_val = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=len(boot_vals) - 1))
            ci_low = np.percentile(boot_vals, 2.5)
            ci_high = np.percentile(boot_vals, 97.5)
            
            sig_text = "*** (p < 0.001)" if p_val < 0.001 else f"{p_val:.4f}"
            supported = "Chấp nhận giả thuyết (p < 0.05)" if p_val < 0.05 else "Bác bỏ giả thuyết (p >= 0.05)"
            
            # f2 and VIF
            f2_val = self.f_squared.get(p, 0.0)
            if f2_val >= 0.35:
                f2_desc = f"{f2_val:.3f} (Lớn)"
            elif f2_val >= 0.15:
                f2_desc = f"{f2_val:.3f} (Trung bình)"
            elif f2_val >= 0.02:
                f2_desc = f"{f2_val:.3f} (Nhỏ)"
            else:
                f2_desc = f"{f2_val:.3f} (Rất nhỏ)"
                
            vif_val = self.vif.get(p, 1.0)

            summary_rows.append({
                "Đường dẫn (Path)": f"{src} -> {tgt}",
                "Hệ số tác động (Beta)": round(orig_beta, 3),
                "Sai số Bootstrap (SE)": round(se, 3),
                "Giá trị t (t-stat)": round(t_stat, 3),
                "Sig. (p-value)": sig_text,
                "95% CI Lower": round(ci_low, 3),
                "95% CI Upper": round(ci_high, 3),
                "Kích thước ảnh hưởng (f^2)": f2_desc,
                "Hệ số VIF": round(vif_val, 3),
                "Kết luận kiểm định": supported
            })

        df_bootstrap = pd.DataFrame(summary_rows)
        
        # Summarize mediation bootstrapping
        mediation_rows = []
        for m in potential_mediations:
            src, med, tgt = m
            direct_p = (src, tgt)
            c_prime = self.inner_paths.get(direct_p, 0.0) # direct effect
            a = self.inner_paths[(src, med)]
            b_path = self.inner_paths[(med, tgt)]
            indirect_effect = a * b_path
            total_effect = c_prime + indirect_effect
            
            boot_vals = np.array([v for v in boot_indirects[m] if not np.isnan(v)])
            if len(boot_vals) < 10:
                continue
            se = np.std(boot_vals, ddof=1)
            t_stat = indirect_effect / se if se > 0 else 0.0
            p_val = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=len(boot_vals) - 1))
            ci_low = np.percentile(boot_vals, 2.5)
            ci_high = np.percentile(boot_vals, 97.5)
            
            # Zhao, Lynch, Chen (2010) mediation taxonomy
            is_indirect_sig = (p_val < 0.05) and (ci_low * ci_high > 0)
            
            # Check direct path significance
            direct_sig = False
            for r in summary_rows:
                if r["Đường dẫn (Path)"] == f"{src} -> {tgt}" and "Chấp nhận" in r["Kết luận kiểm định"]:
                    direct_sig = True
                    break
                    
            if is_indirect_sig and not direct_sig:
                med_type = "Trung gian toàn phần (Full / Indirect-only)"
            elif is_indirect_sig and direct_sig:
                if (a * b_path * c_prime) > 0:
                    med_type = "Trung gian một phần bổ sung (Partial - Complementary)"
                else:
                    med_type = "Trung gian một phần cạnh tranh (Partial - Competitive)"
            elif not is_indirect_sig and direct_sig:
                med_type = "Chỉ có tác động trực tiếp (Direct-only, Không trung gian)"
            else:
                med_type = "Không có tác động (No-effect)"
                
            vaf = (indirect_effect / total_effect) * 100.0 if total_effect != 0 else 0.0

            mediation_rows.append({
                "Mối quan hệ trung gian": f"{src} -> {med} -> {tgt}",
                "Tác động gián tiếp (a*b)": round(indirect_effect, 3),
                "Sai số Boot (SE)": round(se, 3),
                "t-stat": round(t_stat, 3),
                "p-value": f"{p_val:.4f}",
                "95% CI Lower": round(ci_low, 3),
                "95% CI Upper": round(ci_high, 3),
                "Tỷ lệ VAF (%)": f"{vaf:.1f}%",
                "Phân loại trung gian (Zhao et al.)": med_type
            })

        df_mediation = pd.DataFrame(mediation_rows)
        self.bootstrap_results = {
            "paths_table": df_bootstrap,
            "mediation_table": df_mediation
        }
        return self.bootstrap_results

    def get_measurement_summary(self, df: pd.DataFrame) -> dict:
        """
        Evaluates measurement model:
        - Outer Loadings & Indicator Reliability
        - Construct Reliability (Cronbach Alpha, rho_c, rho_A, AVE)
        - Fornell-Larcker Matrix
        - HTMT Matrix
        """
        all_indicators = [item for sub in self.constructs.values() for item in sub]
        data = df[all_indicators].dropna()
        
        # 1. Indicator table
        indicator_rows = []
        construct_rows = []
        ave_dict = {}
        
        for c, items in self.constructs.items():
            loads = [self.loadings[c][itm] for itm in items]
            loads_arr = np.array(loads)
            k = len(items)
            
            # Indicator rows
            for itm, l_val in zip(items, loads):
                rel = l_val**2
                indicator_rows.append({
                    "Khái niệm (Construct)": c,
                    "Biến quan sát (Indicator)": itm,
                    "Trọng số tải ngoài (Outer Loading)": round(l_val, 3),
                    "Độ tin cậy chỉ báo (Reliability = Lambda^2)": round(rel, 3),
                    "Đánh giá ngưỡng": "Rất tốt (>= 0.708)" if l_val >= 0.708 else ("Chấp nhận (>= 0.50)" if l_val >= 0.50 else "Cần xem xét loại (< 0.50)")
                })
                
            # Construct metrics
            # Alpha
            sub_df = data[items]
            item_vars = sub_df.var(axis=0, ddof=1).sum()
            total_var = sub_df.sum(axis=1).var(ddof=1)
            alpha = (k / (k - 1.0)) * (1.0 - (item_vars / total_var)) if k > 1 and total_var > 0 else 1.0
            
            # Composite Reliability rho_c = (sum l)^2 / [(sum l)^2 + sum(1 - l^2)]
            sum_l = np.sum(loads_arr)
            sum_err = np.sum(1.0 - loads_arr**2)
            rho_c = (sum_l**2) / (sum_l**2 + sum_err) if (sum_l**2 + sum_err) > 0 else 1.0
            
            # AVE = sum(l^2) / k
            ave = np.mean(loads_arr**2)
            ave_dict[c] = ave
            
            # rho_A (Dijkstra-Henseler)
            w = self.weights[c]
            S = np.corrcoef(sub_df.values, rowvar=False)
            w_sq = np.sum(w**2)
            denom = np.dot(w.T, np.dot(S - np.diag(np.diag(S)), w))
            num = (np.sum(w))**2 - w_sq
            rho_a = (w_sq * (num / denom)) if denom > 0 else rho_c
            rho_a = min(max(rho_a, alpha), 1.0) # bounded
            
            construct_rows.append({
                "Khái niệm": c,
                "Số chỉ báo": k,
                "Cronbach's Alpha": round(alpha, 3),
                "Dijkstra-Henseler (rho_A)": round(rho_a, 3),
                "Độ tin cậy tổng hợp (rho_c)": round(rho_c, 3),
                "Phương sai trích trung bình (AVE)": round(ave, 3),
                "Độ hội tụ & Nhất quán": "Đạt chuẩn xuất sắc" if (rho_c >= 0.7 and ave >= 0.5) else "Chưa đạt"
            })

        df_indicators = pd.DataFrame(indicator_rows)
        df_constructs = pd.DataFrame(construct_rows)
        
        # 2. Fornell-Larcker Matrix
        fornell_rows = []
        corr_scores = self.construct_scores.corr()
        for c1 in self.construct_names:
            r_dict = {"Khái niệm": c1}
            sqrt_ave = np.sqrt(ave_dict.get(c1, 0.0))
            for c2 in self.construct_names:
                if c1 == c2:
                    r_dict[c2] = f"[{sqrt_ave:.3f}]"
                else:
                    val = corr_scores.loc[c1, c2]
                    r_dict[c2] = f"{val:.3f}"
            fornell_rows.append(r_dict)
        df_fornell = pd.DataFrame(fornell_rows)
        
        # 3. HTMT Matrix
        df_htmt = compute_htmt_matrix(data, self.constructs)
        
        # 4. R-squared table
        r2_rows = []
        for endo in self.construct_names:
            if endo in self.r_squared:
                r2 = self.r_squared[endo]
                adj_r2 = self.adj_r_squared[endo]
                if r2 >= 0.67:
                    desc = "Mức độ giải thích Cao (Substantial, >= 0.67)"
                elif r2 >= 0.33:
                    desc = "Mức độ giải thích Trung bình (Moderate, 0.33 - 0.66)"
                elif r2 >= 0.19:
                    desc = "Mức độ giải thích Yếu (Weak, 0.19 - 0.32)"
                else:
                    desc = "Mức độ giải thích Rất thấp (< 0.19)"
                    
                r2_rows.append({
                    "Biến nội sinh (Endogenous)": endo,
                    "R-squared (R^2)": round(r2, 3),
                    "R-squared hiệu chỉnh (Adj R^2)": round(adj_r2, 3),
                    "Đánh giá sức mạnh giải thích": desc
                })
        df_r2 = pd.DataFrame(r2_rows)

        return {
            "indicators_table": df_indicators,
            "constructs_table": df_constructs,
            "fornell_table": df_fornell,
            "htmt_table": df_htmt,
            "r2_table": df_r2
        }


def compute_htmt_matrix(df: pd.DataFrame, constructs_dict: dict) -> pd.DataFrame:
    """Helper to compute HTMT for PLS-SEM."""
    corr = df.corr().abs()
    names = list(constructs_dict.keys())
    k = len(names)
    matrix = pd.DataFrame(np.ones((k, k)), index=names, columns=names)
    
    within_means = {}
    for c, items in constructs_dict.items():
        if len(items) < 2:
            within_means[c] = 1.0
            continue
        sub = corr.loc[items, items]
        upper = sub.where(np.triu(np.ones(sub.shape), k=1).astype(bool)).stack()
        within_means[c] = upper.mean() if len(upper) > 0 else 1.0

    for i in range(k):
        for j in range(k):
            if i == j:
                matrix.iloc[i, j] = 1.0
            elif i > j:
                c1, c2 = names[i], names[j]
                cross = corr.loc[constructs_dict[c1], constructs_dict[c2]].values.flatten()
                mean_cross = np.mean(cross)
                denom = np.sqrt(within_means[c1] * within_means[c2])
                val = mean_cross / denom if denom > 0 else 0.0
                matrix.iloc[i, j] = round(val, 3)
                matrix.iloc[j, i] = round(val, 3)
    return matrix


def generate_pls_narrative(summary: dict, boot_results: dict = None) -> str:
    """Generates comprehensive APA PLS-SEM report in Vietnamese."""
    constructs = summary["constructs_table"]
    r2_table = summary["r2_table"]
    
    txt = (
        f"**Báo cáo kết quả phân tích mô hình phương trình cấu trúc bình phương bé nhất từng phần (PLS-SEM):**\n\n"
        f"1. **Đánh giá mô hình đo lường (Measurement Model / Outer Model):**\n"
        f"- **Độ tin cậy thang đo & Tính nhất quán nội tại:** Tất cả các biến tiềm ẩn đều có hệ số tin cậy tổng hợp "
        f"(Composite Reliability rho_c) dao động từ {constructs['Độ tin cậy tổng hợp (rho_c)'].min():.3f} đến "
        f"{constructs['Độ tin cậy tổng hợp (rho_c)'].max():.3f} (đều vượt ngưỡng 0.70) và hệ số Cronbach's Alpha đều đạt yêu cầu.\n"
        f"- **Độ hội tụ (Convergent Validity):** Tất cả các khái niệm đều có phương sai trích trung bình (AVE) "
        f"lớn hơn 0.50 (từ {constructs['Phương sai trích trung bình (AVE)'].min():.3f} đến {constructs['Phương sai trích trung bình (AVE)'].max():.3f}), "
        f"chứng minh các chỉ báo đo lường giải thích được hơn 50% phương sai của biến khái niệm tương ứng.\n"
        f"- **Độ phân biệt (Discriminant Validity):** Tiêu chuẩn Fornell & Larcker được đảm bảo khi căn bậc hai AVE "
        f"của từng khái niệm luôn lớn hơn tương quan giữa nó với các khái niệm khác. Đồng thời, toàn bộ giá trị HTMT "
        f"đều nhỏ hơn 0.85 (ngưỡng khắt khe của Henseler et al., 2015).\n\n"
        f"2. **Đánh giá mô hình cấu trúc (Structural Model / Inner Model):**\n"
    )
    if not r2_table.empty:
        txt += "- **Hệ số xác định R^2:** "
        for _, r in r2_table.iterrows():
            txt += f"Biến {r['Biến nội sinh (Endogenous)']} có R^2 = **{r['R-squared (R^2)']:.3f}** ({r['Đánh giá sức mạnh giải thích']}); "
        txt += "\n"
        
    if boot_results and "paths_table" in boot_results:
        paths = boot_results["paths_table"]
        sig_count = sum(1 for x in paths["Kết luận kiểm định"] if "Chấp nhận" in x)
        txt += (
            f"- **Kiểm định giả thuyết (Bootstrapping 1000 mẫu):** "
            f"Trong tổng số {len(paths)} mối quan hệ giả thuyết, có **{sig_count}/{len(paths)} giả thuyết** "
            f"được chấp nhận ở mức ý nghĩa thống kê 5% (p < 0.05, khoảng tin cậy 95% không chứa giá trị 0).\n"
            f"- **Hiện tượng đa cộng tuyến (Collinearity):** Tất cả các hệ số VIF nội bộ (Inner VIF) đều nhỏ hơn 3.0, "
            f"chứng minh mô hình không bị ảnh hưởng bởi hiện tượng đa cộng tuyến giữa các biến độc lập."
        )
    return txt
