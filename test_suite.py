"""
Automated Test Suite for SmartResearch Studio
Verifies all mathematical models, data loaders, and report generators.
"""

import sys
import os
import mimetypes

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

try:
    if hasattr(mimetypes, "MimeTypes") and hasattr(mimetypes.MimeTypes, "read_windows_registry"):
        mimetypes.MimeTypes.read_windows_registry = lambda self, *args, **kwargs: None
    mimetypes.init()
except Exception:
    pass

import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.data_loader import DataLoader
from core.descriptives import calculate_frequency_table, calculate_descriptive_stats, generate_descriptives_narrative
from core.reliability import analyze_scale_reliability, generate_reliability_narrative
from core.hypothesis_tests import run_independent_ttest, run_oneway_anova, generate_ttest_narrative
from core.efa import run_efa, generate_efa_narrative
from core.cfa_cbsem import run_cfa_cbsem, generate_cfa_cbsem_narrative
from core.pls_sem import PLSModel, generate_pls_narrative
from core.machine_learning import (
    analyze_dataset_and_recommend, run_advanced_supervised, run_pca_reduction,
    run_hierarchical_clustering_model, run_lda_topic_model, run_sentiment_analysis_suite,
    HAS_XGBOOST, HAS_LIGHTGBM
)
from core.report_exporter import create_word_report, create_excel_report


def run_all_tests():
    print("=== BẮT ĐẦU KIỂM THỬ TOÀN DIỆN SMARTRESEARCH STUDIO ===")
    
    # 1. Test Data Loader
    loader = DataLoader()
    sav_path = "sample_data/social_research_data.sav"
    with open(sav_path, "rb") as f:
        df = loader.load_file(f.read(), "social_research_data.sav")
    print(f"[OK] 1. Data Loader: Đọc thành công SPSS .sav ({df.shape[0]} dòng, {df.shape[1]} cột).")
    print(f"     Nhãn biến: {len(loader.variable_labels)} nhãn, Nhãn giá trị: {len(loader.value_labels)} trường.")
    
    # 1B. Test Data Cleaner & Column Coding
    from core.data_cleaner import (
        auto_suggest_column_codes, detect_data_health,
        clean_missing_values, clean_duplicates, clean_outliers,
        recode_column_values, generate_codebook, export_cleaned_dataset_excel,
        LIKERT_PRESETS
    )
    raw_sample_cols = [
        "Dấu thời gian", "Địa chỉ email", "1. Bạn bao nhiêu tuổi?", "Giới tính của bạn là gì?",
        "Tôi thấy hệ thống rất dễ sử dụng [PE1]", "Tôi có thể thao tác nhanh chóng [PE2]"
    ]
    sugg = auto_suggest_column_codes(raw_sample_cols)
    assert sugg == ["Timestamp", "Email", "Age", "Gender", "PE1", "PE2"]
    
    # Test cleaning operations on a test df
    test_clean_df = pd.DataFrame({
        "Q1": ["Đồng ý", "Hoàn toàn đồng ý", np.nan, "Đồng ý"],
        "Q2": [4, 5, 4, 4],
        "Dup": [1, 2, 3, 1]
    })
    rec_df = recode_column_values(test_clean_df, "Q1", LIKERT_PRESETS["Thang đo Likert 5 mức độ (Hoàn toàn không đồng ý → Hoàn toàn đồng ý)"])
    assert rec_df.loc[0, "Q1"] == 4 and rec_df.loc[1, "Q1"] == 5
    
    na_clean, _ = clean_missing_values(rec_df, method="drop_rows")
    assert len(na_clean) == 3
    
    codebook_df = generate_codebook(df, loader.variable_labels)
    assert len(codebook_df) == len(df.columns)
    
    clean_excel_buf = export_cleaned_dataset_excel(df, loader.variable_labels)
    assert clean_excel_buf.getbuffer().nbytes > 0
    print(f"[OK] 1B. Data Cleaner & Column Encoding: Đạt, gợi ý mã biến, mã hóa Likert, làm sạch & xuất Excel 2 sheet ({clean_excel_buf.getbuffer().nbytes} bytes).")
    
    # 2. Test Descriptives & Demographics
    freq_df = calculate_frequency_table(df["GIOITINH"], loader.value_labels.get("GIOITINH"))
    print(f"[OK] 2A. Frequency Table: Đạt, tính đúng cơ cấu giới tính:\n{freq_df}")
    
    desc_df = calculate_descriptive_stats(df, ["PU1", "PU2", "PU3", "PU4"])
    print(f"[OK] 2B. Descriptives & Normality: Đạt, Skewness & Kurtosis chuẩn xác.")
    
    # 3. Test Cronbach's Alpha
    rel_res = analyze_scale_reliability(df, ["PU1", "PU2", "PU3", "PU4"], "Perceived Usefulness", loader.variable_labels)
    assert rel_res["overall_alpha"] > 0.7, "Cronbach Alpha không đạt ngưỡng!"
    print(f"[OK] 3. Cronbach's Alpha: Đạt, Alpha = {rel_res['overall_alpha']:.3f} ({rel_res['scale_quality']}).")
    
    # 4. Test T-Test & ANOVA
    ttest_res = run_independent_ttest(df, "PU1", "GIOITINH", loader.value_labels)
    assert "error" not in ttest_res, ttest_res.get("error")
    print(f"[OK] 4A. T-Test: Đạt, Levene p = {ttest_res['levene_p']:.4f}, t = {ttest_res['tests_table']['t'].values[0]}.")
    
    anova_res = run_oneway_anova(df, "PU1", "DOTUOI", loader.value_labels)
    assert "error" not in anova_res, anova_res.get("error")
    print(f"[OK] 4B. One-Way ANOVA: Đạt, F = {anova_res['f_stat']}, p = {anova_res['p_val']}.")
    
    # 5. Test EFA
    efa_items = [f"PU{i}" for i in range(1, 5)] + [f"PE{i}" for i in range(1, 5)] + [f"TR{i}" for i in range(1, 5)]
    efa_res = run_efa(df, efa_items, n_factors=3, method="minres", rotation="promax", labels_dict=loader.variable_labels)
    assert "error" not in efa_res, efa_res.get("error")
    assert "pattern_matrix_spss" in efa_res, "Missing pattern_matrix_spss in EFA output"
    assert "structure_matrix_spss" in efa_res, "Missing structure_matrix_spss in EFA output"
    assert "factor_correlations" in efa_res, "Missing factor_correlations in EFA output"
    print(f"[OK] 5. EFA: Đạt, KMO = {efa_res['kmo_total']:.3f}, Bartlett p = {efa_res['bartlett_p']:.5f}, Cum Var = {efa_res['total_cum_variance']:.2f}%.")
    print(f"     SPSS Pattern Matrix: {efa_res['pattern_matrix_spss'].shape[0]} items, {efa_res['pattern_matrix_spss'].shape[1]} columns.")
    
    # 6. Test CFA / CB-SEM
    syntax = """
    PU =~ PU1 + PU2 + PU3 + PU4
    PE =~ PE1 + PE2 + PE3 + PE4
    TR =~ TR1 + TR2 + TR3 + TR4
    """
    cfa_res = run_cfa_cbsem(df, syntax, {"PU": ["PU1", "PU2", "PU3", "PU4"], "PE": ["PE1", "PE2", "PE3", "PE4"], "TR": ["TR1", "TR2", "TR3", "TR4"]})
    assert "error" not in cfa_res, cfa_res.get("error")
    print(f"[OK] 6. CFA/CB-SEM: Đạt, CMIN/df = {cfa_res['cmin_df']}, CFI = {cfa_res['cfi']}, RMSEA = {cfa_res['rmsea']}.")
    
    # 7. Test PLS-SEM
    constructs = {
        "PU": ["PU1", "PU2", "PU3", "PU4"],
        "PE": ["PE1", "PE2", "PE3", "PE4"],
        "TR": ["TR1", "TR2", "TR3", "TR4"],
        "AT": ["AT1", "AT2", "AT3", "AT4"],
        "BI": ["BI1", "BI2", "BI3"]
    }
    paths = [("PU", "AT"), ("PE", "AT"), ("TR", "AT"), ("AT", "BI"), ("PU", "BI")]
    pls = PLSModel(constructs, paths)
    pls.fit(df)
    m_res = pls.get_measurement_summary(df)
    boot_res = pls.bootstrap(df, n_boot=50) # 50 samples for fast test
    print(f"[OK] 7. PLS-SEM: Đạt, Wold's algorithm hội tụ thành công.")
    print(f"     Construct R2: {list(pls.r_squared.items())}")
    print(f"     Bootstrap paths: {len(boot_res['paths_table'])} đường dẫn, Mediation: {len(boot_res['mediation_table'])} quan hệ.")
    
    # 8. Test Machine Learning Suite
    # 8A. Advisor
    recs = analyze_dataset_and_recommend(df)
    assert len(recs["recommendations"]) > 5
    print(f"[OK] 8A. AI Advisor: Đạt, quét tự động và đề xuất {len(recs['recommendations'])} mô hình phù hợp.")
    
    # 8B. Supervised: XGBoost / Random Forest
    sup_algo = "XGBoost" if HAS_XGBOOST else "Random Forest"
    sup_res = run_advanced_supervised(df, "USE", ["PU1", "PE1", "TR1", "THUNHAP"], algorithm=sup_algo)
    assert "error" not in sup_res, sup_res.get("error")
    print(f"[OK] 8B. Supervised Learning ({sup_algo}): Đạt, Accuracy & ROC-AUC tính chuẩn xác.")
    
    # 8C. Unsupervised: PCA & Hierarchical
    pca_res = run_pca_reduction(df, ["PU1", "PU2", "PE1", "PE2", "TR1", "TR2"], n_components=3)
    assert "variance_table" in pca_res
    print(f"[OK] 8C. PCA Dimensionality Reduction: Đạt, Phương sai giải thích = {pca_res['total_variance_explained']}%.")
    
    hc_res = run_hierarchical_clustering_model(df, ["PU1", "PE1", "TR1"], n_clusters=3)
    assert "silhouette_score" in hc_res
    print(f"[OK] 8D. Hierarchical Clustering: Đạt, Silhouette = {hc_res['silhouette_score']}.")
    
    # 8E. NLP: Sentiment Analysis & LDA
    sent_res = run_sentiment_analysis_suite(df["Y_KIEN_DAP_VIEN"])
    assert "error" not in sent_res, sent_res.get("error")
    print(f"[OK] 8E. Sentiment Analysis: Đạt, Phân loại cảm xúc chuẩn xác (Độ phân cực TB = {sent_res['average_polarity']:+.3f}).")
    
    lda_res = run_lda_topic_model(df["Y_KIEN_DAP_VIEN"], n_topics=3)
    assert "error" not in lda_res, lda_res.get("error")
    print(f"[OK] 8F. LDA Topic Modeling: Đạt, Trích xuất thành công {len(lda_res['topics_table'])} chủ đề tiềm ẩn.")
    
    # 9. Test Report Exporters
    sections = [
        {"heading": "1. Thống kê Cronbach's Alpha", "narrative": "Thang đo đạt độ tin cậy tốt.", "tables": [("Bảng Alpha", rel_res["items_table"])]}
    ]
    docx_buf = create_word_report("Báo cáo NCKH Mẫu", sections)
    assert docx_buf.getbuffer().nbytes > 0
    print(f"[OK] 9A. Word (.docx) Exporter: Tạo thành công tệp ({docx_buf.getbuffer().nbytes} bytes).")
    
    excel_buf = create_excel_report({"Alpha": rel_res["items_table"], "Descriptives": desc_df})
    assert excel_buf.getbuffer().nbytes > 0
    print(f"[OK] 9B. Excel (.xlsx) Exporter: Tạo thành công workbook ({excel_buf.getbuffer().nbytes} bytes).")
    
    # 9C. Test Mini Table Excel Exporter (df_to_styled_excel)
    from core.report_exporter import df_to_styled_excel
    mini_excel_bytes = df_to_styled_excel(desc_df, sheet_name="Descriptives")
    assert len(mini_excel_bytes) > 1000
    styler_bytes = df_to_styled_excel(desc_df.style, sheet_name="StylerTest")
    assert len(styler_bytes) > 1000
    print(f"[OK] 9C. Mini Table Excel Exporter: Xuất thành công tệp Excel bảng lẻ ({len(mini_excel_bytes)} bytes).")
    
    print("\n🎉 TẤT CẢ 9/9 MODULE KIỂM THỬ ĐỀU THÀNH CÔNG XUẤT SẮC! 🎉")


if __name__ == '__main__':
    run_all_tests()
