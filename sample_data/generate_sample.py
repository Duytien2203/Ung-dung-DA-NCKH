"""
Generates a realistic social science research dataset
with 350 respondents, 5-point Likert scales, demographics,
and latent constructs (PU, PE, TR, AT, BI, USE).
Exports both Excel (.xlsx) and SPSS (.sav) formats.
"""

import mimetypes
mimetypes.init = lambda *args, **kwargs: None

import numpy as np
import pandas as pd
import pyreadstat


def generate_research_dataset(n_samples: int = 350, seed: int = 42) -> tuple:
    np.random.seed(seed)
    
    # 1. Demographics
    gioitinh = np.random.choice([1, 2], size=n_samples, p=[0.48, 0.52])
    dotuoi = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.30, 0.40, 0.20, 0.10])
    thunhap = np.random.choice([1, 2, 3], size=n_samples, p=[0.35, 0.45, 0.20])
    hocvan = np.random.choice([1, 2, 3], size=n_samples, p=[0.15, 0.65, 0.20])
    
    # 2. Latent variable true scores (multivariate normal)
    # PU, PE, TR -> AT -> BI
    # Covariance structure
    mean_latents = [0, 0, 0]
    cov_latents = [
        [1.0, 0.45, 0.40],
        [0.45, 1.0, 0.35],
        [0.40, 0.35, 1.0]
    ]
    exogenous = np.random.multivariate_normal(mean_latents, cov_latents, size=n_samples)
    pu_true = exogenous[:, 0]
    pe_true = exogenous[:, 1]
    tr_true = exogenous[:, 2]
    
    # Structural equations:
    # AT = 0.35 * PU + 0.30 * PE + 0.25 * TR + error
    at_true = 0.35 * pu_true + 0.30 * pe_true + 0.25 * tr_true + np.random.normal(0, 0.4, size=n_samples)
    
    # BI = 0.55 * AT + 0.25 * PU + error
    bi_true = 0.55 * at_true + 0.25 * pu_true + np.random.normal(0, 0.35, size=n_samples)
    
    # 3. Generate Likert items from latents (5-point scale: 1 to 5)
    def make_likert_items(true_score, loadings, base_mean=3.8):
        items = {}
        for idx, (name, lambda_val) in enumerate(loadings):
            raw = base_mean + lambda_val * true_score + np.random.normal(0, np.sqrt(1 - lambda_val**2) * 0.7, size=n_samples)
            # Clip to 1..5 and round to integer
            items[name] = np.clip(np.round(raw), 1, 5).astype(int)
        return items
        
    pu_items = make_likert_items(pu_true, [('PU1', 0.82), ('PU2', 0.85), ('PU3', 0.78), ('PU4', 0.75)], base_mean=3.9)
    pe_items = make_likert_items(pe_true, [('PE1', 0.80), ('PE2', 0.84), ('PE3', 0.79), ('PE4', 0.76)], base_mean=3.7)
    tr_items = make_likert_items(tr_true, [('TR1', 0.78), ('TR2', 0.82), ('TR3', 0.86), ('TR4', 0.74)], base_mean=3.6)
    at_items = make_likert_items(at_true, [('AT1', 0.83), ('AT2', 0.87), ('AT3', 0.81), ('AT4', 0.79)], base_mean=3.85)
    bi_items = make_likert_items(bi_true, [('BI1', 0.85), ('BI2', 0.88), ('BI3', 0.82)], base_mean=3.95)
    
    # 4. Target variable for ML (Actual Adoption: 0 = Chua dung, 1 = Da dung)
    prob_use = 1.0 / (1.0 + np.exp(-(0.8 * bi_true + 0.3 * (thunhap - 2))))
    use_target = (np.random.rand(n_samples) < prob_use).astype(int)
    
    # 5. Realistic Open-ended Survey Feedback for NLP & Sentiment Analysis & LDA
    positive_comments = [
        "Giao diện trực quan, thao tác rất đơn giản và mượt mà, tôi rất hài lòng.",
        "Dịch vụ hỗ trợ khách hàng chu đáo, giải quyết vấn đề nhanh chóng, tính bảo mật cao.",
        "Tiết kiệm rất nhiều thời gian trong công việc hàng ngày, nền tảng rất thông minh.",
        "Hệ thống an toàn, bảo mật dữ liệu tốt, tôi hoàn toàn an tâm khi sử dụng.",
        "Ứng dụng hiện đại, trải nghiệm thanh toán thuận tiện, rất đáng tin cậy."
    ]
    neutral_comments = [
        "Tính năng ở mức cơ bản, đáp ứng vừa đủ nhu cầu hiện tại.",
        "Giao diện tạm ổn, tốc độ xử lý ở mức trung bình, có thể cải thiện thêm.",
        "Chất lượng chấp nhận được so với mức phí dịch vụ, không có gì quá nổi bật.",
        "Hệ thống hoạt động bình thường, đôi khi cần hướng dẫn chi tiết hơn."
    ]
    negative_comments = [
        "Hệ thống thỉnh thoảng bị lỗi kết nối và phản hồi chậm, gây gián đoạn công việc.",
        "Giao diện còn hơi phức tạp với người mới bắt đầu, thao tác rườm rà.",
        "Chính sách bảo mật cần minh bạch hơn, tôi vẫn còn lo ngại về an toàn thông tin.",
        "Tốc độ tải dữ liệu chậm vào giờ cao điểm, rất bất tiện khi cần gấp.",
        "Dịch vụ chăm sóc khách hàng phản hồi chưa kịp thời, cần cải thiện tốc độ hỗ trợ."
    ]
    
    comments = []
    for i in range(n_samples):
        # Base on satisfaction score AT
        at_score = np.mean([at_items['AT1'][i], at_items['AT2'][i], at_items['AT3'][i], at_items['AT4'][i]])
        if at_score >= 3.8:
            comments.append(np.random.choice(positive_comments))
        elif at_score <= 2.8:
            comments.append(np.random.choice(negative_comments))
        else:
            comments.append(np.random.choice(neutral_comments))

    # Assemble DataFrame
    data_dict = {
        'ID': np.arange(1, n_samples + 1),
        'GIOITINH': gioitinh,
        'DOTUOI': dotuoi,
        'THUNHAP': thunhap,
        'HOCVAN': hocvan,
        **pu_items,
        **pe_items,
        **tr_items,
        **at_items,
        **bi_items,
        'USE': use_target,
        'Y_KIEN_DAP_VIEN': comments
    }
    df = pd.DataFrame(data_dict)
    
    # Metadata for SPSS
    column_labels = [
        "Mã định danh đối tượng",
        "Giới tính của đáp viên",
        "Độ tuổi của đáp viên",
        "Mức thu nhập hàng tháng",
        "Trình độ học vấn cao nhất",
        "PU1: Hệ thống giúp nâng cao hiệu quả công việc",
        "PU2: Hệ thống tiết kiệm thời gian giao dịch",
        "PU3: Dịch vụ đáp ứng đúng nhu cầu của tôi",
        "PU4: Nhìn chung dịch vụ này rất hữu ích",
        "PE1: Thao tác sử dụng hệ thống rất đơn giản",
        "PE2: Giao diện trực quan, dễ làm quen",
        "PE3: Dễ dàng tìm kiếm thông tin cần thiết",
        "PE4: Các bước thực hiện giao dịch rõ ràng",
        "TR1: Tôi tin tưởng vào tính bảo mật của hệ thống",
        "TR2: Đơn vị cung cấp bảo vệ tốt dữ liệu cá nhân",
        "TR3: Các cam kết dịch vụ luôn được đảm bảo",
        "TR4: Tôi cảm thấy an tâm khi thực hiện giao dịch",
        "AT1: Tôi thích sử dụng các dịch vụ số này",
        "AT2: Sử dụng hệ thống là một quyết định thông minh",
        "AT3: Tôi có thái độ tích cực đối với nền tảng",
        "AT4: Trải nghiệm sử dụng mang lại sự hài lòng",
        "BI1: Tôi dự định tiếp tục sử dụng trong tương lai",
        "BI2: Tôi sẽ giới thiệu hệ thống cho người khác",
        "BI3: Tôi ưu tiên sử dụng dịch vụ này khi có nhu cầu",
        "Thực tế sử dụng dịch vụ (0: Chưa, 1: Đã dùng)",
        "Ý kiến đóng góp và cảm nhận của đáp viên về dịch vụ"
    ]
    
    variable_value_labels = {
        'GIOITINH': {1: 'Nam', 2: 'Nữ'},
        'DOTUOI': {1: 'Dưới 25 tuổi', 2: 'Từ 25 - 35 tuổi', 3: 'Từ 36 - 45 tuổi', 4: 'Trên 45 tuổi'},
        'THUNHAP': {1: 'Dưới 10 triệu/tháng', 2: 'Từ 10 - 20 triệu/tháng', 3: 'Trên 20 triệu/tháng'},
        'HOCVAN': {1: 'Trung cấp / Cao đẳng', 2: 'Đại học', 3: 'Sau đại học'},
        'USE': {0: 'Chưa sử dụng thường xuyên', 1: 'Đã sử dụng thường xuyên'}
    }
    
    # Add Likert labels
    likert_labels = {1: 'Rất không đồng ý', 2: 'Không đồng ý', 3: 'Trung lập', 4: 'Đồng ý', 5: 'Rất đồng ý'}
    for col in list(pu_items.keys()) + list(pe_items.keys()) + list(tr_items.keys()) + list(at_items.keys()) + list(bi_items.keys()):
        variable_value_labels[col] = likert_labels

    variable_measure = {
        'ID': 'nominal',
        'GIOITINH': 'nominal',
        'DOTUOI': 'ordinal',
        'THUNHAP': 'ordinal',
        'HOCVAN': 'ordinal',
        'USE': 'nominal',
        'Y_KIEN_DAP_VIEN': 'nominal'
    }
    for col in list(pu_items.keys()) + list(pe_items.keys()) + list(tr_items.keys()) + list(at_items.keys()) + list(bi_items.keys()):
        variable_measure[col] = 'scale'
        
    return df, column_labels, variable_value_labels, variable_measure


if __name__ == '__main__':
    df, col_labels, val_labels, measures = generate_research_dataset(n_samples=350)
    
    # Save Excel
    df.to_excel('sample_data/social_research_data.xlsx', index=False)
    print("Created sample_data/social_research_data.xlsx")
    
    # Save SPSS
    pyreadstat.write_sav(
        df, 
        'sample_data/social_research_data.sav',
        column_labels=col_labels,
        variable_value_labels=val_labels,
        variable_measure=measures
    )
    print("Created sample_data/social_research_data.sav")
