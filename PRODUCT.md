# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

- **Người dùng chính**: Giảng viên, Nghiên cứu sinh (PhD), Học viên Cao học (Thạc sĩ), và Sinh viên năm cuối thực hiện Khóa luận tốt nghiệp hoặc Đề tài Nghiên cứu Khoa học (NCKH) thuộc các khối ngành Khoa học Xã hội, Kinh tế, Quản trị Kinh doanh, Marketing, Tài chính, Giáo dục và Xã hội học.
- **Tình huống & Mục tiêu công việc**: Người dùng sở hữu bộ dữ liệu khảo sát định lượng (từ Google Forms, khảo sát trực tiếp...) cần hoàn thành quy trình xử lý, kiểm định mô hình thống kê phức tạp và xuất kết quả chuẩn quy cách học thuật quốc tế (APA 7th) mà không cần bản quyền đắt tiền hay kỹ năng lập trình chuyên sâu.

## Product Purpose

- Cung cấp một nền tảng phân tích thống kê toàn diện, chuẩn xác và tức thì trên trình duyệt web, đóng vai trò là giải pháp thay thế hoàn hảo và tiện lợi cho bộ ba phần mềm truyền thống: **SPSS, AMOS và SmartPLS**.
- Giúp người làm nghiên cứu rút ngắn thời gian xử lý dữ liệu từ nhiều ngày xuống còn vài phút, loại bỏ các lỗi thao tác thủ công, tự động hóa diễn giải học thuật và xuất báo cáo Word/Excel sẵn sàng nộp hội đồng hoặc xuất bản bài báo khoa học.

## Positioning

- **Nền tảng All-in-one Khép kín 10 Bước**: Duy nhất tích hợp liền mạch từ khâu nạp tệp gốc (.sav, .xlsx), làm sạch mã hóa tự động, kiểm định độ tin cậy (Cronbach's Alpha), EFA, mô hình hiệp phương sai (CFA / CB-SEM) lẫn mô hình bình phương bé nhất từng phần (PLS-SEM), tích hợp Machine Learning và tự động kết xuất báo cáo Word định dạng chuẩn APA 7th.
- **Không yêu cầu cài đặt phần mềm & Không rào cản chi phí**: Hoạt động 100% trên nền tảng web, giao diện tiếng Việt học thuật chuẩn mực, trực quan, hỗ trợ tính toán giải thuật hội tụ chính xác tuyệt đối (Wold's Algorithm 2022, Maximum Likelihood, Bootstrapping).

## Operating Context

- **Quy trình nghiên cứu chuẩn mực**: Tuân thủ quy trình 10 bước nghiêm ngặt trong phân tích định lượng học thuật:
  1. Quản lý & Khảo sát Dữ liệu (SPSS Metadata & Data Dictionary)
  2. Tiền xử lý: Tự động mã hóa thang đo Likert, làm sạch khuyết & ngoại lai
  3. Thống kê Nhân khẩu học & Mô tả (Tần số, Tỷ lệ, Skewness, Kurtosis)
  4. Kiểm định Độ tin cậy thang đo (Cronbach's Alpha & Item-Total Correlation)
  5. So sánh Trung bình (Independent Samples T-Test & One-Way ANOVA)
  6. Phân tích Nhân tố Khám phá (EFA - PCA & PAF, Varimax & Promax)
  7. Phân tích Nhân tố Khẳng định (CFA) & Mô hình Cấu trúc CB-SEM
  8. Mô hình Cấu trúc PLS-SEM (Bootstrapping, HTMT, Phân tích Trung gian Mediation)
  9. Machine Learning trong NCKH (XGBoost, PCA, Clustering, NLP Topic Modeling)
  10. Xuất Báo cáo Học thuật (Word .docx trọn gói chuẩn APA 7th, Excel đa sheet)
- **Môi trường vận hành**: Trình duyệt Web máy tính để bàn và laptop (Chrome, Edge, Safari, Firefox), độ phân giải từ 1280px trở lên, tương thích môi trường Cloud (`.streamlit.app`) và Localhost (`localhost:8502`).

## Capabilities and Constraints

- **Khả năng cốt lõi**:
  - Hỗ trợ đầy đủ các định dạng dữ liệu nghiên cứu: SPSS (.sav, .zsav), Excel (.xlsx, .xls) và CSV (.csv).
  - Khôi phục trạng thái và hoàn tác (Audit Trail History Manager với Data Snapshot Rollback).
  - Trực quan hóa tương tác cao cấp với Plotly: Sơ đồ đường dẫn cấu trúc (Path Diagram), biểu đồ cơ cấu, phân phối chuẩn.
  - Tự động sinh nhận xét học thuật (Academic Narrative) chuẩn APA-7 dựa trên các ngưỡng lý thuyết kinh điển (Hair et al., Kline, Fornell & Larcker, Henseler).
- **Ràng buộc bất biến**:
  - **Bảo mật dữ liệu**: Xử lý hoàn toàn trong bộ nhớ session cục bộ, tuyệt đối không lưu trữ dữ liệu khảo sát nhạy cảm của người dùng lên server bên ngoài.
  - **Chuẩn xác toán học**: Kết quả phân tích (giá trị p, hệ số Beta, F, t, Chi-square, CFI, RMSEA, HTMT) phải chuẩn xác tuyệt đối, khớp với kết quả từ SPSS và SmartPLS.
  - **Giao diện & Trải nghiệm**: Giữ nguyên vẹn thiết kế thanh tác vụ tinh giản (Stepper Navigation) với nút mũi tên cuộn không viền, các ô radio tùy chọn có khung viền bo cong hiện đại.

## Brand Commitments

- **Tên thương hiệu**: SmartResearch Studio (StaNDTistics • v2.5 Academic Edition).
- **Tác giả & Bản quyền**: NgDuyTien (Duy Tiến).
- **Hệ màu chủ đạo**:
  - Royal Academic Blue (`#1d4ed8`, `#2563eb`): Thể hiện sự chuẩn mực, tin cậy, thông tuệ học thuật.
  - Slate Dark (`#0f172a`, `#1e293b`): Thanh lịch, tương phản cao, hiện đại.
  - Surface Background (`#edf2f7`, `#ffffff`): Tươi sáng, dễ đọc, giảm mỏi mắt khi phân tích dữ liệu dài giờ.
  - Accent Success (`#10b981`), Warning (`#f59e0b`), Info (`#3b82f6`): Chỉ báo trạng thái thống kê đạt chuẩn/chưa đạt.
- **Giọng văn (Tone of Voice)**: Khách quan, chuẩn xác, chuẩn mực văn phong học thuật nghiên cứu khoa học, lịch sự và hỗ trợ tận tâm.

## Evidence on Hand

- **Bộ dữ liệu mẫu thực nghiệm**: `sample_data/social_research_data.sav` (350 quan sát, 26 biến khảo sát Likert và nhân khẩu học).
- **Bộ kiểm thử toàn diện**: `test_suite.py` bao quát 10/10 module phân tích từ Data Loader, Cronbach's Alpha, EFA, CFA, PLS-SEM đến Export Word/Excel và History Manager.
- **Tài sản thương hiệu**: Logo StaNDTistics biểu tượng biểu đồ tăng trưởng kết nối tri thức (`assets/`).

## Product Principles

1. **Chuẩn mực học thuật là tối thượng**: Mọi thuật toán, ngưỡng đánh giá (cut-off criteria) và bảng kết quả đều phải tuân thủ nghiêm ngặt các tiêu chuẩn khoa học quốc tế (APA 7th, Hair et al., 2022).
2. **Không rào cản kỹ thuật**: Biến những phép toán mô hình cấu trúc phức tạp thành trải nghiệm trực quan, dễ thao tác, có giải thích ngữ nghĩa rõ ràng cho từng chỉ số.
3. **Liền mạch và khép kín**: Người dùng hoàn thành toàn bộ công trình phân tích từ A đến Z trên một nền tảng duy nhất, kết quả đầu ra là báo cáo sẵn sàng nộp hoặc đăng bài.
4. **Bảo mật và an tâm**: Tôn trọng tính bảo mật của dữ liệu khoa học; không thu thập hay thương mại hóa tập dữ liệu của nhà nghiên cứu.
5. **Giao diện thủ công tinh tế (Craft & Precision)**: Từng bảng biểu, nhãn số, nút cuộn, chỉ báo trạng thái đều được chăm chút tỉ mỉ, ngăn nắp, không có chi tiết thừa.

## Accessibility & Inclusion

- Tương phản màu sắc văn bản và nền đạt chuẩn WCAG AA trên toàn bộ giao diện bảng số liệu và thẻ KPI.
- Hỗ trợ thẻ thuộc tính `translate="no"` và class `notranslate` để ngăn trình duyệt tự động làm sai lệch ký hiệu học thuật quốc tế.
- Hỗ trợ tooltip giải nghĩa chi tiết khi rê chuột vào các chỉ số thống kê viết tắt (CMIN, CFI, RMSEA, HTMT, AVE, CR).
