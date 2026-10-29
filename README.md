# SmartResearch Studio - Nền tảng Phân tích Thống kê Toàn diện cho NCKH Xã hội

> **Tác giả phát triển:** **NgDuyTien**  
> **Phiên bản:** 2.5 Academic Edition  
> **Mục đích:** Xử lý dữ liệu định lượng, kiểm định mô hình phương trình cấu trúc (CB-SEM & PLS-SEM) và Machine Learning trong Nghiên cứu Khoa học Xã hội.

**SmartResearch Studio** là phần mềm web-based/desktop chuyên dụng cho các nhà nghiên cứu, giảng viên, nghiên cứu sinh và sinh viên trong khối ngành Khoa học Xã hội (Kinh tế, Quản trị kinh doanh, Marketing, Tài chính, Xã hội học, Tâm lý học, Giáo dục...).

Ứng dụng tích hợp toàn bộ quy trình xử lý dữ liệu định lượng khoa học từ cơ bản đến nâng cao trong một nền tảng thống nhất, thay thế việc phải mua bản quyền và luân chuyển qua lại giữa nhiều phần mềm đắt đỏ như **SPSS, AMOS, và SmartPLS**.

---

## 🌟 Tính năng nổi bật & Chuẩn mực Toán học

Mọi phép tính trong SmartResearch Studio đều được xây dựng dựa trên nền tảng toán học chuẩn xác, tuân thủ nghiêm ngặt các quy ước học thuật quốc tế (APA 7th, Hair et al. 2022, Kline 2015, Hu & Bentler 1999):

### 1. Quản lý Dữ liệu & Hỗ trợ Đa Định dạng
- **Hỗ trợ tệp SPSS (`.sav`, `.zsav`)**: Tự động đọc và lưu giữ toàn bộ siêu dữ liệu (Metadata), bao gồm **Nhãn biến (Variable Labels)** và **Nhãn giá trị (Value Labels)** (ví dụ: `1 = Nam, 2 = Nữ; 1 = Rất không đồng ý ... 5 = Rất đồng ý`).
- **Hỗ trợ tệp Excel (`.xlsx`, `.xls`) và `.csv`**.
- Xử lý dữ liệu khuyết tật (Missing values): Loại bỏ theo dòng (Listwise deletion), Pairwise, hoặc gán giá trị trung bình/trung vị (Imputation).

### 2. Thống kê Mô tả & Phân tích Nhân khẩu học (Demographics)
- Bảng tần số (Frequency), tỷ lệ phần trăm (Percentage), tỷ lệ hợp lệ và tỷ lệ tích lũy.
- Thống kê mô tả định lượng: Cỡ mẫu $N$, Trung bình (Mean), Sai số chuẩn (SE Mean), Độ lệch chuẩn (Std. Dev), Phương sai (Variance), Min, Max.
- Kiểm định phân phối chuẩn: Hệ số bất đối xứng (Skewness) và Độ nhọn (Kurtosis) theo tiêu chuẩn **Kline (2015)** ($|Skew| < 3, |Kurt| < 10$) và **Hair et al. (2022)** ($|Skew| < 1, |Kurt| < 1$).
- Trực quan hóa tương tác: Biểu đồ tròn (Donut), Biểu đồ cột, Histogram phân phối tần số với đường cong mật độ KDE.

### 3. Kiểm định Độ tin cậy Thang đo (Cronbach's Alpha)
- Hệ số tin cậy tổng thể $\alpha \ge 0.70$ (hoặc $\ge 0.60$ đối với nghiên cứu khám phá ban đầu).
- Hệ số tương quan biến - tổng hiệu chỉnh (Corrected Item-Total Correlation) $\ge 0.30$.
- Hệ số Alpha nếu loại biến (Cronbach's Alpha if Item Deleted).
- Tự động gắn nhãn và cảnh báo biến rác cần loại bỏ.
- Chế độ phân tích hàng loạt (Batch analysis) tất cả các thang đo cùng lúc.

### 4. Kiểm định So sánh Trung bình (T-Test & ANOVA)
- **Independent Samples T-Test (So sánh 2 nhóm độc lập):**
  - Kiểm định tính đồng nhất của phương sai Levene's Test ($F$, $p$).
  - Tự động đối chiếu và áp dụng kiểm định Student's t (khi phương sai bằng nhau) hoặc Welch's t (khi phương sai khác nhau).
  - Khoảng tin cậy 95% của độ chênh lệch trung bình và kích thước ảnh hưởng Cohen's $d$.
- **One-Way ANOVA (So sánh từ 2 nhóm trở lên):**
  - Bảng phân tích phương sai tổng hợp (Between Groups, Within Groups, Total; $SS, df, MS, F, p$).
  - Kích thước ảnh hưởng Eta-squared ($\eta^2$).
  - Kiểm định so sánh cặp Post-Hoc Tukey HSD với mức ý nghĩa hiệu chỉnh $p_{\text{adj}} < 0.05$.

### 5. Phân tích Nhân tố Khám phá (EFA - Exploratory Factor Analysis)
- Kiểm định độ thích hợp của mẫu: Hệ số $KMO \ge 0.50$ (đánh giá từ Meritorious, Good đến Poor).
- Kiểm định độ cầu Bartlett (Bartlett's Test of Sphericity) với mức ý nghĩa $p < 0.05$.
- Phương pháp trích nhân tố: Principal Components Analysis (PCA) hoặc Principal Axis Factoring (PAF).
- Phương pháp xoay nhân tố: Phép xoay vuông góc Varimax hoặc phép xoay xiên Promax.
- Tiêu chuẩn trích nhân tố: Tiêu chuẩn Kaiser (Eigenvalue $\ge 1.0$), Tổng phương sai trích (Cumulative Variance) $\ge 50\%$, biểu đồ Scree Plot.
- Ma trận xoay (Rotated Component Matrix / Pattern Matrix): Tự động phát hiện biến có hệ số tải thấp ($< 0.50$) và hiện tượng tải kép (Cross-loading với khoảng cách tải $\Delta < 0.30$).

### 6. CFA & CB-SEM (Mô hình Phương trình Cấu trúc dựa trên Hiệp phương sai)
- **Độ phù hợp tổng thể mô hình (Model Fit Indices):**
  - Chi-square ($\chi^2$), Bậc tự do ($df$), Tỷ số $CMIN/df < 3.0$ (Tốt) hoặc $< 5.0$ (Chấp nhận).
  - CFI (Comparative Fit Index) $\ge 0.90$.
  - TLI (Tucker-Lewis Index) $\ge 0.90$.
  - GFI $\ge 0.90$, AGFI $\ge 0.80$.
  - RMSEA (Root Mean Square Error of Approximation) $< 0.08$ (Tốt $< 0.05$).
  - AIC, BIC phục vụ so sánh mô hình cạnh tranh.
- **Độ hội tụ (Convergent Validity):**
  - Trọng số tải nhân tố chuẩn hóa $\lambda \ge 0.50$.
  - Độ tin cậy tổng hợp $CR = \frac{(\sum \lambda)^2}{(\sum \lambda)^2 + \sum(1 - \lambda^2)} \ge 0.70$.
  - Phương sai trích trung bình $AVE = \frac{\sum \lambda^2}{k} \ge 0.50$.
- **Độ phân biệt (Discriminant Validity):**
  - Tiêu chuẩn Fornell & Larcker (1981): Đường chéo chứa $\sqrt{AVE}$ phải lớn hơn tương quan ngoài chéo.
  - Tỷ số tương quan dị biệt - đơn thuộc tính (HTMT Matrix) $< 0.85$ hoặc $< 0.90$.
- **Mô hình cấu trúc (CB-SEM Paths):**
  - Hệ số hồi quy chuẩn hóa $\beta$, chưa chuẩn hóa $B$, sai số chuẩn $SE$, giá trị tới hạn C.R ($z$), mức ý nghĩa $p$.

### 7. PLS-SEM (Mô hình Bình phương bé nhất từng phần)
- Triển khai chính xác thuật toán hội tụ lặp Wold's PLS algorithm với Path Weighting Scheme (theo cẩm nang của Hair, Hult, Ringle, & Sarstedt, 2022).
- **Mô hình đo lường (Outer Model):** Outer loadings ($\ge 0.708$), Indicator reliability, Cronbach's $\alpha$, Composite Reliability $\rho_c \ge 0.70$, Dijkstra-Henseler's $\rho_A \ge 0.70$, $AVE \ge 0.50$.
- **Mô hình cấu trúc (Inner Model):**
  - Kiểm tra đa cộng tuyến: Inner VIF $< 3.0$.
  - Hệ số giải thích $R^2$ và $R^2$ hiệu chỉnh.
  - Kích thước tác động $f^2 = \frac{R^2_{\text{incl}} - R^2_{\text{excl}}}{1 - R^2_{\text{incl}}}$ (0.02 nhỏ, 0.15 trung bình, 0.35 lớn).
  - Thuật toán Bootstrapping phi tham số (500 - 2000 lần lặp lại) để tính sai số chuẩn, $t$-statistic, $p$-value và khoảng tin cậy 95% (Percentile).
- **Phân tích tác động trung gian (Mediation Analysis):**
  - Tính tác động gián tiếp cụ thể (Specific indirect effect $a \times b$), tác động trực tiếp ($c'$), tác động tổng thể ($c$).
  - Tỷ lệ giải thích $VAF = \frac{\text{Indirect}}{\text{Total}}$.
  - Phân loại trung gian theo thang phân loại học thuật của Zhao, Lynch, & Chen (2010): Trung gian toàn phần (Full), Trung gian một phần bổ sung (Complementary), Trung gian một phần cạnh tranh (Competitive), hoặc Không có trung gian.

### 8. Machine Learning & Trí tuệ Nhân tạo trong NCKH Xã hội
- **Hệ thống Cố vấn Phương pháp Tự động (AI Methodology Advisor):**
  - Tự động quét kích thước mẫu ($N$), số lượng thuộc tính ($p$), loại biến mục tiêu (Nhị phân 0/1, Liên tục, Văn bản câu hỏi mở).
  - Đánh giá mức độ phù hợp: 🟢 Rất phù hợp | 🟡 Có thể xem xét | 🔴 Chưa phù hợp.
  - Cung cấp cẩm nang chi tiết: Bản chất mô hình, công thức toán học hàm mục tiêu/mất mát bằng LaTeX, ý nghĩa ứng dụng và cách biện luận trong bài báo Scopus/ISI, ưu & nhược điểm.
- **Học có giám sát (Supervised Learning):**
  - **Hồi quy Quy chuẩn hóa (Regularized Regression):** Ridge ($L_2$), Lasso ($L_1$), Elastic Net ($L_1 + L_2$) - giải quyết đa cộng tuyến và tự động lựa chọn biến quan trọng (Feature Selection).
  - **Cây quyết định & Học tập hợp:** Decision Tree, Random Forest (Feature Importance ranking), Gradient Boosting (GBM).
  - **Mô hình Boosting hiện đại:** **XGBoost** và **LightGBM** tối đa hóa khả năng giải thích và dự báo.
  - **Máy Vectơ Hỗ trợ (SVM):** SVC và SVR với Kernel RBF và Tuyến tính.
  - **Hồi quy Logistic Nhị phân:** Tỷ số chênh Odds Ratios ($OR = e^B$), kiểm định Wald, ROC-AUC.
- **Học không giám sát (Unsupervised Learning):**
  - **Phân cụm K-Means:** Đánh giá số cụm tối ưu qua Elbow Inertia và Silhouette Score, Bảng hồ sơ đặc trưng cụm (Cluster Profiles) phục vụ phân khúc thị trường.
  - **Phân cụm Phân tầng (Hierarchical Clustering):** Phương pháp liên kết Ward's Linkage, phân tích cấu trúc cây phân nhóm.
  - **Phân tích Thành phần Chính (PCA):** Nén chiều dữ liệu, bảng phương sai trích, ma trận hệ số tải và biểu đồ chiếu 2D Biplot.
- **Mạng Nơ-ron & Xử lý Ngôn ngữ Tự nhiên (NLP):**
  - **Mạng Nơ-ron Nhân tạo (ANN / Multi-Layer Perceptron):** Mô hình hóa quan hệ phi tuyến tính phức tạp (ứng dụng phương pháp kết hợp Two-stage SEM-ANN).
  - **Phân tích Cảm xúc Văn bản (Sentiment Analysis):** Tính điểm phân cực Polarity Score $[-1.0, +1.0]$, phân loại Tích cực / Trung lập / Tiêu cực cho câu trả lời mở tiếng Việt & tiếng Anh.
  - **Mô hình Chủ đề Tiềm ẩn (Latent Dirichlet Allocation - LDA):** Trích xuất tự động các chủ đề thảo luận (Topics) và từ khóa then chốt (Keywords) từ ý kiến phản hồi của đáp viên.

### 9. Xuất Báo cáo Học thuật (Word & Excel)
- **Tệp Microsoft Word (`.docx`) chuẩn APA 7th Edition:**
  - Bảng biểu được tạo đúng quy chuẩn luận văn/bài báo ISI/Scopus (3 đường kẻ ngang: trên header, dưới header, chân bảng; không có đường kẻ dọc).
  - Đoạn văn thuyết minh học thuật tự động bằng tiếng Việt với các câu trích dẫn khoa học.
- **Tệp Microsoft Excel (`.xlsx`)**: Xuất trọn bộ bảng biểu sang các sheet riêng biệt.

---

## 🚀 Hướng dẫn Khởi chạy Ứng dụng

### Cách 1: Khởi chạy nhanh bằng 1 click chuột (Khuyến nghị)
Trong thư mục `d:\Data Antigravity\Ung dung NCKH\`, bạn chỉ cần nhấp đúp vào tệp:
```text
run_app.bat
```
Ứng dụng sẽ tự động khởi động và mở giao diện trình duyệt web tại địa chỉ `http://localhost:8501`.

### Cách 2: Khởi chạy bằng dòng lệnh
Mở PowerShell hoặc CMD tại thư mục dự án và chạy:
```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m streamlit run app.py
```

---

## 📁 Cấu trúc Thư mục Dự án

```text
d:\Data Antigravity\Ung dung NCKH\
├── app.py                     # Ứng dụng giao diện chính Streamlit
├── run_app.bat                # Phím tắt khởi chạy ứng dụng 1 chạm
├── test_suite.py              # Bộ kiểm thử tự động toàn diện 9 module
├── README.md                  # Tài liệu hướng dẫn sử dụng chi tiết
├── core/                      # Các động cơ toán học và thuật toán thống kê
│   ├── __init__.py
│   ├── data_loader.py         # Đọc SPSS (.sav) & Excel, quản lý nhãn và dữ liệu khuyết
│   ├── descriptives.py        # Thống kê tần số, Mean, SD, Skewness, Kurtosis
│   ├── reliability.py         # Cronbach's Alpha, tương quan biến - tổng, alpha nếu loại
│   ├── hypothesis_tests.py    # Independent T-Test, One-Way ANOVA, Levene, Tukey HSD
│   ├── efa.py                 # EFA, KMO, Bartlett, PCA/PAF, Varimax/Promax, tải kép
│   ├── cfa_cbsem.py           # CFA & CB-SEM: CMIN/df, CFI, TLI, RMSEA, CR, AVE, Fornell, HTMT
│   ├── pls_sem.py             # PLS-SEM: Wold's algorithm, Bootstrapping, HTMT, f2, mediation
│   ├── machine_learning.py    # Logistic Regression (Odds Ratios), Random Forest, K-Means
│   └── report_exporter.py     # Xuất bảng Word (.docx chuẩn APA 7) và Excel (.xlsx)
├── sample_data/               # Bộ dữ liệu mẫu nghiên cứu thực nghiệm
│   ├── generate_sample.py     # Script tái lập bộ dữ liệu thực nghiệm 350 mẫu
│   ├── social_research_data.sav   # Tệp SPSS có đầy đủ Variable Labels & Value Labels
│   └── social_research_data.xlsx  # Tệp Excel tương ứng
└── assets/
    └── style.css              # Giao diện học thuật trang nhã, hiện đại
```

---

## 📚 Tài liệu Tham khảo Học thuật làm Căn cứ Thuật toán

1. **Hair, J. F., Hult, G. T. M., Ringle, C. M., & Sarstedt, M. (2022).** *A Primer on Partial Least Squares Structural Equation Modeling (PLS-SEM)* (3rd ed.). SAGE Publications.
2. **Hair, J. F., Black, W. C., Babin, B. J., & Anderson, R. E. (2019).** *Multivariate Data Analysis* (8th ed.). Cengage Learning.
3. **Hu, L. T., & Bentler, P. M. (1999).** Cutoff criteria for fit indexes in covariance structure analysis: Conventional criteria versus new alternatives. *Structural Equation Modeling: A Multidisciplinary Journal*, 6(1), 1-55.
4. **Kline, R. B. (2015).** *Principles and Practice of Structural Equation Modeling* (4th ed.). Guilford Publications.
5. **Nunnally, J. C., & Bernstein, I. H. (1994).** *Psychometric Theory* (3rd ed.). McGraw-Hill.
6. **Henseler, J., Ringle, C. M., & Sarstedt, M. (2015).** A new criterion for assessing discriminant validity in variance-based structural equation modeling. *Journal of the Academy of Marketing Science*, 43(1), 115-135.
7. **Zhao, X., Lynch, J. G., & Chen, Q. (2010).** Reconsidering Baron and Kenny: Myths and truths about mediation analysis. *Journal of Consumer Research*, 37(2), 197-206.
