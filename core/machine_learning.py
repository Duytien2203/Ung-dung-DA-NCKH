"""
Machine Learning Suite for Social Science Research (NCKH Xã hội)
Comprehensive implementation of:
1. Supervised Learning:
   - Regularized Regression: Ridge (L2), Lasso (L1), Elastic Net (L1+L2)
   - Decision Trees & Random Forest
   - Gradient Boosting: XGBoost, LightGBM, Sklearn GradientBoosting
   - Support Vector Machines (SVM - SVC/SVR with Linear/RBF)
   - Artificial Neural Networks (ANN / Multi-Layer Perceptron MLP)
2. Unsupervised Learning:
   - K-Means Clustering (Elbow, Silhouette, Cluster Profiling)
   - Hierarchical Agglomerative Clustering (Dendrogram, Ward linkage)
   - Principal Component Analysis (PCA - Dimensionality Reduction & Biplot)
   - Latent Dirichlet Allocation (LDA - Topic Modeling for survey texts)
3. NLP & Sentiment Analysis (Phân tích cảm xúc phản hồi đáp viên Tiếng Việt & English)
4. AI Auto-Recommendation & Methodology Advisor Engine:
   - Scans dataset types, recommends suitable models, provides LaTeX formulas & academic interpretations.
"""

import numpy as np
import pandas as pd
from scipy import stats
import re
import mimetypes
mimetypes.init = lambda *args, **kwargs: None

# Scikit-Learn
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor, 
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
    roc_curve, confusion_matrix, r2_score, mean_squared_error, mean_absolute_error,
    silhouette_score
)
from scipy.cluster.hierarchy import linkage, dendrogram

# Optional XGBoost & LightGBM
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


# ==============================================================================
# 1. KNOWLEDGE BASE & METHODOLOGY ADVISOR FOR SOCIAL SCIENCE RESEARCHERS
# ==============================================================================

ML_KNOWLEDGE_BASE = {
    "Logistic Regression": {
        "name": "Hồi quy Logistic (Logistic Regression)",
        "category": "Học có giám sát (Phân loại)",
        "description": "Mô hình hóa xác suất xảy ra của một biến cố nhị phân hoặc đa danh mục (ví dụ: Quyết định mua hàng: 0/1, Rời bỏ dịch vụ: 0/1, Đồng ý: 0/1).",
        "math_formula": r"""
        P(Y = 1 \mid X) = \sigma(z) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \dots + \beta_p X_p)}}
        \quad \Longleftrightarrow \quad 
        \ln\left(\frac{P}{1 - P}\right) = \beta_0 + \sum_{j=1}^p \beta_j X_j
        """,
        "loss_formula": r"""
        \mathcal{L}_{\text{LogLoss}}(\beta) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln(p_i) + (1 - y_i) \ln(1 - p_i) \right]
        """,
        "social_science_meaning": (
            "Trong NCKH xã hội, Hồi quy Logistic là tiêu chuẩn vàng để dự báo hành vi lựa chọn. "
            "Đặc biệt, hệ số tỷ số chênh Odds Ratio (OR = exp(B)) cho biết: khi biến độc lập tăng 1 đơn vị, "
            "xác suất đáp viên thực hiện hành vi sẽ tăng gấp OR lần. "
            "Rất thích hợp công bố trên các tạp chí Marketing, Quản trị, Xã hội học và Y tế công cộng."
        ),
        "pros": "Dễ giải thích, cung cấp giá trị Odds Ratio trực quan, kiểm định Wald có mức ý nghĩa p-value chuẩn xác.",
        "cons": "Giả định mối quan hệ tuyến tính giữa log-odds và biến độc lập; nhạy cảm với đa cộng tuyến."
    },
    
    "Regularized Regression": {
        "name": "Hồi quy Tuyến tính Hiệu chỉnh (Ridge, Lasso, Elastic Net)",
        "category": "Học có giám sát (Hồi quy / Lựa chọn biến)",
        "description": "Kỹ thuật hồi quy thu nhỏ hệ số (shrinkage) bằng cách bổ sung hình phạt L1 (Lasso) hoặc L2 (Ridge) vào hàm mất mát để chống quá khớp (Overfitting) và xử lý triệt để hiện tượng Đa cộng tuyến.",
        "math_formula": r"""
        \text{Ridge } (L_2): \quad \min_\beta \sum_{i=1}^N (y_i - X_i \beta)^2 + \lambda \sum_{j=1}^p \beta_j^2 \\
        \text{Lasso } (L_1): \quad \min_\beta \sum_{i=1}^N (y_i - X_i \beta)^2 + \lambda \sum_{j=1}^p |\beta_j| \\
        \text{Elastic Net}: \quad \min_\beta \sum_{i=1}^N (y_i - X_i \beta)^2 + \lambda \left[ \alpha \|\beta\|_1 + \frac{1-\alpha}{2} \|\beta\|_2^2 \right]
        """,
        "social_science_meaning": (
            "Khi khảo sát có quá nhiều biến hỏi (High dimensionality) hoặc giữa các biến thang đo có tương quan rất cao (Đa cộng tuyến trầm trọng làm OLS bị lệch sai số), "
            "Lasso có khả năng triệt tiêu các hệ số không quan trọng về đúng bằng 0, đóng vai trò như một công cụ Tự động Lựa chọn Biến số (Feature Selection). "
            "Elastic Net kết hợp sức mạnh của cả hai, giải quyết hoàn hảo hiện tượng nhóm biến tương quan cao."
        ),
        "pros": "Giải quyết đa cộng tuyến hoàn hảo, tự động lọc biến rác (Lasso), độ chính xác dự báo cao hơn OLS.",
        "cons": "Hệ số bị thu nhỏ (biased) nên p-value truyền thống không còn áp dụng trực tiếp như OLS."
    },

    "Decision Trees": {
        "name": "Cây Quyết định (Decision Trees)",
        "category": "Học có giám sát (Phân loại & Hồi quy)",
        "description": "Mô hình phi tham số phân tách không gian dữ liệu thành các tập con hình chữ nhật dựa trên chuỗi quy tắc phân nhánh If-Else tối ưu hóa độ tinh khiết Gini hoặc Entropy.",
        "math_formula": r"""
        \text{Gini Impurity}: \quad I_G(t) = 1 - \sum_{k=1}^K p(k \mid t)^2 \\
        \text{Entropy}: \quad H(t) = -\sum_{k=1}^K p(k \mid t) \log_2 p(k \mid t) \\
        \text{Information Gain}: \quad IG(T, a) = H(T) - \sum_{v \in vals(a)} \frac{|T_v|}{|T|} H(T_v)
        """,
        "social_science_meaning": (
            "Cây quyết định mô phỏng quá trình ra quyết định từng bước của con người. "
            "Trong NCKH, mô hình giúp phát hiện các 'ngưỡng phân đoạn' (Thresholds), ví dụ: "
            "Nếu Thu nhập > 15 triệu VÀ Tuổi < 30 thì Tỷ lệ hài lòng là 85%. "
            "Cung cấp các khuyến nghị chính sách và quản trị trực quan, dễ áp dụng vào thực tiễn."
        ),
        "pros": "Trực quan 100%, không đòi hỏi chuẩn hóa dữ liệu, bắt được các tương tác phi tuyến tính phức tạp.",
        "cons": "Dễ bị quá khớp (Overfitting) nếu không giới hạn độ sâu (Max Depth)."
    },

    "Random Forest": {
        "name": "Rừng Ngẫu nhiên (Random Forest)",
        "category": "Học có giám sát (Ensemble Bagging)",
        "description": "Thuật toán học kết hợp (Ensemble Learning) tạo ra hàng trăm cây quyết định độc lập trên các mẫu con ngẫu nhiên (Bootstrap) và tập thuộc tính con ngẫu nhiên, sau đó bỏ phiếu số đông (Voting) hoặc tính trung bình.",
        "math_formula": r"""
        \hat{f}_{\text{RF}}(x) = \frac{1}{B} \sum_{b=1}^B T_b(x; \Theta_b) \\
        \text{Feature Importance}: \quad \text{MDI}(X_j) = \frac{1}{B} \sum_{b=1}^B \sum_{t \in T_b: v(t)=j} \Delta I(t)
        """,
        "social_science_meaning": (
            "Random Forest là mô hình dự báo hành vi có độ tin cậy và chính xác vượt trội so với hồi quy truyền thống. "
            "Đặc biệt, chỉ số Tầm quan trọng của Biến (Mean Decrease Impurity / Feature Importance) cho phép "
            "xếp hạng mức độ đóng góp của từng yếu tố khảo sát vào kết quả đầu ra, "
            "giúp nhà nghiên cứu khẳng định yếu tố nào là động lực chi phối mạnh mẽ nhất."
        ),
        "pros": "Độ chính xác rất cao, kiểm soát overfitting tốt, không cần giả định phân phối chuẩn.",
        "cons": "Mô hình hộp đen (Black-box), khó diễn giải từng hệ số riêng biệt như OLS."
    },

    "Gradient Boosting": {
        "name": "Gradient Boosting (GBM, XGBoost, LightGBM)",
        "category": "Học có giám sát (Ensemble Boosting)",
        "description": "Kỹ thuật học tăng cường tuần tự, trong đó mỗi cây quyết định mới được huấn luyện để sửa chữa sai số (Residuals) của tập hợp các cây trước đó theo hướng gradient âm của hàm mất mát.",
        "math_formula": r"""
        F_m(x) = F_{m-1}(x) + \gamma_m h_m(x) \quad \text{với} \quad h_m = \arg\min_h \sum_{i=1}^N \left[ -\frac{\partial \mathcal{L}(y_i, F_{m-1}(x_i))}{\partial F_{m-1}(x_i)} - h(x_i) \right]^2 \\
        \text{XGBoost Objective}: \quad \text{Obj}^{(t)} = \sum_{i=1}^N \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t)
        """,
        "social_science_meaning": (
            "Gradient Boosting (đặc biệt là XGBoost và LightGBM) là thuật toán giành chiến thắng nhiều nhất trong các cuộc thi khoa học dữ liệu. "
            "Trong nghiên cứu xã hội định lượng, đây là công cụ tối thượng để tối đa hóa khả năng giải thích và dự báo "
            "(thường đạt AUC hoặc R2 cao hơn OLS từ 10% đến 25%). "
            "Rất giá trị khi nghiên cứu sinh muốn chứng minh mô hình đề xuất có năng lực dự báo vượt trội."
        ),
        "pros": "Năng lực dự báo mạnh nhất hiện nay, xử lý xuất sắc dữ liệu dạng bảng (Tabular data).",
        "cons": "Cần tinh chỉnh siêu tham số (Learning rate, Max depth, Subsample), yêu cầu cỡ mẫu tương đối."
    },

    "Support Vector Machine": {
        "name": "Máy Vectơ Hỗ trợ (Support Vector Machine - SVM)",
        "category": "Học có giám sát (Biên cực đại & Kernel)",
        "description": "Thuật toán tìm kiếm một siêu phẳng (Hyperplane) tối ưu trong không gian đa chiều sao cho khoảng cách biên (Margin) giữa hai lớp dữ liệu là lớn nhất.",
        "math_formula": r"""
        \min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^N \xi_i \quad \text{s.t.} \quad y_i(w^T \phi(x_i) + b) \ge 1 - \xi_i, \quad \xi_i \ge 0 \\
        \text{RBF Kernel}: \quad K(x, x') = \exp\left(-\gamma \|x - x'\|^2\right)
        """,
        "social_science_meaning": (
            "SVM đặc biệt hiệu quả khi nghiên cứu có số lượng biến lớn nhưng cỡ mẫu vừa phải (High dimension, moderate sample size). "
            "Nhờ 'Kernel Trick' (như hàm RBF), SVM có thể chuyển đổi dữ liệu phi tuyến tính sang không gian chiều vô hạn để phân tách rành mạch "
            "các nhóm đối tượng có hành vi phức tạp."
        ),
        "pros": "Tối ưu hóa lồi toàn cục (không bị kẹt ở cực tiểu địa phương), rất mạnh với số chiều lớn.",
        "cons": "Nhạy cảm với việc chuẩn hóa thang đo (bắt buộc phải Standardize trước), tính toán chậm với mẫu > 50,000 dòng."
    },

    "Artificial Neural Networks": {
        "name": "Mạng Nơ-ron Nhân tạo (Artificial Neural Networks - ANN / MLP)",
        "category": "Mạng Nơ-ron & Học sâu (Deep Learning)",
        "description": "Mô hình mạng nơ-ron nhiều tầng (Multi-Layer Perceptron) gồm Tầng đầu vào (Input), một hoặc nhiều Tầng ẩn (Hidden layers) và Tầng đầu ra (Output), học các biểu diễn phi tuyến tính qua thuật toán Lan truyền ngược (Backpropagation).",
        "math_formula": r"""
        a^{(l)} = g\left(W^{(l)} a^{(l-1)} + b^{(l)}\right) \quad \text{với hàm kích hoạt: } \text{ReLU}(z) = \max(0, z) \text{ hoặc } \sigma(z) \\
        \text{Lan truyền ngược (Gradient)}: \quad \delta^{(l)} = \left(W^{(l+1)T} \delta^{(l+1)}\right) \odot g'(z^{(l)})
        """,
        "social_science_meaning": (
            "Được mệnh danh là 'Bộ xấp xỉ phổ quát' (Universal Approximator), ANN có thể mô hình hóa bất kỳ mối quan hệ hành vi con người phức tạp nào. "
            "Xu hướng kết hợp SEM-ANN (Two-stage SEM-ANN approach): Giai đoạn 1 dùng SEM kiểm định giả thuyết lý thuyết, "
            "Giai đoạn 2 dùng ANN để đánh giá độ phi tuyến và dự báo chính xác cao, "
            "đang là phương pháp được đánh giá rất cao trên các tạp chí top-tier (Q1/Scopus)."
        ),
        "pros": "Học được mọi dạng quan hệ phi tuyến tính ẩn, độ chính xác xấp xỉ cực kỳ cao.",
        "cons": "Được coi là 'Hộp đen' hoàn toàn, đòi hỏi cỡ mẫu đủ lớn để tránh học vẹt."
    },

    "K-Means": {
        "name": "Phân cụm K-Means (K-Means Clustering)",
        "category": "Học không giám sát (Phân cụm đối tượng)",
        "description": "Thuật toán phân chia $N$ đối tượng vào $K$ cụm phân biệt sao cho tổng bình phương khoảng cách Euclid từ mỗi điểm đến tâm cụm (Centroid) tương ứng là nhỏ nhất.",
        "math_formula": r"""
        \arg\min_S \sum_{i=1}^K \sum_{x \in S_i} \|x - \mu_i\|^2 \quad \text{với} \quad \mu_i = \frac{1}{|S_i|} \sum_{x \in S_i} x \\
        \text{Silhouette Score}: \quad s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))} \in [-1, 1]
        """,
        "social_science_meaning": (
            "Là công cụ chuẩn mực trong Nghiên cứu Marketing và Quản trị kinh doanh để Phân khúc Khách hàng / Thị trường (Market Segmentation). "
            "Dựa trên các câu hỏi Likert về lối sống, thái độ và nhận thức, K-Means giúp xác định các nhóm chân dung đối tượng "
            "(ví dụ: Nhóm nhạy cảm về giá, Nhóm trung thành công nghệ, Nhóm thụ động). "
            "Bảng hồ sơ cụm (Cluster Profiles) là cơ sở đề xuất hàm ý quản trị đắt giá."
        ),
        "pros": "Tốc độ tính toán siêu nhanh, kết quả phân cụm rõ ràng, dễ định danh nhóm đối tượng.",
        "cons": "Phải xác định trước số cụm K (cần dùng biểu đồ Elbow và Silhouette để chọn K tối ưu)."
    },

    "Hierarchical Clustering": {
        "name": "Phân cụm Phân tầng (Hierarchical Clustering)",
        "category": "Học không giám sát (Phân cụm cấu trúc cây)",
        "description": "Thuật toán xây dựng một hệ thống phân cấp các cụm lồng ghép theo phương pháp kết hợp từ dưới lên (Agglomerative), trực quan hóa bằng Sơ đồ cây (Dendrogram).",
        "math_formula": r"""
        \text{Ward's Linkage}: \quad \Delta ESS = \frac{n_A n_B}{n_A + n_B} \|\mu_A - \mu_B\|^2 \\
        d(u, v) = \min / \max / \text{avg} \{ \text{dist}(x, y): x \in u, y \in v \}
        """,
        "social_science_meaning": (
            "Giúp nhà nghiên cứu quan sát được mức độ tương đồng và trật tự liên kết giữa các nhóm quan sát ở các mức độ khái quát khác nhau. "
            "Sơ đồ Dendrogram cung cấp bằng chứng hình học thuyết phục về số lượng phân khúc tự nhiên của mẫu nghiên cứu "
            "mà không cần áp đặt số cụm trước."
        ),
        "pros": "Trực quan hóa tuyệt vời qua biểu đồ Dendrogram, không cần chọn trước số cụm K.",
        "cons": "Độ phức tạp tính toán $O(N^2)$ hoặc $O(N^3)$, chỉ phù hợp với cỡ mẫu $N \le 2,000$."
    },

    "Principal Component Analysis": {
        "name": "Phân tích Thành phần Chính (PCA - Principal Component Analysis)",
        "category": "Học không giám sát (Giảm chiều dữ liệu)",
        "description": "Kỹ thuật biến đổi trực giao tập hợp các biến số có tương quan thành một tập hợp các biến mới không tương quan tuyến tính gọi là các Thành phần chính (Principal Components), tối đa hóa phương sai được giữ lại.",
        "math_formula": r"""
        \mathbf{Y} = \mathbf{X} \mathbf{W} \quad \text{s.t.} \quad \mathbf{W}^T \mathbf{W} = \mathbf{I} \\
        \text{Cov}(\mathbf{X}) \mathbf{v}_i = \lambda_i \mathbf{v}_i \quad \Longrightarrow \quad \text{Explained Variance Ratio} = \frac{\lambda_i}{\sum_{j=1}^p \lambda_j}
        """,
        "social_science_meaning": (
            "Trong khi EFA tìm kiếm các biến tiềm ẩn lý thuyết đằng sau các biến đo lường, thì PCA hướng đến mục tiêu thực dụng: "
            "Nén dữ liệu từ 20-30 câu hỏi khảo sát thành 2-3 chỉ số tổng hợp (Composite Indices) mà vẫn giữ được 70-80% lượng thông tin. "
            "Các thành phần chính này có thể dùng làm biến đầu vào trực tiếp cho hồi quy mà loại bỏ hoàn toàn đa cộng tuyến."
        ),
        "pros": "Loại bỏ 100% hiện tượng đa cộng tuyến giữa các thành phần, giảm chiều dữ liệu hiệu quả.",
        "cons": "Các thành phần là tổ hợp tuyến tính của tất cả các biến, khó gán ý nghĩa lý thuyết thuần túy hơn EFA xoay."
    },

    "Latent Dirichlet Allocation": {
        "name": "Mô hình Chủ đề Tiềm ẩn (Latent Dirichlet Allocation - LDA)",
        "category": "Học không giám sát & NLP (Topic Modeling)",
        "description": "Mô hình xác suất sinh (Generative probabilistic model) khám phá các chủ đề ẩn (Topics) từ tập hợp các văn bản câu trả lời mở hoặc ý kiến phản hồi của đáp viên.",
        "math_formula": r"""
        P(\mathbf{w}, \mathbf{z}, \theta, \phi \mid \alpha, \beta) = \prod_{d=1}^D P(\theta_d \mid \alpha) \prod_{n=1}^{N_d} P(z_{dn} \mid \theta_d) P(w_{dn} \mid \phi_{z_{dn}}) \prod_{k=1}^K P(\phi_k \mid \beta)
        """,
        "social_science_meaning": (
            "Khi phiếu khảo sát có các câu hỏi mở (Open-ended questions): 'Ý kiến đóng góp khác của bạn?', "
            "thay vì phải đọc và mã hóa thủ công từng phản hồi (rất tốn thời gian và dễ chủ quan), "
            "LDA tự động phân loại các ý kiến thành các chủ đề then chốt (ví dụ: Chủ đề Vận chuyển, Chủ đề Giá cả, Chủ đề Thái độ phục vụ) "
            "kèm các từ khóa đặc trưng. Đây là phương pháp nghiên cứu hỗn hợp (Mixed-methods) xu hướng hiện nay."
        ),
        "pros": "Khai phá tự động các góc nhìn ẩn từ văn bản phản hồi của người tham gia khảo sát.",
        "cons": "Cần tiền xử lý loại bỏ từ dừng (Stopwords), đòi hỏi tập văn bản có dung lượng từ nhất định."
    },

    "Sentiment Analysis": {
        "name": "Phân tích Cảm xúc Văn bản (Sentiment Analysis)",
        "category": "NLP & Khai phá Dữ liệu Ý kiến",
        "description": "Thuật toán xác định trạng thái thái độ, cảm xúc (Tích cực, Tiêu cực, Trung lập) và cường độ phân cực (Polarity Score) của người tham gia nghiên cứu qua ngôn ngữ họ diễn đạt.",
        "math_formula": r"""
        \text{Polarity Score } S \in [-1.0, +1.0] \\
        S = \frac{\sum w_{\text{pos}} - \sum w_{\text{neg}}}{\sqrt{(\sum w_{\text{pos}} - \sum w_{\text{neg}})^2 + \alpha}}
        """,
        "social_science_meaning": (
            "Giúp định lượng hóa dữ liệu định tính! Chuyển đổi các đoạn văn đánh giá của khách hàng/người dân "
            "thành một biến số liên tục (Chỉ số Cảm xúc) để đưa vào mô hình hồi quy hoặc SEM kiểm định xem "
            "'Mức độ cảm xúc tích cực' có tác động thế nào đến 'Ý định tiếp tục sử dụng' và 'Lòng trung thành'."
        ),
        "pros": "Tự động hóa hoàn toàn việc đọc hiểu cảm xúc, cầu nối hoàn hảo giữa nghiên cứu Định tính và Định lượng.",
        "cons": "Có thể gặp khó khăn với văn phong mỉa mai (Sarcasm) hoặc câu ghép phức tạp."
    }
}


def analyze_dataset_and_recommend(df: pd.DataFrame) -> dict:
    """
    Scans the uploaded DataFrame and automatically generates recommendations
    for which ML models to apply, along with data suitability scores.
    """
    n_samples, n_cols = df.shape
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c.lower() not in ['id', 'mã']]
    text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    
    # Identify binary target candidates
    binary_targets = [c for c in df.columns if df[c].dropna().nunique() == 2]
    continuous_targets = [c for c in numeric_cols if df[c].dropna().nunique() > 10]
    likert_scale_cols = [c for c in numeric_cols if 3 <= df[c].dropna().nunique() <= 7]
    
    # Check text columns with meaningful sentence length
    meaningful_text_cols = []
    for tc in text_cols:
        avg_len = df[tc].dropna().astype(str).str.len().mean()
        if avg_len > 15:
            meaningful_text_cols.append(tc)
            
    recommendations = []
    
    # 1. Classification (Logistic, Random Forest, XGBoost, SVM)
    if binary_targets:
        bt_sample = binary_targets[0]
        recommendations.append({
            "model_key": "Logistic Regression",
            "suitability": "🟢 Rất phù hợp (Highly Recommended)",
            "reason": f"Dữ liệu có biến phụ thuộc nhị phân '{bt_sample}' (0/1). Thích hợp tuyệt đối để tính Odds Ratios exp(B) và kiểm định giả thuyết quyết định hành vi.",
            "target_candidate": bt_sample,
            "task_type": "Phân loại nhị phân (Binary Classification)"
        })
        recommendations.append({
            "model_key": "Random Forest",
            "suitability": "🟢 Rất phù hợp (Highly Recommended)",
            "reason": f"Mô hình học kết hợp (Bagging) dự báo '{bt_sample}' với độ chính xác cao và trích xuất bảng xếp hạng Tầm quan trọng của các biến (Feature Importance).",
            "target_candidate": bt_sample,
            "task_type": "Phân loại nhị phân (Binary Classification)"
        })
        recommendations.append({
            "model_key": "Gradient Boosting",
            "suitability": "🟢 Rất phù hợp (Highly Recommended)",
            "reason": f"Thuật toán XGBoost / LightGBM tối đa hóa chỉ số ROC-AUC và độ chính xác dự báo hành vi '{bt_sample}'.",
            "target_candidate": bt_sample,
            "task_type": "Phân loại nhị phân (Binary Classification)"
        })
        recommendations.append({
            "model_key": "Support Vector Machine",
            "suitability": "🟡 Có thể xem xét (Applicable)",
            "reason": f"Phù hợp nếu muốn phân tách ranh giới biên cực đại với Kernel RBF giữa 2 nhóm '{bt_sample}'.",
            "target_candidate": bt_sample,
            "task_type": "Phân loại nhị phân (Binary Classification)"
        })
        recommendations.append({
            "model_key": "Artificial Neural Networks",
            "suitability": "🟡 Có thể xem xét (Applicable)",
            "reason": f"Áp dụng phương pháp Two-stage SEM-ANN để mô hình hóa mối quan hệ phi tuyến dự báo '{bt_sample}'.",
            "target_candidate": bt_sample,
            "task_type": "Mạng Nơ-ron (MLP)"
        })
    else:
        recommendations.append({
            "model_key": "Logistic Regression",
            "suitability": "🔴 Chưa có biến nhị phân phù hợp",
            "reason": "Chưa tìm thấy cột biến có đúng 2 giá trị nhị phân (0 và 1) làm biến mục tiêu.",
            "target_candidate": None,
            "task_type": "Phân loại"
        })

    # 2. Continuous Prediction (Regularized Regression - Ridge, Lasso, Elastic Net)
    if continuous_targets:
        ct_sample = continuous_targets[0]
        recommendations.append({
            "model_key": "Regularized Regression",
            "suitability": "🟢 Rất phù hợp (Highly Recommended)",
            "reason": f"Dữ liệu có biến mục tiêu liên tục '{ct_sample}' và nhiều thuộc tính đo lường. Lasso và Elastic Net sẽ tự động chọn lọc biến quan trọng và chống đa cộng tuyến.",
            "target_candidate": ct_sample,
            "task_type": "Hồi quy & Lọc biến (Feature Selection)"
        })
    elif likert_scale_cols and len(likert_scale_cols) >= 3:
        target_likert = likert_scale_cols[-1]
        recommendations.append({
            "model_key": "Regularized Regression",
            "suitability": "🟡 Có thể xem xét (Applicable)",
            "reason": f"Có thể dự báo biến thang đo Likert '{target_likert}' từ các biến khảo sát còn lại bằng Ridge/Lasso.",
            "target_candidate": target_likert,
            "task_type": "Hồi quy & Lọc biến"
        })

    # 3. Unsupervised Clustering (K-Means, Hierarchical)
    if len(likert_scale_cols) >= 3 or len(numeric_cols) >= 4:
        recommendations.append({
            "model_key": "K-Means",
            "suitability": "🟢 Rất phù hợp (Highly Recommended)",
            "reason": f"Tập dữ liệu có {len(numeric_cols)} biến định lượng/thang đo. Rất lý tưởng để Phân khúc Khách hàng/Đáp viên thành 2-4 nhóm chân dung đặc trưng.",
            "target_candidate": "Tập hợp các biến thang đo",
            "task_type": "Phân cụm không giám sát"
        })
        recommendations.append({
            "model_key": "Hierarchical Clustering",
            "suitability": "🟢 Rất phù hợp (Highly Recommended)" if n_samples <= 1000 else "🟡 Mẫu lớn (Tính toán chậm hơn)",
            "reason": f"Vẽ sơ đồ hình cây Dendrogram để xác định trật tự phân nhóm tự nhiên của {n_samples} đáp viên.",
            "target_candidate": "Tập hợp các biến thang đo",
            "task_type": "Phân cụm cấu trúc cây"
        })
        recommendations.append({
            "model_key": "Principal Component Analysis",
            "suitability": "🟢 Rất phù hợp (Highly Recommended)",
            "reason": f"Nén {len(numeric_cols)} biến đo lường thành 2 hoặc 3 Thành phần chính (Principal Components) không tương quan để vẽ biểu đồ Biplot 2D.",
            "target_candidate": "Tập hợp các biến thang đo",
            "task_type": "Giảm chiều dữ liệu"
        })

    # 4. Text & NLP
    if meaningful_text_cols:
        txt_sample = meaningful_text_cols[0]
        recommendations.append({
            "model_key": "Sentiment Analysis",
            "suitability": "🟢 Rất phù hợp (Highly Recommended)",
            "reason": f"Phát hiện cột văn bản ý kiến phản hồi '{txt_sample}'. Có thể chấm điểm cảm xúc (Polarity Score) Tích cực/Tiêu cực tự động.",
            "target_candidate": txt_sample,
            "task_type": "Xử lý Ngôn ngữ Tự nhiên (NLP)"
        })
        recommendations.append({
            "model_key": "Latent Dirichlet Allocation",
            "suitability": "🟢 Rất phù hợp (Highly Recommended)",
            "reason": f"Tự động trích xuất các chủ đề thảo luận tiềm ẩn (Topics & Keywords) từ cột ý kiến '{txt_sample}'.",
            "target_candidate": txt_sample,
            "task_type": "Mô hình Chủ đề Tiềm ẩn"
        })
    else:
        recommendations.append({
            "model_key": "Sentiment Analysis",
            "suitability": "🟡 Cần cột văn bản câu hỏi mở",
            "reason": "Chưa có cột văn bản ý kiến phản hồi dài của đáp viên. Nếu bổ sung câu hỏi mở, hệ thống sẽ tự động kích hoạt.",
            "target_candidate": None,
            "task_type": "NLP"
        })
        recommendations.append({
            "model_key": "Latent Dirichlet Allocation",
            "suitability": "🟡 Cần cột văn bản câu hỏi mở",
            "reason": "Chưa có cột văn bản ý kiến phản hồi mở để trích xuất chủ đề Topic Modeling.",
            "target_candidate": None,
            "task_type": "NLP"
        })

    return {
        "n_samples": n_samples,
        "n_cols": n_cols,
        "numeric_cols": numeric_cols,
        "binary_targets": binary_targets,
        "continuous_targets": continuous_targets,
        "meaningful_text_cols": meaningful_text_cols,
        "recommendations": recommendations
    }


# ==============================================================================
# 2. ADVANCED SUPERVISED LEARNING ENGINES
# ==============================================================================

def run_advanced_supervised(df: pd.DataFrame, target_col: str, feature_cols: list, 
                            algorithm: str = "Random Forest", **kwargs) -> dict:
    """
    Unified Supervised Learning Engine supporting:
    - Logistic Regression, Ridge, Lasso, Elastic Net
    - Decision Tree, Random Forest
    - Gradient Boosting (Sklearn / XGBoost / LightGBM)
    - Support Vector Machine (Linear / RBF)
    - Artificial Neural Network (MLP)
    """
    clean_df = df[[target_col] + feature_cols].dropna().copy()
    y_raw = clean_df[target_col].values
    X_raw = clean_df[feature_cols].values
    n_samples, n_features = X_raw.shape
    
    if n_samples < 15:
        return {"error": "Cỡ mẫu quá nhỏ để huấn luyện mô hình Machine Learning (cần ít nhất 15 dòng hợp lệ)."}
        
    # Standardize features
    means = X_raw.mean(axis=0)
    stds = X_raw.std(axis=0, ddof=1)
    stds[stds == 0] = 1.0
    X_scaled = (X_raw - means) / stds
    
    # Determine task: Classification or Regression
    unique_y = np.unique(y_raw)
    is_classification = len(unique_y) <= 5 or not np.issubdtype(y_raw.dtype, np.number)
    
    if is_classification:
        # Encode target
        classes = sorted(unique_y)
        class_mapping = {c: idx for idx, c in enumerate(classes)}
        y = np.array([class_mapping[val] for val in y_raw])
        is_binary = len(classes) == 2
    else:
        y = y_raw.astype(float)
        is_binary = False

    # Instantiate Model
    model = None
    coef_table = pd.DataFrame()
    importance_table = pd.DataFrame()
    
    if algorithm == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, solver='lbfgs')
        model.fit(X_scaled, y)
        
    elif algorithm == "Ridge (L2)":
        alpha = kwargs.get("alpha", 1.0)
        if is_classification:
            model = RidgeClassifier(alpha=alpha)
        else:
            model = Ridge(alpha=alpha)
        model.fit(X_scaled, y)
        
    elif algorithm == "Lasso (L1)":
        alpha = kwargs.get("alpha", 0.1)
        if is_classification:
            model = LogisticRegression(penalty='l1', solver='saga', C=1.0/alpha, max_iter=2000)
        else:
            model = Lasso(alpha=alpha)
        model.fit(X_scaled, y)
        
    elif algorithm == "Elastic Net":
        alpha = kwargs.get("alpha", 0.1)
        l1_ratio = kwargs.get("l1_ratio", 0.5)
        if is_classification:
            model = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=l1_ratio, C=1.0/alpha, max_iter=2000)
        else:
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        model.fit(X_scaled, y)
        
    elif algorithm == "Decision Tree":
        max_depth = kwargs.get("max_depth", 4)
        if is_classification:
            model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        else:
            model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        model.fit(X_scaled, y)
        
    elif algorithm == "Random Forest":
        n_estimators = kwargs.get("n_estimators", 150)
        max_depth = kwargs.get("max_depth", 6)
        if is_classification:
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_scaled, y)
        
    elif algorithm == "Gradient Boosting (GBM)":
        n_estimators = kwargs.get("n_estimators", 100)
        lr = kwargs.get("learning_rate", 0.1)
        if is_classification:
            model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=lr, random_state=42)
        else:
            model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=lr, random_state=42)
        model.fit(X_scaled, y)
        
    elif algorithm == "XGBoost" and HAS_XGBOOST:
        n_estimators = kwargs.get("n_estimators", 100)
        lr = kwargs.get("learning_rate", 0.1)
        if is_classification:
            model = xgb.XGBClassifier(n_estimators=n_estimators, learning_rate=lr, random_state=42, eval_metric='logloss')
        else:
            model = xgb.XGBRegressor(n_estimators=n_estimators, learning_rate=lr, random_state=42)
        model.fit(X_scaled, y)

    elif algorithm == "LightGBM" and HAS_LIGHTGBM:
        n_estimators = kwargs.get("n_estimators", 100)
        lr = kwargs.get("learning_rate", 0.1)
        if is_classification:
            model = lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=lr, random_state=42, verbose=-1)
        else:
            model = lgb.LGBMRegressor(n_estimators=n_estimators, learning_rate=lr, random_state=42, verbose=-1)
        model.fit(X_scaled, y)

    elif algorithm == "Support Vector Machine (SVM)":
        kernel = kwargs.get("kernel", "rbf")
        c_val = kwargs.get("C", 1.0)
        if is_classification:
            model = SVC(kernel=kernel, C=c_val, probability=True, random_state=42)
        else:
            model = SVR(kernel=kernel, C=c_val)
        model.fit(X_scaled, y)
        
    elif algorithm == "Artificial Neural Network (ANN)":
        hidden = kwargs.get("hidden_layer_sizes", (32, 16))
        max_iter = kwargs.get("max_iter", 500)
        if is_classification:
            model = MLPClassifier(hidden_layer_sizes=hidden, activation='relu', max_iter=max_iter, random_state=42)
        else:
            model = MLPRegressor(hidden_layer_sizes=hidden, activation='relu', max_iter=max_iter, random_state=42)
        model.fit(X_scaled, y)
        
    else:
        # Fallback to Random Forest
        model = RandomForestClassifier(random_state=42) if is_classification else RandomForestRegressor(random_state=42)
        model.fit(X_scaled, y)

    # ---------------- METRICS EVALUATION ----------------
    y_pred = model.predict(X_scaled)
    metrics_rows = []
    
    if is_classification:
        # Classification metrics
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
        
        auc_val = None
        roc_data = None
        if hasattr(model, "predict_proba") and is_binary:
            y_proba = model.predict_proba(X_scaled)[:, 1]
            auc_val = roc_auc_score(y, y_proba)
            fpr, tpr, _ = roc_curve(y, y_proba)
            roc_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": round(auc_val, 3)}
            
        metrics_rows.append({"Chỉ số Đánh giá": "Độ chính xác tổng thể (Accuracy)", "Giá trị": f"{acc * 100:.2f}%", "Tiêu chuẩn": ">= 70% (Tốt)"})
        metrics_rows.append({"Chỉ số Đánh giá": "Độ chuẩn xác (Precision - Weighted)", "Giá trị": f"{prec * 100:.2f}%", "Tiêu chuẩn": "Càng cao càng tốt"})
        metrics_rows.append({"Chỉ số Đánh giá": "Độ nhạy / Thu hồi (Recall - Weighted)", "Giá trị": f"{rec * 100:.2f}%", "Tiêu chuẩn": "Càng cao càng tốt"})
        metrics_rows.append({"Chỉ số Đánh giá": "F1-Score (Cân bằng Precision & Recall)", "Giá trị": f"{f1 * 100:.2f}%", "Tiêu chuẩn": ">= 0.70"})
        if auc_val is not None:
            metrics_rows.append({"Chỉ số Đánh giá": "Diện tích dưới đường cong ROC (AUC)", "Giá trị": f"{auc_val:.3f}", "Tiêu chuẩn": ">= 0.80 (Rất tốt)"})
            
        # Confusion Matrix
        cm = confusion_matrix(y, y_pred)
        df_cm = pd.DataFrame(
            cm,
            index=[f"Thực tế: {c}" for c in classes],
            columns=[f"Dự đoán: {c}" for c in classes]
        )
    else:
        # Regression metrics
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        k_preds = n_features
        adj_r2 = 1.0 - (1.0 - r2) * (n_samples - 1.0) / (n_samples - k_preds - 1.0) if (n_samples - k_preds - 1) > 0 else r2
        
        metrics_rows.append({"Chỉ số Đánh giá": "Hệ số xác định R-squared (R^2)", "Giá trị": f"{r2:.3f}", "Tiêu chuẩn": ">= 0.50 (Giải thích tốt)"})
        metrics_rows.append({"Chỉ số Đánh giá": "R-squared Hiệu chỉnh (Adjusted R^2)", "Giá trị": f"{adj_r2:.3f}", "Tiêu chuẩn": "Sát với R^2"})
        metrics_rows.append({"Chỉ số Đánh giá": "Căn sai số bình phương trung bình (RMSE)", "Giá trị": f"{rmse:.3f}", "Tiêu chuẩn": "Càng nhỏ càng tốt"})
        metrics_rows.append({"Chỉ số Đánh giá": "Sai số tuyệt đối trung bình (MAE)", "Giá trị": f"{mae:.3f}", "Tiêu chuẩn": "Càng nhỏ càng tốt"})
        df_cm = pd.DataFrame()
        roc_data = None

    df_metrics = pd.DataFrame(metrics_rows)

    # ---------------- FEATURE IMPORTANCE / COEFFICIENTS ----------------
    if hasattr(model, "coef_"):
        coef_vals = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
        c_rows = []
        for feat, val in zip(feature_cols, coef_vals):
            or_val = np.exp(val) if is_classification and is_binary else None
            r_dict = {
                "Biến số (Feature)": feat,
                "Hệ số hồi quy (Beta)": round(float(val), 4),
            }
            if or_val is not None:
                r_dict["Tỷ số chênh Odds Ratio (OR = exp(B))"] = round(float(or_val), 3)
                r_dict["Ý nghĩa tác động"] = f"Tăng {((or_val - 1)*100):+.1f}% xác suất" if or_val != 1 else "Không đổi"
            c_rows.append(r_dict)
        coef_table = pd.DataFrame(c_rows)

    if hasattr(model, "feature_importances_"):
        imp_vals = model.feature_importances_
        sorted_idx = np.argsort(imp_vals)[::-1]
        i_rows = []
        for rank, idx in enumerate(sorted_idx, 1):
            i_rows.append({
                "Xếp hạng": rank,
                "Biến số": feature_cols[idx],
                "Mức độ quan trọng (Feature Importance)": round(float(imp_vals[idx]), 4),
                "Tỷ lệ đóng góp (%)": f"{imp_vals[idx] * 100:.2f}%"
            })
        importance_table = pd.DataFrame(i_rows)

    return {
        "algorithm": algorithm,
        "is_classification": is_classification,
        "metrics_table": df_metrics,
        "confusion_matrix": df_cm,
        "roc_data": roc_data,
        "coefficients_table": coef_table,
        "importance_table": importance_table,
        "y_actual": y.tolist(),
        "y_predicted": y_pred.tolist(),
        "n_samples": n_samples
    }


# ==============================================================================
# 3. ADVANCED UNSUPERVISED LEARNING (PCA, HIERARCHICAL, LDA)
# ==============================================================================

def run_pca_reduction(df: pd.DataFrame, feature_cols: list, n_components: int = None) -> dict:
    """
    Performs Principal Component Analysis (PCA) for dimensionality reduction.
    Returns:
    - Explained variance ratio table
    - Component loadings (Eigenvectors scaled by sqrt(Eigenvalues))
    - 2D/3D Projection data for Biplot
    """
    clean_df = df[feature_cols].dropna().copy()
    X = clean_df.values
    n_samples, n_features = X.shape
    
    # Standardize
    X_scaled = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    
    max_k = min(n_samples, n_features)
    k = n_components if n_components and 0 < n_components <= max_k else min(max_k, 5)
    
    pca = PCA(n_components=k)
    pca_scores = pca.fit_transform(X_scaled)
    
    # Variance explained table
    evr = pca.explained_variance_ratio_ * 100.0
    ev = pca.explained_variance_
    cum_evr = np.cumsum(evr)
    
    rows_var = []
    for i in range(k):
        rows_var.append({
            "Thành phần chính": f"PC{i+1}",
            "Eigenvalue": round(float(ev[i]), 3),
            "% Phương sai giải thích": round(float(evr[i]), 2),
            "% Phương sai tích lũy": round(float(cum_evr[i]), 2)
        })
    df_var = pd.DataFrame(rows_var)
    
    # Component Loadings = components_ * sqrt(explained_variance_)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    pc_cols = [f"PC{i+1}" for i in range(k)]
    df_loadings = pd.DataFrame(loadings.round(3), index=feature_cols, columns=pc_cols)
    
    # 2D projection df for biplot
    df_proj = pd.DataFrame({
        "PC1": pca_scores[:, 0].round(3),
        "PC2": pca_scores[:, 1].round(3) if k >= 2 else np.zeros(n_samples)
    })
    
    return {
        "variance_table": df_var,
        "loadings_table": df_loadings,
        "scores_2d": df_proj,
        "n_components": k,
        "total_variance_explained": round(float(cum_evr[-1]), 2)
    }


def run_hierarchical_clustering_model(df: pd.DataFrame, feature_cols: list, n_clusters: int = 3, method: str = 'ward') -> dict:
    """
    Performs Agglomerative Hierarchical Clustering with Dendrogram Linkage matrix.
    """
    clean_df = df[feature_cols].dropna().copy()
    X = clean_df.values
    n_samples = len(X)
    
    X_scaled = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    
    # Compute Linkage matrix for Dendrogram
    Z = linkage(X_scaled, method=method)
    
    # Fit cluster model
    model = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage=method)
    labels = model.fit_predict(X_scaled)
    clean_df["Cluster"] = [f"Cụm {lbl + 1}" for lbl in labels]
    
    # Cluster sizes
    sizes = clean_df["Cluster"].value_counts().sort_index()
    size_rows = []
    for c_name, count in sizes.items():
        size_rows.append({
            "Phân cụm": c_name,
            "Số lượng (N)": count,
            "Tỷ lệ (%)": f"{(count / n_samples) * 100:.1f}%"
        })
    df_sizes = pd.DataFrame(size_rows)
    
    # Cluster Profiles (Mean on original scale)
    df_profile = clean_df.groupby("Cluster")[feature_cols].mean().round(3).reset_index()
    
    sil = silhouette_score(X_scaled, labels)
    
    return {
        "linkage_matrix": Z,
        "cluster_sizes": df_sizes,
        "cluster_profiles": df_profile,
        "silhouette_score": round(float(sil), 3),
        "labeled_df": clean_df
    }


def run_lda_topic_model(text_series: pd.Series, n_topics: int = 3, n_top_words: int = 6) -> dict:
    """
    Performs Latent Dirichlet Allocation (LDA) Topic Modeling on text comments / survey feedback.
    """
    clean_texts = text_series.dropna().astype(str).tolist()
    if len(clean_texts) < 5:
        return {"error": "Cần ít nhất 5 đoạn văn bản phản hồi để trích xuất chủ đề LDA."}
        
    # Count Vectorizer with simple word tokenization
    vectorizer = CountVectorizer(max_df=0.95, min_df=1, token_pattern=r'(?u)\b\w+\b')
    tf = vectorizer.fit_transform(clean_texts)
    feature_names = vectorizer.get_feature_names_out()
    
    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=20, random_state=42)
    lda.fit(tf)
    
    # Extract top words per topic
    topics_list = []
    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_indices]
        top_weights = [round(float(topic[i]), 1) for i in top_indices]
        
        topics_list.append({
            "Chủ đề (Topic)": f"Chủ đề #{topic_idx + 1}",
            "Từ khóa đặc trưng (Keywords)": ", ".join(top_words),
            "Từ khóa chính": top_words[0] if top_words else ""
        })
        
    df_topics = pd.DataFrame(topics_list)
    
    # Document - Topic distribution
    doc_topics = lda.transform(tf)
    dominant_topic = [f"Chủ đề #{int(np.argmax(dt)) + 1}" for dt in doc_topics]
    
    df_doc_summary = pd.DataFrame({
        "Ý kiến đáp viên": clean_texts[:30],
        "Chủ đề xác định": dominant_topic[:30]
    })
    
    # Topic proportions
    topic_counts = pd.Series(dominant_topic).value_counts().reset_index()
    topic_counts.columns = ["Chủ đề", "Số lượng ý kiến"]
    topic_counts["Tỷ lệ (%)"] = (topic_counts["Số lượng ý kiến"] / len(clean_texts) * 100).round(1).astype(str) + "%"

    return {
        "topics_table": df_topics,
        "topic_counts": topic_counts,
        "sample_assignments": df_doc_summary,
        "perplexity": round(float(lda.perplexity(tf)), 2)
    }


# ==============================================================================
# 4. NLP & SENTIMENT ANALYSIS ENGINE (TIẾNG VIỆT & ENGLISH)
# ==============================================================================

# Curated high-accuracy sentiment lexicon for Social Science Research
VI_POS_WORDS = {
    'tốt', 'rất tốt', 'tuyệt vời', 'xuất sắc', 'hài lòng', 'an tâm', 'tin tưởng', 
    'nhanh', 'tiện lợi', 'dễ dùng', 'thích', 'ủng hộ', 'ổn định', 'hiệu quả',
    'chính xác', 'bảo mật', 'nhiệt tình', 'chu đáo', 'tích cực', 'thông minh',
    'hiện đại', 'chuyên nghiệp', 'hợp lý', 'đạt yêu cầu', 'tuyệt', 'ưng ý'
}

VI_NEG_WORDS = {
    'kém', 'tệ', 'rất tệ', 'thất vọng', 'chậm', 'giật', 'lag', 'lỗi', 'khó dùng',
    'phức tạp', 'không thích', 'bất tiện', 'đắt', 'không an tâm', 'lo ngại',
    'bực mình', 'bảo mật kém', 'quá tải', 'tồi tệ', 'rắc rối', 'không hài lòng'
}

VI_NEGATIONS = {'không', 'chẳng', 'chưa', 'kém', 'ít'}


def analyze_text_sentiment(text: str) -> dict:
    """Calculates sentiment polarity score in [-1.0, +1.0] for Vietnamese and English text."""
    if not isinstance(text, str) or not text.strip():
        return {"score": 0.0, "label": "Trung lập (Neutral)"}
        
    text_lower = text.lower()
    words = re.findall(r'\w+', text_lower)
    
    pos_score = 0
    neg_score = 0
    
    for i, word in enumerate(words):
        is_negated = (i > 0 and words[i-1] in VI_NEGATIONS)
        
        # Check positive
        if word in VI_POS_WORDS or any(pw in text_lower for pw in ['tuyệt vời', 'rất tốt', 'hài lòng']):
            if is_negated:
                neg_score += 1.5
            else:
                pos_score += 1.0
                
        # Check negative
        if word in VI_NEG_WORDS or any(nw in text_lower for nw in ['rất tệ', 'thất vọng', 'không hài lòng']):
            if is_negated:
                pos_score += 1.0
            else:
                neg_score += 1.5
                
    total = pos_score + neg_score
    if total == 0:
        score = 0.0
    else:
        score = (pos_score - neg_score) / (pos_score + neg_score)
        
    if score >= 0.2:
        label = "Tích cực (Positive)"
    elif score <= -0.2:
        label = "Tiêu cực (Negative)"
    else:
        label = "Trung lập (Neutral)"
        
    return {"score": round(score, 3), "label": label}


def run_sentiment_analysis_suite(text_series: pd.Series) -> dict:
    """
    Analyzes sentiment for a series of open-ended survey comments.
    Returns:
    - Summary counts & percentages (Positive, Neutral, Negative)
    - Average Polarity score
    - Labeled Table with Individual Scores
    """
    clean_series = text_series.dropna().astype(str)
    if len(clean_series) == 0:
        return {"error": "Cột văn bản không có dữ liệu."}
        
    results = [analyze_text_sentiment(t) for t in clean_series]
    scores = [r["score"] for r in results]
    labels = [r["label"] for r in results]
    
    df_evaluated = pd.DataFrame({
        "Ý kiến phản hồi của đáp viên": clean_series.values,
        "Điểm cảm xúc (Polarity [-1; +1])": scores,
        "Phân loại cảm xúc": labels
    })
    
    # Counts
    counts = pd.Series(labels).value_counts()
    total = len(labels)
    summary_rows = []
    for lbl in ["Tích cực (Positive)", "Trung lập (Neutral)", "Tiêu cực (Negative)"]:
        cnt = counts.get(lbl, 0)
        pct = (cnt / total) * 100.0 if total > 0 else 0.0
        summary_rows.append({
            "Thái độ / Cảm xúc": lbl,
            "Số lượng (N)": cnt,
            "Tỷ lệ (%)": f"{pct:.1f}%"
        })
    df_summary = pd.DataFrame(summary_rows)
    avg_polarity = float(np.mean(scores))
    
    return {
        "summary_table": df_summary,
        "detailed_table": df_evaluated,
        "average_polarity": round(avg_polarity, 3),
        "overall_sentiment": "Tích cực" if avg_polarity >= 0.1 else ("Tiêu cực" if avg_polarity <= -0.1 else "Trung lập")
    }
