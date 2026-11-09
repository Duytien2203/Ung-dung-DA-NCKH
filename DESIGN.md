# DESIGN.md - SmartResearch Studio (StaNDTistics) Design System

**Phiên bản**: 2.0 (Academic Modernist)  
**Tone & Mood**: Chuyên nghiệp, Chuẩn mực Khoa học, Điềm tĩnh, Tập trung cao độ (Academic Blue Dominant)  
**Mục tiêu sử dụng (Mode)**: Operate (Phân tích dữ liệu & Tính toán ma trận) + Read (Báo cáo tường thuật APA 7th)

---

## 1. Triết lý Thiết kế (Design Tenets)

1. **Restraint over Decoration (Tiết chế & Chuẩn mực)**:
   Không sử dụng hiệu ứng màu mè hay các đặc điểm giao diện AI mặc định (không viền màu dày lệch cạnh `border-left`, không đổ bóng cứng, không gradient chữ). Mọi thành phần đều phục vụ việc đọc số liệu rõ ràng và chính xác.
2. **Deep Academic Blue Accent (Xanh học thuật chủ đạo)**:
   Sắc xanh hoàng gia (`#1d4ed8`, `#1e40af`) kết hợp cùng hệ màu trung tính Slate cao cấp (`#0f172a`, `#334155`, `#f8fafc`) tạo cảm giác tin cậy, sắc bén và đậm chất nghiên cứu định lượng.
3. **Accessibility First (Tiếp cận chuẩn mực)**:
   Toàn bộ văn bản đảm bảo tỷ lệ tương phản WCAG 2.1 AA $\ge 4.5:1$ (phần lớn đạt chuẩn AAA $\ge 7:1$). Nút bấm tương tác đáp ứng chuẩn WCAG 2.5.8 ($\ge 24\text{px}$). Có chỉ báo `:focus-visible` rõ ràng cho người dùng bàn phím.
4. **Fluid Responsiveness & Zero Overlap**:
   Thanh điều hướng 10 bước (Workflow Stepper) trang bị bộ điều khiển cuộn hai chiều (`‹` và `›`) tự động ẩn/hiện thông minh kèm mặt nạ fade mịn, triệt tiêu hoàn toàn hiện tượng che khuất nhãn bước.

---

## 2. Bảng Màu & Hệ Thống Tokens (Design Tokens)

### 2.1. Brand Spectrum (Sắc độ xanh học thuật)
- `--brand-primary`: `#1d4ed8` (Royal Blue 700 - Trọng tâm nhận diện, nút Primary, trạng thái Active)
- `--brand-primary-hover`: `#1e40af` (Deep Royal 800)
- `--brand-primary-active`: `#172554` (Navy 950)
- `--brand-accent`: `#2563eb` (Blue 600 - Hover highlight, focus ring)
- `--brand-light`: `#eff6ff` (Blue 50 - Nền thẻ chọn, tag thống kê)
- `--brand-subtle`: `#dbeafe` (Blue 100 - Selection background)
- `--brand-border`: `#bfdbfe` (Blue 200 - Viền thẻ phân tích, viền hộp APA)

### 2.2. Neutral Surfaces (Bề mặt & Nền)
- `--bg-page`: `#edf2f7` / `#f8fafc` (Nền xám đá phiến nhẹ, dịu mắt khi làm việc kéo dài)
- `--bg-surface`: `#ffffff` (Bề mặt thẻ báo cáo, bảng dữ liệu)
- `--bg-subtle`: `#f1f5f9` (Slate 100 - Nền hàng xen kẽ, ô input)
- `--bg-hover`: `#e2e8f0` (Slate 200 - Hover nhẹ trên bảng)

### 2.3. Typography & Contrast (Văn bản & Tương phản WCAG)
- `--text-primary`: `#0f172a` (Slate 900 - Tỷ lệ 14.2:1 trên nền trắng, cực kỳ sắc nét)
- `--text-secondary`: `#334155` (Slate 700 - Tỷ lệ 9.6:1, đạt chuẩn AAA cho văn bản nội dung)
- `--text-muted`: `#475569` (Slate 600 - Tỷ lệ 5.8:1, vượt chuẩn AA cho nhãn phụ)
- `--text-inverse`: `#ffffff` (Chữ trắng trên nền xanh đậm)

### 2.4. Semantic Status (Chỉ báo thống kê)
- **Thành công / Đạt chuẩn** (Cronbach's $\alpha \ge 0.7$, Sig $< 0.05$): `#10b981` (Emerald), nền `#ecfdf5`, viền `#a7f3d0`
- **Cảnh báo / Cần xem xét**: `#f59e0b` (Amber), nền `#fffbeb`, viền `#fde68a`
- **Vi phạm / Không đạt**: `#ef4444` (Rose Red), nền `#fef2f2`, viền `#fecaca`

---

## 3. Kiểu Chữ (Typography Scale)

| Cấp bậc | Cỡ chữ | Độ đậm | Line-height | Khoảng cách chữ | Mục đích sử dụng |
|---|---|---|---|---|---|
| **Display / Header** | 24px - 28px | 850 | 1.2 | -0.5px | Tiêu đề trang, chỉ số KPI quan trọng |
| **Section Header** | 18px - 20px | 800 | 1.3 | -0.3px | Tên bước phân tích, tiêu đề khối bảng |
| **Subsection Header** | 15px - 16px | 750 | 1.4 | -0.15px | Tên bảng kết quả, nhãn biểu đồ |
| **Body / Narrative** | 14px - 14.5px| 500 - 600| 1.75 - 1.8 | Normal | Báo cáo nhận định học thuật APA 7th |
| **Data / Numbers** | 13px - 13.5px| 650 | 1.3 | JetBrains Mono | Số liệu ma trận, ma trận xoay EFA, P-value |
| **Micro Labels** | 11px - 12px | 700 - 800 | 1.2 | +0.5px (Uppercase) | Nhãn cột, huy hiệu trạng thái (Badge) |

---

## 4. Cấu Trúc Thành Phần Giao Diện (Components Anatomy)

### 4.1. Workflow Stepper Navigation Bar
- **Bố cục**: Tỷ lệ `[8.6 : 1.4]` giữa thanh Stepper và nút Nhật ký thao tác.
- **Dải Pills**: Bo góc 8px, nền trong suốt, chữ Slate 700 (`#334155`).
- **Pill Đang Chọn**: Nền Gradient Blue (`#1d4ed8` $\to$ `#2563eb`), chữ trắng tinh, bóng mờ `rgba(29, 78, 216, 0.35)`.
- **Nút Cuộn Điều Hướng (`‹` & `›`)**:
  - Kích thước: $28 \times 28\text{px}$ (vượt chuẩn WCAG 2.5.8 $\ge 24\text{px}$).
  - Hiệu ứng: Translucent glass với backdrop blur 6px, tự động ẩn khi cuộn tới mép.
  - Mặt nạ: Mask gradient alpha mềm ở mép cuộn, không bao giờ đè lên nhãn chữ.

### 4.2. Khối Nhận Định Học Thuật (Academic Narrative Box)
- Chuẩn hóa báo cáo theo phong cách APA 7th Edition.
- Viền mảnh đồng đều 1px (`#bfdbfe`), nền chuyển nhẹ dịu mắt, khoảng cách lề thoáng đãng ($22 \times 28\text{px}$).
- Loại bỏ hoàn toàn đường viền dày một bên (`border-left: 5px`) chống hiện tượng slop.

### 4.3. Thẻ Chỉ Số Khoa Học (Scientific KPI Card)
- Bo góc 12px, viền đơn sắc Slate 200 (`#e2e8f0`).
- Hiệu ứng hover nhấc nhẹ 2px (`translateY(-2px)`) kèm viền xanh Blue 400 (`#60a5fa`).
- Tuyệt đối không dùng dải màu trang trí phía trên (`::before` 3px stripe).

---

## 5. Quy Chuẩn Kỹ Thuật (Engineering Standards)

- **Độ nhạy chuyển động (Motion Sensitivity)**: Toàn bộ animation và transition tuân thủ `@media (prefers-reduced-motion: reduce)` nhằm bảo vệ người dùng nhạy cảm tiền đình.
- **Tiêu điểm bàn phím (Focus Rings)**: Vòng bao tiêu điểm 2px đồng nhất với khoảng cách offset 2px (`outline: 2px solid var(--border-focus); outline-offset: 2px`).
- **Responsive Viewports**:
  - Desktop lớn: $\ge 1280\text{px}$ (hiển thị trọn vẹn 10 bước).
  - Tablet: $768\text{px} - 1024\text{px}$ (thu nhỏ lề, cuộn mượt mà).
  - Mobile: $\le 480\text{px}$ (chiều cao nút bấm đạt chuẩn cảm ứng $\ge 44\text{px}$).
