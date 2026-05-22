# 📊 Phân tích Chi phí và Hiệu năng AI Agent (05/2026 - 08/2026)

## 🎯 Tổng quan Dự án (Project Overview)

Dự án tập trung xây dựng hệ thống báo cáo Business Intelligence (BI) nhằm phân tích, đánh giá, và tối ưu hóa chi phí (Cost) cũng như hiệu năng (Performance/Latency) của các hệ thống AI Agent dựa trên dữ liệu Telemetry thu thập trong giai đoạn tháng 05/2026 đến tháng 08/2026.

**Mục tiêu cốt lõi:**

1. Định lượng chi phí vận hành (Token usage API costs) cho từng phiên giao dịch của Agent.
2. Đo lường độ trễ (Latency/Pre-gap) khi Agent gọi các công cụ (Tool-calling).
3. Cung cấp Dashboard trực quan để hỗ trợ quyết định quản trị (AIOps).

---

## 🛠 Kiến trúc & Công cụ (Tech Stack)

- **Tiền xử lý & ETL:** Python (`Pandas`, `Polars` hoặc các thư viện xử lý dữ liệu tương đương để thao tác và làm sạch dữ liệu JSON nested/Parquet).
- **Mô hình hóa dữ liệu (Data Modeling):** Xây dựng Fact & Dimension tables (Star Schema).
- **Phân tích nâng cao (Advanced Analytics):** Sử dụng các thuật toán và thư viện Machine Learning/Time Series linh hoạt (như `Statsmodels`, `Prophet`, `XGBoost`, hoặc mạng nơ-ron nếu cần) để dự báo chi phí hoặc phân cụm hành vi.
- **Trực quan hóa (BI Tools):** Power BI (Sử dụng DAX/Power Query để thiết lập các dynamic dashboard).
- **Quản lý mã nguồn:** Git/GitHub (Toàn bộ logic được viết bằng Python scripts `.py` và quản lý phiên bản chặt chẽ).

---

## 📂 Cấu trúc Thư mục (Directory Structure)

```text
├── data/
│   ├── raw/                  # Dữ liệu Telemetry thô (JSON/Parquet gốc - KHÔNG PUSH LÊN GIT)
│   ├── processed/            # Dữ liệu đã qua làm sạch (CSV/Parquet - KHÔNG PUSH LÊN GIT)
│   └── database/             # Script SQL hoặc schema định nghĩa database
├── src/
│   ├── exploration/
│   │   └── eda_analysis.py         # Script sinh các báo cáo phân tích dữ liệu thô
│   ├── data_pipeline/
│   │   ├── etl_pipeline.py         # Script chạy ETL tự động (Extract, Transform, Load)
│   │   └── data_cleaning.py        # Module làm sạch và flatten JSON/arrays
│   ├── models/
│   │   ├── cost_forecasting.py     # Script huấn luyện và chạy thuật toán dự báo chi phí
│   │   └── anomaly_detection.py    # Script phát hiện bất thường trong độ trễ/chi phí
│   └── features/
│       └── token_calculator.py     # Module tính toán input/output tokens ra tỷ giá USD
├── dashboards/
│   └── ai_agent_telemetry.pbix     # File Power BI Report
├── .gitignore                # Chặn các file data lớn, môi trường ảo, cache
├── requirements.txt          # Danh sách thư viện Python
└── CLAUDE.md                 # Tài liệu hướng dẫn phát triển (File này)
```
