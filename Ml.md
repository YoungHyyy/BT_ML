### Hướng dẫn tích hợp Máy chủ Cục bộ và Chiến lược Triển khai AI

#### Chủ đề: Thiết lập, Kiểm thử và Triển khai mô hình Support Vector Machine (SVM)

#### 1. Mục tiêu

- Vận hành đồng thời API Service Server và máy chủ AI sử dụng mô hình Support Vector Machine (SVM) ổn định trên máy tính cục bộ.
- Làm chủ hoàn toàn mã nguồn, đặc biệt là quy trình huấn luyện, tìm siêu phẳng phân tách và dự đoán của mô hình Support Vector Machine (SVM).
- Thực hiện ảo hóa bằng Docker để đồng bộ môi trường chạy mô hình giữa môi trường phát triển cục bộ và Production.
- Đóng gói, kiểm thử Endpoint dự đoán, kiểm tra trạng thái hệ thống và triển khai mô hình Support Vector Machine (SVM) ổn định trên Production.

---

### 2. Công nghệ bắt buộc

- Docker
- Support Vector Machine (SVM) Model (Mô hình phân loại Support Vector Machine (SVM))
- Model Training & Prediction (Huấn luyện và dự đoán)
- API Service Server (Máy chủ dịch vụ tích hợp)
- RESTful Endpoint
- Health Check Monitoring (Kiểm tra trạng thái hệ thống)
- Command Line Testing Tools (Kiểm thử qua dòng lệnh)

---

### 3. Kiến trúc

Áp dụng kiến trúc đa cấu phần (Multi-component Architecture), trong đó API Server tiếp nhận dữ liệu đầu vào, chuyển dữ liệu cho tầng xử lý Support Vector Machine (SVM) và trả về lớp dự đoán cùng xác suất tương ứng.

```text
Hệ thống tích hợp (Local Integration Suite)

Cục bộ (Local machine) / Container:
 ├── Component 1 (API Endpoint)
 ├── Component 2 (Tiền xử lý dữ liệu)
 ├── Support Vector Machine (SVM) Model (Tìm siêu phẳng phân tách & Tính hàm quyết định - Decision Function)
 └── Docker Environment (Môi trường ảo hóa)
```

Tách riêng API, tiền xử lý và mô hình để có thể kiểm thử từng thành phần độc lập.

---

### 4. Luồng hoạt động chi tiết

#### 4.1 Thiết kế cấu phần (Component Design)

- Dựng API Endpoint đầu tiên để nhận dữ liệu cần phân loại và chạy thử nghiệm độc lập.
- Phân tách riêng bước tiền xử lý dữ liệu, tính toán Support Vector Machine (SVM) và trả kết quả dự đoán để dễ kiểm thử.
- Khởi tạo Endpoint và xác định đường dẫn để API giao tiếp với tầng xử lý mô hình.

---

#### 4.2 Kiểm thử tham số đầu vào (Parameter Testing)

- Hỗ trợ truyền dữ liệu đầu vào dưới dạng JSON, file hoặc chuỗi ký tự tùy theo đặc tả Endpoint.
- Kiểm tra dữ liệu đầu vào sau tiền xử lý trước khi đưa vào mô hình Support Vector Machine (SVM).
- Kiểm tra mã trạng thái phản hồi của Endpoint, đồng thời đối chiếu lớp dự đoán và kết quả phân loại với kết quả mong đợi.
- Thực thi lệnh kiểm thử trực tiếp từ dòng lệnh để xác thực Endpoint và mô hình.

---

#### 4.3 Tích hợp máy chủ AI cục bộ (Local AI Integration)

- Huấn luyện mô hình Support Vector Machine (SVM) từ tập dữ liệu đã chuẩn hóa và lưu các tham số cần thiết cho bước dự đoán.
- Khi có request, API Server thực hiện tiền xử lý dữ liệu rồi truyền dữ liệu vào mô hình Support Vector Machine (SVM).
- Mô hình SVM xác định lớp dựa trên siêu phẳng phân tách và hàm quyết định (decision function).
- Vận hành API Server và mô hình Support Vector Machine (SVM) trên môi trường cục bộ để kiểm chứng toàn bộ luồng từ input đến prediction.

---

#### 4.4 Đóng gói ảo hóa và Triển khai Production (Dockerization & Production)

- Sử dụng Docker để đóng gói API Server, mã nguồn mô hình và các dependency cần thiết.
- Thiết lập Health Check để kiểm tra API và trạng thái mô hình Support Vector Machine (SVM) đã sẵn sàng phục vụ dự đoán.
- Thực hiện đóng gói sản phẩm hoàn chỉnh và triển khai lên Production sau khi kiểm thử thành công.

---

### 5. API Response mẫu

Phản hồi Endpoint dự đoán Support Vector Machine (SVM) thành công:

```json
{
  "success": true,
  "status": 200,
  "message": "Dự đoán Support Vector Machine (SVM) thành công",
  "data": {
    "model": "support_vector_machine",
    "endpoint": "/api/v1/predict",
    "prediction": "class_1",
    "probability": 0.87,
    "health_status": "healthy"
  }
}
```

> **Lưu ý:** Với mô hình SVM chuẩn, đầu ra gốc chỉ là nhãn lớp dự đoán và khoảng cách đến siêu phẳng (decision function), không mặc định có xác suất (`probability`). Để trả về trường `probability` như trên, cần bật tùy chọn ước lượng xác suất bằng kỹ thuật Platt Scaling (ví dụ tham số `probability=True` khi dùng scikit-learn `SVC`), đồng thời lưu ý chi phí tính toán tăng lên khi bật tùy chọn này.

---

### 6. Thao tác Docker cơ bản

Tệp cấu hình chạy thử nghiệm cục bộ nhanh:

```bash
# Khởi chạy API Server và mô hình Support Vector Machine (SVM) bằng Docker
docker compose up --build

# Kiểm thử Endpoint dự đoán
curl -X POST http://localhost:3000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"features":["feature_1","feature_2","feature_3"]}'
```

---

### 7. Tiêu chí đánh giá

## Hạng mục Điểm

Kiến trúc tách cấu phần (Architecture) 20
Tiền xử lý & Kiểm thử dữ liệu đầu vào 20
Huấn luyện & Dự đoán bằng Support Vector Machine (SVM) 25
Đóng gói Container & Cấu hình Docker 15
Health Check & Triển khai Production 10
Tư duy tự chủ mã nguồn (Code Ownership) 10

---

### 8. Yêu cầu nộp bài

- Mã nguồn API và mô hình Support Vector Machine (SVM) hoàn chỉnh.
- Tệp cấu hình Dockerfile và docker-compose.yml.
- Tài liệu mô tả dữ liệu đầu vào, quy trình huấn luyện, dự đoán và Endpoint.
- Kịch bản kiểm thử Endpoint bằng dòng lệnh.
- Video/Hình ảnh minh chứng hệ thống chạy mô hình Support Vector Machine (SVM) thành công.
