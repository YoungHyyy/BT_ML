# Mô tả Project SVM Prediction API

## 1. Tổng quan

Project này xây dựng một hệ thống Machine Learning dùng mô hình **Support Vector Machine (SVM)** để phân loại dữ liệu thành hai nhóm: `0` hoặc `1`.

Mô hình được đóng gói thành REST API bằng FastAPI. Client gửi vào đúng 4 giá trị số, API thực hiện tiền xử lý, đưa dữ liệu qua model SVM và trả về nhãn dự đoán, xác suất dự đoán cùng điểm `decision_function`.

Project cũng hỗ trợ:

- Huấn luyện và lưu model bằng `joblib`.
- Kiểm tra dữ liệu đầu vào bằng Pydantic và NumPy.
- Kiểm tra trạng thái API và model qua health check.
- Chạy bằng Docker và Docker Compose.
- Kiểm thử tự động bằng Pytest.
- Kiểm thử endpoint bằng PowerShell.

## 2. Project đang giải quyết vấn đề gì?

Project minh họa cách biến một mô hình Machine Learning thành một dịch vụ có thể được ứng dụng khác gọi qua HTTP.

Thay vì chỉ chạy model trực tiếp trong một file Python, project cung cấp endpoint:

```text
POST /api/v1/predict
```

Ứng dụng khác có thể gửi dữ liệu JSON:

```json
{
  "features": [0.2, -1.1, 0.5, 1.3]
}
```

API sẽ trả về kết quả phân loại, ví dụ:

```json
{
  "success": true,
  "status": 200,
  "message": "Dự đoán Support Vector Machine (SVM) thành công",
  "data": {
    "model": "support_vector_machine",
    "endpoint": "/api/v1/predict",
    "prediction": "1",
    "probability": 0.91,
    "decision_function": 1.8,
    "health_status": "healthy"
  }
}
```

Có thể hiểu đơn giản:

- `prediction: "1"`: dữ liệu được phân loại vào nhóm 1.
- `probability: 0.91`: model ước tính xác suất của class được chọn là 91%.
- `decision_function: 1.8`: điểm quyết định của SVM, cho biết vị trí của dữ liệu so với siêu phẳng phân tách.

## 3. Kiến trúc tổng thể

```text
                    +--------------------+
                    |  Client/Application |
                    +----------+---------+
                               |
                               | HTTP JSON
                               v
                    +--------------------+
                    |    FastAPI Server   |
                    |  /health            |
                    |  /api/v1/predict   |
                    +----------+---------+
                               |
                               v
                    +--------------------+
                    | Input Validation   |
                    | Pydantic + NumPy   |
                    +----------+---------+
                               |
                               v
                    +--------------------+
                    | Preprocessing      |
                    | StandardScaler     |
                    +----------+---------+
                               |
                               v
                    +--------------------+
                    | SVM Pipeline       |
                    | RBF SVC             |
                    +----------+---------+
                               |
                               v
                    +--------------------+
                    | Prediction Response|
                    | Label + Probability|
                    | + Decision Function|
                    +--------------------+
```

Luồng chính của hệ thống:

```text
Tạo dữ liệu
    -> Huấn luyện model
    -> Lưu model
    -> Khởi động API
    -> Load model
    -> Nhận 4 feature
    -> Validate và reshape dữ liệu
    -> Chuẩn hóa dữ liệu
    -> SVM dự đoán
    -> Trả kết quả JSON
```

## 4. Cấu trúc thư mục

```text
.
├── app/
│   ├── main.py
│   ├── model.py
│   ├── preprocessing.py
│   └── schemas.py
├── models/
│   └── svm.joblib
├── scripts/
│   └── test_endpoint.ps1
├── tests/
│   └── test_api.py
├── training/
│   └── train.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── Ml.md
└── project.md
```

## 5. Giải thích từng thành phần

### 5.1. `training/train.py`

Đây là chương trình huấn luyện model.

File này hiện dùng `make_classification` của scikit-learn để tạo dữ liệu giả:

- 600 mẫu dữ liệu.
- 4 feature cho mỗi mẫu.
- 2 class phân loại.
- 3 feature có thông tin hữu ích.
- `random_state=42` để kết quả có thể tái lập.

Dữ liệu được chia thành tập train và test bằng `train_test_split`. Hiện tại chỉ sử dụng tập train để huấn luyện model.

Pipeline huấn luyện gồm hai bước:

```text
StandardScaler -> SVC(kernel="rbf", probability=True)
```

Ý nghĩa:

- `StandardScaler`: đưa các feature về thang đo phù hợp.
- `SVC`: mô hình Support Vector Classifier.
- `kernel="rbf"`: cho phép xử lý ranh giới phân loại phi tuyến.
- `probability=True`: bật khả năng trả về xác suất.
- `random_state=42`: giúp quá trình huấn luyện ổn định hơn.

Sau khi train, pipeline được lưu tại:

```text
models/svm.joblib
```

### 5.2. `models/svm.joblib`

Đây là file chứa pipeline SVM đã được huấn luyện.

Pipeline bao gồm cả scaler và model, vì vậy lúc dự đoán dữ liệu được xử lý giống với lúc huấn luyện. Điều này tránh tình trạng train và prediction sử dụng hai cách chuẩn hóa khác nhau.

### 5.3. `app/preprocessing.py`

File này kiểm tra và định dạng dữ liệu trước khi đưa vào model.

Các điều kiện kiểm tra:

- Request phải có đúng 4 feature.
- Các feature phải là số.
- Không chấp nhận `NaN`, `inf` hoặc `-inf`.
- Dữ liệu được chuyển sang `numpy.float64`.
- Dữ liệu được reshape thành dạng `(1, 4)`.

Dạng `(1, 4)` nghĩa là:

- Có 1 mẫu dữ liệu.
- Mỗi mẫu có 4 feature.

Nếu dữ liệu sai, hàm ném `ValueError` để API trả HTTP `422`.

### 5.4. `app/model.py`

File này đóng gói logic quản lý model.

Lớp `SVMModel` thực hiện các nhiệm vụ:

- Xác định đường dẫn đến file model.
- Load model bằng `joblib.load`.
- Kiểm tra model đã được load hay chưa.
- Tiền xử lý feature.
- Gọi `predict` để lấy nhãn.
- Gọi `predict_proba` để lấy xác suất.
- Gọi `decision_function` để lấy điểm quyết định.

Kết quả của hàm `predict` là:

```python
prediction, probability, decision
```

Trong đó:

- `prediction`: nhãn dự đoán, được chuyển thành chuỗi như `"0"` hoặc `"1"`.
- `probability`: xác suất của nhãn đã dự đoán.
- `decision`: giá trị decision function của SVM.

Nếu model chưa được load, hàm ném `RuntimeError` và API trả HTTP `503`.

### 5.5. `app/schemas.py`

File này định nghĩa cấu trúc request và response bằng Pydantic.

`PredictionRequest` yêu cầu trường `features` là một danh sách có đúng 4 phần tử.

Các schema response gồm:

- `PredictionData`: dữ liệu chi tiết của kết quả dự đoán.
- `PredictionResponse`: response đầy đủ của endpoint prediction.
- `HealthResponse`: response của endpoint health check.

Pydantic giúp FastAPI tự động validate request và tạo tài liệu OpenAPI tại `/docs`.

### 5.6. `app/main.py`

Đây là entry point của FastAPI.

Khi ứng dụng khởi động, lifecycle `lifespan` gọi `svm_model.load()` để load model trước khi nhận request.

Các endpoint hiện có:

#### `GET /health`

Dùng để kiểm tra API và model:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model": "support_vector_machine"
}
```

#### `POST /api/v1/predict`

Nhận request gồm 4 feature và trả kết quả dự đoán.

Các lỗi được xử lý:

- Input không hợp lệ: HTTP `422`.
- Model chưa load: HTTP `503`.
- Dự đoán thành công: HTTP `200`.

### 5.7. `tests/test_api.py`

File này chứa kiểm thử tự động bằng Pytest.

Các nội dung đã kiểm tra:

- `/health` trả HTTP `200`.
- Model được load thành công.
- Endpoint prediction trả HTTP `200`.
- Nhãn dự đoán là `"0"` hoặc `"1"`.
- Xác suất nằm trong khoảng từ `0` đến `1`.
- `decision_function` là số thực.
- Request có sai số lượng feature bị trả HTTP `422`.

Kết quả kiểm thử hiện tại:

```text
2 passed
```

### 5.8. `scripts/test_endpoint.ps1`

Đây là script kiểm thử thủ công bằng PowerShell.

Script thực hiện:

1. Gọi `GET /health`.
2. Dừng và báo lỗi nếu API không healthy.
3. Gửi request đến `/api/v1/predict`.
4. In response JSON ra terminal.

### 5.9. `Dockerfile`

Dockerfile thực hiện các bước:

1. Dùng image `python:3.12-slim`.
2. Cài dependency từ `requirements.txt`.
3. Copy thư mục `app` và `training` vào container.
4. Chạy `python training/train.py` khi build image.
5. Mở port `3000`.
6. Khởi động Uvicorn.

Việc train model trong lúc build giúp container tự tạo file model trong môi trường Docker.

### 5.10. `docker-compose.yml`

Docker Compose định nghĩa service `svm-api`.

Cấu hình chính:

- Map port host `3000` vào port container `3000`.
- Chạy health check đến `http://127.0.0.1:3000/health`.
- Kiểm tra mỗi 10 giây.
- Timeout mỗi lần kiểm tra là 5 giây.
- Cho phép 5 lần retry.

### 5.11. `requirements.txt`

Các thư viện chính:

- `fastapi`: xây dựng REST API.
- `uvicorn`: chạy ASGI server.
- `scikit-learn`: huấn luyện và dự đoán bằng SVM.
- `numpy`: xử lý dữ liệu dạng mảng.
- `joblib`: lưu và load model.
- `pytest`: kiểm thử tự động.
- `httpx`: hỗ trợ FastAPI TestClient.

## 6. API usage

### Chạy local

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python training/train.py
uvicorn app.main:app --reload --port 3000
```

Sau đó mở tài liệu API:

```text
http://localhost:3000/docs
```

### Gọi endpoint prediction

Request:

```powershell
$body = @{ features = @(0.2, -1.1, 0.5, 1.3) } | ConvertTo-Json
Invoke-RestMethod -Method Post `
  -Uri "http://localhost:3000/api/v1/predict" `
  -ContentType "application/json" `
  -Body $body
```

### Chạy bằng Docker

```powershell
docker compose up --build
```

### Chạy script kiểm thử endpoint

```powershell
.\scripts\test_endpoint.ps1
```

### Chạy test

Trong một số môi trường, cần thêm thư mục gốc project vào `PYTHONPATH`:

```powershell
$env:PYTHONPATH = "D:\BT ML"
pytest -q
```

## 7. Những phần đã hoàn thành

Project hiện đã hoàn thành các chức năng chính sau:

- Tạo dữ liệu huấn luyện.
- Huấn luyện mô hình SVM.
- Chuẩn hóa feature bằng `StandardScaler`.
- Lưu pipeline bằng `joblib`.
- Load model khi API khởi động.
- Validate input bằng Pydantic và NumPy.
- Nhận dữ liệu qua REST API.
- Trả nhãn dự đoán.
- Trả xác suất dự đoán.
- Trả `decision_function`.
- Có endpoint `/health`.
- Có tài liệu OpenAPI tại `/docs`.
- Có Dockerfile.
- Có Docker Compose.
- Có Docker health check.
- Có test tự động.
- Có script kiểm thử PowerShell.

## 8. Những giới hạn hiện tại

### 8.1. Dữ liệu đang là dữ liệu giả

Model hiện được huấn luyện từ `make_classification`, vì vậy 4 feature chưa đại diện cho dữ liệu nghiệp vụ thực tế.

Ví dụ các feature chưa thực sự có nghĩa là tuổi, thu nhập hoặc điểm tín dụng. Project hiện chủ yếu minh họa quy trình triển khai ML API.

### 8.2. Chưa đánh giá chất lượng model

Code hiện chưa tính các chỉ số như:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- Confusion matrix.
- Cross-validation.

Tập test được tạo khi chia dữ liệu nhưng hiện chưa được dùng để báo cáo chất lượng model.

### 8.3. Chưa có authentication

API hiện chưa có:

- API key.
- JWT.
- Đăng nhập người dùng.
- Rate limiting.

Điều này phù hợp với demo local nhưng cần bổ sung nếu triển khai production thực tế.

### 8.4. Tài liệu cũ có ví dụ chưa khớp code

Một số ví dụ trong `Ml.md` cần cập nhật:

- API thực tế yêu cầu đúng 4 feature, không phải 3 feature.
- Prediction thực tế trả `"0"` hoặc `"1"`, không phải `"class_1"`.

## 9. Kết luận

Project đã xây dựng được một pipeline triển khai Machine Learning cơ bản nhưng đầy đủ:

```text
Training
  -> Model artifact
  -> Preprocessing
  -> FastAPI endpoint
  -> Prediction response
  -> Health check
  -> Automated tests
  -> Docker deployment
```

Nói ngắn gọn, đây là một hệ thống biến mô hình SVM thành REST API để ứng dụng khác có thể gửi 4 giá trị đầu vào và nhận kết quả phân loại tự động.

Để phát triển thành hệ thống thực tế hơn, các bước tiếp theo nên là:

1. Thay dữ liệu giả bằng dataset thực tế.
2. Ghi nhận ý nghĩa của từng feature.
3. Đánh giá model bằng tập test.
4. Thêm logging và monitoring.
5. Thêm authentication và giới hạn request.
6. Cập nhật tài liệu cho khớp hoàn toàn với API hiện tại.
