# Local SVM Integration Suite

Service FastAPI phân loại nhị phân bằng Support Vector Machine, có tiền xử lý, huấn luyện, xác suất dự đoán và Docker health check.

## Chạy cục bộ

```powershell
python -m venv .venv

.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

python training/train.py

uvicorn app.main:app --reload --port 3000
```

## Chạy bằng Docker

```bash
docker compose up --build
```

Model được huấn luyện trong lúc build image và lưu tại `models/svm.joblib`. Endpoint chỉ nhận đúng bốn đặc trưng số theo thứ tự đã dùng khi huấn luyện.

## Endpoint

- `GET /health`: trạng thái API và model.
- `POST /api/v1/predict`: body JSON `{"features": [0.2, -1.1, 0.5, 1.3]}`.
- `GET /docs`: tài liệu OpenAPI.

Ví dụ kiểm thử bằng PowerShell:

```powershell
.\scripts\test_endpoint.ps1
```

Response prediction gồm nhãn `0` hoặc `1`, xác suất của nhãn được chọn và khoảng cách `decision_function` đến siêu phẳng phân tách. `StandardScaler` và `SVC(probability=True)` được đóng gói chung trong pipeline để tiền xử lý lúc dự đoán luôn giống lúc huấn luyện.

## Kiểm thử

```powershell
pytest -q
```
