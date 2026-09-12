# SVM Support Diagnostic for Breast Tumor Classification

Service FastAPI hỗ trợ chẩn đoán sơ bộ khối u lành tính hoặc ác tính bằng mô hình Support Vector Machine.
Model được huấn luyện trên bộ dữ liệu `sklearn.datasets.load_breast_cancer`, nhưng chỉ sử dụng 4 feature đầu tiên phù hợp với mô hình demo của project: bán kính, độ nhám, chu vi và diện tích khối u.

> Lưu ý quan trọng: đây là mô hình hỗ trợ chẩn đoán sơ bộ, không thay thế chẩn đoán lâm sàng, chẩn đoán hình ảnh hoặc quyết định của bác sĩ chuyên khoa.

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

Model được huấn luyện trong lúc build image và lưu tại `models/svm.joblib`. Endpoint nhận đúng 4 đặc trưng số theo thứ tự đã dùng khi huấn luyện.

## Dữ liệu đầu vào

API yêu cầu JSON theo mẫu:

```json
{
  "features": [14.2, 18.4, 92.3, 650.5]
}
```

Trong đó:

- `features[0]`: bán kính khối u
- `features[1]`: độ nhám
- `features[2]`: chu vi
- `features[3]`: diện tích khối u

Nhãn dự đoán:

- `0`: khối u lành tính
- `1`: khối u ác tính

## Endpoint

- `GET /health`: trạng thái API và model.
- `POST /api/v1/predict`: body JSON với 4 value số.
- `GET /docs`: tài liệu OpenAPI.

Ví dụ kiểm thử bằng PowerShell:

```powershell
.\scripts\test_endpoint.ps1
```

Response prediction gồm nhãn `0` hoặc `1`, xác suất của nhãn được chọn và khoảng cách `decision_function` đến siêu phẳng phân tách. `StandardScaler` và `SVC(probability=True)` được đóng gói chung trong pipeline để tiền xử lý lúc dự đoán luôn giống lúc huấn luyện.

Mỗi kết quả chỉ nên được xem như tín hiệu hỗ trợ chẩn đoán sơ bộ về khối u lành tính/ác tính, không phải chẩn đoán chính thức.

## Kiểm thử

```powershell
pytest -q
```
