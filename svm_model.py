
# ============================================================
# MODEL SUPPORT VECTOR MACHINE (SVM)
# Bài toán: Phân loại hoa Iris
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names

print("===== THÔNG TIN DATASET =====")
print(f"Số lượng mẫu: {X.shape[0]}")
print(f"Số lượng feature: {X.shape[1]}")
print(f"Các class: {list(target_names)}")

print("\nCác feature:")
for feature in feature_names:
    print("-", feature)


# Chuyển dữ liệu thành DataFrame để dễ xem
df = pd.DataFrame(X, columns=feature_names)
df["target"] = y

print("\n===== 5 DÒNG DỮ LIỆU ĐẦU TIÊN =====")
print(df.head())


# ============================================================
# 2. CHIA DATASET
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n===== CHIA DATASET =====")
print(f"Số mẫu train: {len(X_train)}")
print(f"Số mẫu test : {len(X_test)}")


# ============================================================
# 3. CHUẨN HÓA DỮ LIỆU
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 4. XÂY DỰNG MODEL SVM
# ============================================================

model = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale"
)

print("\n===== THÔNG SỐ MODEL =====")
print("Algorithm : Support Vector Machine")
print("Kernel    :", model.kernel)
print("C         :", model.C)
print("Gamma     :", model.gamma)


# ============================================================
# 5. TRAIN MODEL
# ============================================================

model.fit(X_train_scaled, y_train)

print("\nTrain model thành công!")


# ============================================================
# 6. DỰ ĐOÁN
# ============================================================

y_pred = model.predict(X_test_scaled)


# ============================================================
# 7. ĐÁNH GIÁ MODEL
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n===== KẾT QUẢ =====")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\n===== CLASSIFICATION REPORT =====")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=target_names
    )
)


# ============================================================
# 8. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\n===== CONFUSION MATRIX =====")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=target_names
)

disp.plot()
plt.title("Confusion Matrix - SVM")
plt.tight_layout()
plt.show()


# ============================================================
# 9. DỰ ĐOÁN MỘT BÔNG HOA MỚI
# ============================================================

# Dữ liệu:
# Sepal Length = 5.1
# Sepal Width  = 3.5
# Petal Length = 1.4
# Petal Width  = 0.2

new_flower = np.array([
    [5.1, 3.5, 1.4, 0.2]
])

new_flower_scaled = scaler.transform(new_flower)

prediction = model.predict(new_flower_scaled)

predicted_class = target_names[prediction[0]]

print("\n===== DỰ ĐOÁN HOA MỚI =====")
print("Dữ liệu:", new_flower[0])
print("Kết quả:", predicted_class)


# ============================================================
# 10. XÁC SUẤT DỰ ĐOÁN
# ============================================================

# probability=True cần được bật khi tạo model.
# Vì model phía trên chưa bật nên phần này không sử dụng.

print("\n===== HOÀN TẤT =====")
print("Model SVM đã được huấn luyện và sử dụng để dự đoán.")

