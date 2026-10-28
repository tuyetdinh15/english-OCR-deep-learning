# Optical Character Recognition bằng Deep Learning

## Giới thiệu

Đồ án xây dựng hệ thống nhận dạng ký tự quang học (Optical Character Recognition - OCR) cho văn bản tiếng Anh. Hệ thống kết hợp các kỹ thuật xử lý ảnh truyền thống với mạng nơ-ron tích chập và các mô hình học chuyển giao để nhận dạng ký tự trong ảnh.

Mục tiêu của đồ án là nhận dạng các chữ cái tiếng Anh viết thường từ ảnh, đồng thời xây dựng pipeline có thể mở rộng cho các bài toán nhận dạng văn bản khác.

## Dữ liệu

Đồ án sử dụng bộ dữ liệu OCR-Dataset gồm ảnh các ký tự chữ cái, chữ số, chữ hoa và chữ thường được tạo từ nhiều kiểu font khác nhau. Trong phạm vi đồ án, hệ thống tập trung vào 26 lớp chữ cái tiếng Anh viết thường.

Dữ liệu và mã tiền xử lý được tổ chức chung trong một thư mục `dataset/`:

```text
dataset/
├── __init__.py
├── raw/                       # Ảnh gốc, mỗi thư mục con là một lớp
├── data_preprocessing/        # Mã Python tiền xử lý
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── image_preprocessing.py
│   └── prepare_data.py
└── processed/                 # CSV, ánh xạ nhãn và thống kê đã xử lý
```

## Xử lý dữ liệu ảnh

Quá trình chuẩn bị dữ liệu gồm các bước chính:

- Duyệt cấu trúc thư mục và tạo bảng dữ liệu gồm đường dẫn ảnh và nhãn ký tự.
- Lọc các lớp ký tự viết thường theo phạm vi của bài toán.
- Chia dữ liệu theo tỷ lệ **train 80% / test 10% / validation 10%**, phân tầng theo nhãn với `random_state=41`.
- Tạo image generator để đọc ảnh theo batch và thực hiện tăng cường độ sáng cho dữ liệu huấn luyện.
- Đưa ảnh về kích thước chuẩn phục vụ đầu vào mô hình.

### Chạy tiền xử lý dữ liệu

Mã nguồn nằm trong `dataset/data_preprocessing/` và lệnh chạy nằm trong
`dataset/data_preprocessing/prepare_data.py`. Cài các thư viện trong `pyproject.toml` vào môi trường Python
trước khi chạy. Lệnh xuất CSV chỉ cần pandas, scikit-learn và Pillow;
TensorFlow được nạp khi tạo generator để train.

Chạy từ thư mục gốc dự án:

```powershell
python -m dataset.data_preprocessing.prepare_data
```

Nếu sử dụng môi trường `.venv` của dự án:

```powershell
.\.venv\Scripts\python.exe -m dataset.data_preprocessing.prepare_data
```

Các thư mục lớp hiện nằm trực tiếp trong `dataset/raw/`. Lệnh mặc định đọc vị trí
này và cũng hỗ trợ dữ liệu còn bọc trong thư mục `dataset/` sau khi giải nén.
Chỉ lấy các lớp `a`–`z` (cả tên thư mục dạng `a_lower`), bỏ file không
phải ảnh và cảnh báo khi bỏ qua ảnh hỏng. Không thay đổi ảnh gốc.

Kết quả được lưu vào `dataset/processed/`:

- `train.csv`: khoảng 80% số mẫu.
- `test.csv`: khoảng 10% số mẫu.
- `validation.csv`: khoảng 10% số mẫu.
- `class_indices.json`: ánh xạ nhãn sang chỉ số lớp, theo thứ tự chữ cái.
- `summary.json`: seed, tỷ lệ, số mẫu mỗi tập và số mẫu mỗi lớp.

Hai cột CSV là `image,label`; đường dẫn ảnh là đường dẫn tuyệt đối trên máy đã
chạy tiền xử lý. Chạy lại lệnh nếu di chuyển dataset sang vị trí hoặc máy khác.
Số mẫu test và validation được làm tròn lên, train nhận phần còn lại. Các tập được
phân tầng theo lớp, không trùng đường dẫn mẫu và có thể tái lập với cùng dataset
và seed. Dataset quá nhỏ có thể không đủ mẫu để phân tầng cả ba tập.

Có thể chỉ định đường dẫn và seed:

```powershell
python -m dataset.data_preprocessing.prepare_data --dataset dataset/raw --output dataset/processed --seed 41
```

Ảnh được chuyển RGB, resize 32×32 và dùng dtype `int32` khi generator đọc batch.
Chỉ tập train có augmentation độ sáng `[0.0, 1.0]`; validation/test không augmentation
và không shuffle. Giữ thang pixel 0–255 như notebook, không chia 255 hoặc áp dụng
`preprocess_input`. Otsu và dilation dành cho xử lý ảnh văn bản bên ngoài.
Không xuất bản sao ảnh resize; CSV trỏ tới ảnh gốc và xử lý ảnh diễn ra lúc train.

Cả 4 script train hiện có gọi chung `split_dataframe`, nên đều dùng tỷ lệ mặc định
80/10/10. Hàm trả về theo thứ tự **train, validation, test**. Có thể sử dụng trực tiếp
các tập đã lưu:

```python
import pandas as pd
from config import PROCESSED_DATA_DIR
from dataset.data_preprocessing import create_generators

training = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
validation = pd.read_csv(PROCESSED_DATA_DIR / "validation.csv")
testing = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
train_gen, valid_gen, test_gen = create_generators(training, validation, testing)
```

Các script train nhận đường dẫn ảnh gốc và chạy từ thư mục gốc dự án. Nếu dùng
môi trường `.venv` trên Windows, thay `python` bằng
`.\.venv\Scripts\python.exe`:

```powershell
python -m baseline_models.custom_cnn_train dataset/raw --epochs 20
python -m baseline_models.efficientnetb7_train dataset/raw --epochs 10
python -m baseline_models.mobilenetv2_train dataset/raw --epochs 10
python -m models.densenet121_train dataset/raw --epochs 10
```

`config.py` định nghĩa `DATASET_DIR` (gốc `dataset/`), `RAW_DATA_DIR`
(`dataset/raw/`) và `PROCESSED_DATA_DIR` (`dataset/processed/`).

Với ảnh đầu vào bên ngoài, pipeline xử lý gồm:

- Chuyển ảnh RGB sang ảnh grayscale.
- Nhị phân hóa ảnh bằng phương pháp Otsu.
- Dùng phép giãn hình thái học để nối các vùng ký tự hoặc vùng từ.
- Tìm các hình chữ nhật bao quanh bằng contour detection.
- Tách ảnh theo từng từ và từng ký tự.
- Chuẩn hóa ký tự về ảnh kích thước chuẩn trước khi đưa vào mô hình nhận dạng.

## Mô hình cơ sở

Các mô hình cơ sở được dùng để so sánh trong thư mục `baseline_models/`:

- **Custom CNN:** mạng CNN tự xây dựng gồm các lớp tích chập, pooling, dropout và lớp fully connected.
- **EfficientNetB7:** mô hình học chuyển giao dựa trên EfficientNetB7.
- **MobileNetV2:** mô hình nhẹ, phù hợp cho các ứng dụng cần giảm chi phí tính toán.

Mỗi mô hình có một file kiến trúc và một file train riêng.

## Mô hình chính

**DenseNet121** là mô hình chính của đồ án và được đặt trong thư mục `models/`. Mô hình sử dụng DenseNet121 để trích xuất đặc trưng, sau đó thêm các lớp fully connected để phân loại ký tự.

DenseNet121 được chọn làm mô hình chính vì khả năng tái sử dụng đặc trưng giữa các lớp, hỗ trợ huấn luyện hiệu quả và phù hợp với giai đoạn nhận dạng ảnh bên ngoài của pipeline.

## Pipeline đồ án

```text
Dataset
   ↓
Đọc cấu trúc thư mục và tạo nhãn
   ↓
Chia train / validation / test
   ↓
Resize và tăng cường dữ liệu
   ↓
Huấn luyện DenseNet121 và các mô hình baseline
   ↓
Đánh giá mô hình
   ↓
Đọc ảnh bên ngoài
   ↓
Grayscale → Otsu → Dilation → Tìm vùng ký tự
   ↓
Nhận dạng từng ký tự bằng mô hình đã train
   ↓
Ghép các ký tự thành văn bản
```

Các mô hình sau khi huấn luyện được lưu trong thư mục `saved_models/`.

## Đánh giá mô hình

Sau khi train, chạy test riêng cho từng model. Mỗi lệnh đánh giá model trên tập
`dataset/processed/test.csv` và lưu báo cáo JSON vào `evaluation/output/`:

```powershell
python -m tests.custom_cnn_test
python -m tests.efficientnetb7_test
python -m tests.mobilenetv2_test
python -m tests.densenet121_test
```

Sau khi có đủ bốn báo cáo, tạo bảng so sánh và báo cáo tổng hợp:

```powershell
python -m evaluation.evaluation
```

Kết quả tổng hợp gồm `model_comparison.csv` và `all_model_reports.json`, cùng
các báo cáo riêng và `DanhGia.md` trong `evaluation/output/`.

## Các chỉ số đánh giá

- **Accuracy:** tỷ lệ dự đoán đúng trên tổng số mẫu.
- **Precision:** mức độ chính xác của các dự đoán thuộc một lớp.
- **Recall:** khả năng tìm đúng các mẫu thực sự thuộc một lớp.
- **F1-score:** trung bình điều hòa giữa precision và recall.
- **Classification report:** bảng tổng hợp precision, recall, F1-score và số lượng mẫu theo từng lớp.

## Kết quả đánh giá hiện tại

Bốn mô hình được so sánh trên cùng tập test gồm 8.520 ảnh. DenseNet121 hiện đứng
đầu với accuracy **96,16%** và weighted F1 **96,17%**. Xem bảng đầy đủ và nhận xét
tại [`evaluation/output/DanhGia.md`](evaluation/output/DanhGia.md); báo cáo từng model,
CSV so sánh và JSON tổng hợp cũng được lưu trong `evaluation/output/`.

## Phạm vi hiện tại

Dự án đã có tiền xử lý dữ liệu, huấn luyện bốn mô hình, đánh giá trên tập test và
lưu báo cáo. Pipeline OCR hoàn chỉnh cho ảnh bên ngoài — gồm tách vùng chữ, nhận
dạng từng ký tự và ghép kết quả thành văn bản — cùng giao diện `GUI/` chưa được
triển khai đầy đủ.

Kiểm thử:

```powershell
python -m pytest tests -q
```

Bài kiểm thử generator thực tế sẽ được bỏ qua nếu chưa cài TensorFlow; các kiểm thử
chia tập, đọc dữ liệu và xuất CSV không cần TensorFlow.
