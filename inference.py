from ultralytics import YOLO
import cv2
import argparse


# Parse đường dẨn ảnh từ dòng lệnh
parser = argparse.ArgumentParser(description='Run inference')
parser.add_argument('--image_path', type=str, help='Path to the image file.')
args = parser.parse_args()

# Đường dẫn tới tấm ảnh muốn dự đoán
image_path = args.image_path

# Tải mô hình YOLO đã được train
model = YOLO('./runs/classify/train4/weights/best.pt')

# Đọc ảnh bằng OpenCV
image = cv2.imread(image_path)

# Thực hiện dự đoán trên ảnh
results = model.predict(source=image)

for result in results:
    # Sử dụng top1 và top1conf để lấy lớp dự đoán hàng đầu và xác suất của nó
    top1_index = result.probs.top1
    top1_confidence = result.probs.top1conf
    top1_name = model.names[int(top1_index)]
    
    print(f"Class: {top1_name}, Confidence: {top1_confidence}")
    