# Sử dụng bản Python 3.10
FROM python:3.10

# Thiết lập working directory
WORKDIR /usr/src/app

# Copy requirements.txt vào trong container trước khi cài đặt dependencies
COPY requirements.txt ./requirements.txt

# Cài đặt các dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir -r requirements.txt

# Copy thư mục data và weights vào trong image (tùy chọn)
COPY checkbox_state_v2 ./checkbox_state_v2

COPY ./ultralytics ./ultralytics
COPY ./args.yaml .
RUN mkdir -p ./runs


# Copy script train và inference vào trong container
COPY train.py ./train.py
COPY inference.py ./inference.py

# Thêm arg cho đường dẫn data và weights
ARG DATA_PATH=./checkbox_state_v2/data/
ARG WEIGHTS_PATH=./runs/classify/train4/weights/best.pt

# Update settings trong script (tùy chỉnh nếu cần)
RUN sed -i "s|'./checkbox_state_v2/data/'|'$DATA_PATH'|g" train.py inference.py
RUN sed -i "s|'./runs/classify/train4/weights/best.pt'|'$WEIGHTS_PATH'|g" train.py inference.py

# Command mặc định khi chạy container
CMD ["bash"]
