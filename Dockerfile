# Python 3.10 slim 이미지 기반
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 설치 (OpenCV에 필요한 libGL + libglib)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 필요한 파일 복사
COPY app.py .
COPY best.onnx .

# 의존성 설치
RUN pip install --no-cache-dir \
    flask \
    onnxruntime \
    numpy \
    opencv-python \
    pillow

# 서버 실행
CMD ["python", "app.py"]
