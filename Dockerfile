FROM python:3.10-slim

# System deps: poppler for pdf2image, plus minimal libs for PaddleOCR (CPU)
RUN apt-get update && apt-get install -y \
    poppler-utils curl \
    build-essential pkg-config swig \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download PaddleOCR models at build time to avoid startup downloads and health check flaps
ENV PADDLEOCR_HOME=/root/.paddleocr
RUN set -euo pipefail; \
    mkdir -p /root/.paddleocr/whl/det/en/en_PP-OCRv3_det_infer \
             /root/.paddleocr/whl/rec/en/en_PP-OCRv4_rec_infer \
             /root/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer; \
    curl -L -o /tmp/en_PP-OCRv3_det_infer.tar https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar; \
    tar -xf /tmp/en_PP-OCRv3_det_infer.tar -C /root/.paddleocr/whl/det/en/en_PP-OCRv3_det_infer --strip-components=1; \
    rm /tmp/en_PP-OCRv3_det_infer.tar; \
    curl -L -o /tmp/en_PP-OCRv4_rec_infer.tar https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar; \
    tar -xf /tmp/en_PP-OCRv4_rec_infer.tar -C /root/.paddleocr/whl/rec/en/en_PP-OCRv4_rec_infer --strip-components=1; \
    rm /tmp/en_PP-OCRv4_rec_infer.tar; \
    curl -L -o /tmp/ch_ppocr_mobile_v2.0_cls_infer.tar https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar; \
    tar -xf /tmp/ch_ppocr_mobile_v2.0_cls_infer.tar -C /root/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer --strip-components=1; \
    rm /tmp/ch_ppocr_mobile_v2.0_cls_infer.tar

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
