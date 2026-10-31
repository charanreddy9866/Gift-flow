# GiftFlow: runs the Streamlit app with real Tesseract OCR.
#   docker build -t giftflow .
#   docker run --rm -p 8501:8501 giftflow                      # offline mock AI
#   docker run --rm -p 8501:8501 --env-file .env giftflow      # real OpenAI (LLM_PROVIDER=openai)
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LLM_PROVIDER=mock \
    GIFTFLOW_DB=/data/giftflow.db

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN useradd --create-home app && mkdir -p /data /app/uploads && chown -R app /data /app
USER app

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
