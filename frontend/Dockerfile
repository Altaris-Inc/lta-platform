FROM python:3.11-slim
WORKDIR /app
COPY frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY frontend/ .
COPY sample_data/ ./sample_data/
CMD ["streamlit", "run", "app_api.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
