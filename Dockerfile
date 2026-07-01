FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip setuptools
RUN pip install --no-cache-dir .
COPY app/ ./app/
COPY static/ ./static/
RUN useradd -u 1000 -m appuser && mkdir -p /app/data && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
