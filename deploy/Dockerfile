FROM python:3.11-slim

WORKDIR /app

COPY deploy/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what's needed for serving
COPY src/ src/
COPY deploy/ deploy/
COPY configs/ configs/

ENV MODEL_PATH=deploy/artifacts/model.onnx
ENV PYTHONPATH=/app

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "deploy.serve:app", "--host", "0.0.0.0", "--port", "8080"]
