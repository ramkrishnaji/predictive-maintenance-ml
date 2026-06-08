FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_TRACKING_URI=http://mlflow:5000

# Set work directory
WORKDIR /app

# Install basic system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and assets
COPY src/ /app/src/
COPY data/ /app/data/
COPY assets/ /app/assets/

# Expose the API port
EXPOSE 8000

# Run uvicorn server
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
