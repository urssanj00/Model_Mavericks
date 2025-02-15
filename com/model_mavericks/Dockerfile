# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Ensure logs directory exists
RUN mkdir -p /app/logs

# Copy all project files
COPY . .

# Expose port for Flask API
EXPOSE 5050

# Environment variable for MLflow
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

# Mount volume for logs
VOLUME ["/mnt/logs"]

# Run the Flask application
CMD ["python", "EnsembleFlaskUI.py"]
