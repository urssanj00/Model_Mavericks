# Use a lightweight Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Use a lightweight Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all project files (to ensure all dependencies exist)
COPY . /app

# Set environment variable to prevent output buffering (logs are displayed in real-time)
ENV PYTHONUNBUFFERED=1

# Install required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose MLflow UI port (optional, only needed if running MLflow in the same container)
EXPOSE 5000

# Run the MLFlow script inside the container
CMD ["python", "./com/model_mavericks/ML_Flow.py", "--docker"]