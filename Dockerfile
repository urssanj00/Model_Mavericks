# Use an official Python image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy all project files (preserving the directory structure)
COPY com/model_mavericks /app/com/model_mavericks
COPY requirements.txt /app/

# Install required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose MLflow UI port (optional, only needed if running MLflow in the same container)
EXPOSE 5000

# Command to run model training (passing --docker flag to ML_Flow.py)
CMD ["python", "com/model_mavericks/ML_Flow.py", "--docker"]
