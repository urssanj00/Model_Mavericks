# Use lightweight Python image
FROM python:3.8

# Set working directory
WORKDIR /app

# Use a virtual environment for dependencies
ENV VENV_PATH=/opt/venv
RUN python -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

# Copy only necessary files first (for caching)
COPY requirements.txt .

# Install dependencies (avoids re-downloading)
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories inside the container
RUN mkdir -p /mlflow/logs /mlflow/best_models /app/com/model_mavericks

# Copy only necessary files (avoid copying large unwanted files)
COPY com/model_mavericks/*.py  com/model_mavericks/
COPY com/model_mavericks/*.json com/model_mavericks/
COPY com/model_mavericks/*.properties com/model_mavericks/


# Default command
CMD ["python", "com/model_mavericks/ML_Flow.py", "--docker"]
