# Model_Mavericks
VNIT AAI Second Sem MLOPS  Assignment2

Running Everything
Step 1: Build and Start Containers
From the Model_Mavericks/ directory, run:
$ docker-compose up --build

This will: 
✅ Start the MLflow tracking server (mlflow container).
✅ Start the model training process (training container).
✅ Store MLflow logs in mlruns/.

Step 2: Access MLflow UI
After starting the containers, open:

http://localhost:5000
Step 3: View Logs
To check logs from the training container:

docker logs model-training
📌 Summary
File	                        Purpose
Dockerfile (root)	            Defines the container for model training.
docker-compose.yml (root)	    Manages multiple containers (MLflow + Training).
requirements.txt (root)	        Lists required Python packages.
.dockerignore (root)	        Prevents unnecessary files from being copied into the container.
ML_Flow.py              	    Model training & MLflow logging script.
(inside com/model_mavericks/)
 