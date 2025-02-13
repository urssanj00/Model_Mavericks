from flask import Flask, request, jsonify
import joblib
import os
import numpy as np
import asyncio
from ML_Flow import MNISTModelTuning
from logger_config import logger

app = Flask(__name__)

# Directory where best models are saved
best_models_dir = "best_models"
knn_path = os.path.join(best_models_dir, "KNN.joblib")
rf_path = os.path.join(best_models_dir, "Random_Forest.joblib")
dt_path = os.path.join(best_models_dir, "Decision_Tree.joblib")
ensemble_path = os.path.join(best_models_dir, "VotingClassifier.joblib")

# Load best models
knn_model = joblib.load(knn_path) if os.path.exists(knn_path) else None
rf_model = joblib.load(rf_path) if os.path.exists(rf_path) else None
dt_model = joblib.load(dt_path) if os.path.exists(dt_path) else None
ensemble_model = joblib.load(ensemble_path) if os.path.exists(ensemble_path) else None


@app.route("/", methods=["GET"])
def root():
    logger.info(f"called root()")
    return jsonify({"message": "Welcome to the MNIST Ensemble Model API!"})

@app.route("/train", methods=["GET"])
async def train_and_save_best_model():
    logger.info(f"called train_and_save_best_model()")

    mnist_tuning = MNISTModelTuning()
    logger.info(f"async loop started")
    loop = asyncio.get_event_loop()
    logger.info(f"async loop {loop}")
    await loop.run_in_executor(None, mnist_tuning.track_experiment)
    logger.info(f"async loop.run_in_executor Training started successfully!")
    return jsonify({"message": "Training started successfully!"})

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    logger.info(f"predict -data {data}")
    input_features = np.array(data["features"]).reshape(-1, 28 * 28)  # Ensure correct shape for MNIST images
    logger.info(f"predict -input_features {input_features}")

    if ensemble_model:
        predictions = ensemble_model.predict(input_features)
        logger.info(f"predict -predictions {predictions}")

        return jsonify({"predictions": predictions.tolist()})
    else:
        return jsonify({"error": "Ensemble model not found"}), 500


if __name__ == "__main__":
    print("Flask server running at: http://127.0.0.1:5050")
    app.run(host="0.0.0.0", port=5050, debug=True, use_reloader=False)
