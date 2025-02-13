from flask import Flask, request, jsonify
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
import datetime

app = Flask(__name__)


# Helper function to get timestamp
def get_time():
    return datetime.datetime.now().strftime("%d-%m-%y %H:%M:%S")


# Directory for saving models
pickle_dir = "pickle_files"
os.makedirs(pickle_dir, exist_ok=True)

# Model filenames
knn_filename = os.path.join(pickle_dir, "knn_model.pkl")
rf_filename = os.path.join(pickle_dir, "rf_model.pkl")
dt_filename = os.path.join(pickle_dir, "dt_model.pkl")
ensemble_filename = os.path.join(pickle_dir, "mnist_ensemble_model.pkl")


@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "Welcome to the MNIST Ensemble Model API!"})


# Endpoint to get best model parameters
@app.route("/best_model_parameter", methods=["GET"])
def best_model_parameter():
    with open(ensemble_filename, "rb") as file:
        ensemble_model = pickle.load(file)

    # Load MNIST dataset for evaluation
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Evaluate each model and find the best one
    best_model = None
    best_accuracy = 0
    best_params = {}

    for name, estimator in ensemble_model.estimators:
        accuracy = accuracy_score(y_test, estimator.predict(X_test))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = name
            best_params = estimator.get_params()

    return jsonify({"best_model": best_model, "best_params": best_params})


# Endpoint to train and save the ensemble model
@app.route("/train", methods=["POST"])
def train():
    print(f"Training started at {get_time()}")

    # Load MNIST dataset
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define individual classifiers
    knn = KNeighborsClassifier(n_neighbors=3)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    dt = DecisionTreeClassifier(max_depth=10, random_state=42)

    # Train classifiers
    knn.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    dt.fit(X_train, y_train)

    # Save individual models
    with open(knn_filename, "wb") as file:
        pickle.dump(knn, file)
    with open(rf_filename, "wb") as file:
        pickle.dump(rf, file)
    with open(dt_filename, "wb") as file:
        pickle.dump(dt, file)

    # Define and train ensemble model
    ensemble_model = VotingClassifier(estimators=[("KNN", knn), ("RF", rf), ("DT", dt)], voting="hard")
    ensemble_model.fit(X_train, y_train)

    # Evaluate the ensemble model
    y_pred = ensemble_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Save the ensemble model
    with open(ensemble_filename, "wb") as file:
        pickle.dump(ensemble_model, file)

    print(f"Training completed at {get_time()}, Accuracy: {accuracy:.4f}")

    return jsonify({"message": "Model trained successfully!", "accuracy": accuracy})


# Endpoint to make predictions using the trained model
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_features = np.array(data["features"]).reshape(1, -1)

    with open(ensemble_filename, "rb") as file:
        ensemble_model = pickle.load(file)

    prediction = ensemble_model.predict(input_features)
    return jsonify({"prediction": int(prediction[0])})


if __name__ == "__main__":
    print("Flask server running at: http://127.0.0.1:5050")
    app.run(host="0.0.0.0", port=5050, debug=True, use_reloader=False)
