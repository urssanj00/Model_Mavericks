import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from tensorflow.keras.datasets import mnist
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from logger_config import logger
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
# Argument parser to determine where the script is running
# send arg from command line to identify if the file is run from local or docker
parser = argparse.ArgumentParser(description="Run MNIST Model Tuning with MLflow.")
parser.add_argument("--docker", action="store_true", help="Indicate if running inside a Docker container.")
args = parser.parse_args()

# Set MLflow Tracking URI based on the environment
if args.docker:
    mlflow.set_tracking_uri("http://mlflow:5000")  # Docker container
    logger.info("Running inside Docker. Using MLflow at http://mlflow:5000")
else:
    mlflow.set_tracking_uri("http://localhost:5000")  # Local machine
    logger.info("Running locally. Using MLflow at http://localhost:5000")


class MNISTModelTuning:
    def __init__(self):
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.rf_model = RandomForestClassifier()
        self.knn_model = KNeighborsClassifier(n_neighbors=7)
        self.dt_model = DecisionTreeClassifier(random_state=100)
        self.grid_search = None
        self.best_model = None
        self.best_rf = None
        self.best_knn = None
        self.best_dt = None
        self.best_models = []
        self.ensemble_clf = None
        self.MLFLOW = mlflow

    import numpy as np

    def load_and_preprocess_data(self):
        #load only 1000 images
        logger.info("MNIST Data loading start")

        # Load MNIST dataset
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        # Normalize & reshape data
        self.X_train = X_train.reshape(-1, 784) / 255.0
        self.y_train = y_train

        self.X_test = X_test.reshape(-1, 784) / 255.0
        self.y_test = y_test

        # Take only a subset of 1000 test samples
        test_subset_size = 10000
        indices = np.random.choice(len(self.X_train), test_subset_size, replace=False)  # Randomly select 1000 indices

        self.X_train = self.X_train[indices]
        self.y_train = self.y_train[indices]

        logger.info(f"MNIST Data loaded: Train={self.X_train.shape}, Test={self.X_test.shape}")


    def load_and_preprocess_data_all(self):
        logger.info("MNIST Data loading start")
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        self.X_train = X_train.reshape(-1, 28 * 28) / 255.0
        self.X_test = X_test.reshape(-1, 28 * 28) / 255.0
        self.y_train, self.y_test = y_train, y_test
        logger.info("MNIST Data loaded and preprocessed successfully.")

    def run_gridsearch(self):
        logger.info("03.01 Running GridSearchCV for all models")

        # Define hyperparameter grids
        param_grids = {
            "Random Forest": {
                'n_estimators': [50, 100, 150],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            "KNN": {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            },#Will Check later
            "Decision Tree": {
                'criterion': ['gini', 'entropy'],
                #'max_depth': [None, 10, 20, 30],
                'max_depth': [10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
        }

        models = {
            "Random Forest": self.rf_model,
            "KNN": self.knn_model,
            "Decision Tree": self.dt_model
        }

        self.best_models = {}

        # Run GridSearchCV for each model
        for model_name, model in models.items():
            logger.info(f"Running GridSearchCV for {model_name}")
            grid_search = GridSearchCV(
                estimator=model, param_grid=param_grids[model_name],
                scoring='accuracy', cv=5, n_jobs=-1, verbose=1
            )
            logger.info(f"        GridSearchCV.fit for {model_name}")
            grid_search.fit(self.X_train, self.y_train)
            self.best_models[model_name] = grid_search.best_estimator_
            logger.info(f"        GridSearchCV.fit for {model_name}")
            logger.info(f"Best Parameters for {model_name}: {grid_search.best_params_}")

            # Get predictions from best model
            y_pred = self.best_models[model_name].predict(self.X_test)

            # Compute evaluation metrics
            metrics = {
                "accuracy": accuracy_score(self.y_test, y_pred),
                "precision": precision_score(self.y_test, y_pred, average="weighted"),
                "recall": recall_score(self.y_test, y_pred, average="weighted"),
                "f1_score": f1_score(self.y_test, y_pred, average="weighted")
            }

            logger.info(f"{model_name} Metrics: {metrics}")

            # Log into MLflow
            self.log_into_mlflow(model_name, param_grids[model_name], metrics, self.best_models[model_name])
            logger.info(f"{model_name} mlflow push completed")

        # Create ensemble model with VotingClassifier
        self.ensemble_clf = VotingClassifier(
            estimators=[
                ('rf', self.best_models["Random Forest"]),
                ('knn', self.best_models["KNN"]),
                ('dt', self.best_models["Decision Tree"])
            ],
            voting='hard'  # Majority voting
        )

        logger.info(f'VotingClassifier initialized: {self.ensemble_clf}')

        # Train ensemble model
        self.ensemble_clf.fit(self.X_train, self.y_train)
        logger.info(f'VotingClassifier trained successfully')

        # Evaluate ensemble model
        y_pred_ensemble = self.ensemble_clf.predict(self.X_test)

        ensemble_metrics = {
            "accuracy": accuracy_score(self.y_test, y_pred_ensemble),
            "precision": precision_score(self.y_test, y_pred_ensemble, average="weighted"),
            "recall": recall_score(self.y_test, y_pred_ensemble, average="weighted"),
            "f1_score": f1_score(self.y_test, y_pred_ensemble, average="weighted")
        }

        logger.info(f"VotingClassifier Metrics: {ensemble_metrics}")

        # Log ensemble model into MLflow
        self.log_into_mlflow("VotingClassifier", {}, ensemble_metrics, self.ensemble_clf)

    def log_into_mlflow_bak(self, model_name, param_grid, metrics, model):
        """Logs hyperparameters, metrics, model, and artifacts (confusion matrix) into MLflow."""

        with mlflow.start_run(run_name=model_name):
            # Flatten parameter grid before logging
            flat_params = {}
            for param, values in param_grid.items():
                if isinstance(values, list):  # Convert lists to strings to avoid MLflow errors
                    flat_params[param] = values
                else:
                    flat_params[param] = values
            logger.info(f"0. mlflow param_grid (flattened) {flat_params}")
            for key, value in flat_params.items():
                mlflow.log_param(key, value)  # Log individually instead of mlflow.log_params(flat_params)


            #mlflow.log_params(flat_params)

            logger.info(f"1. mlflow metrics {metrics}")

            # Log evaluation metrics
            for key, value in metrics.items():
                logger.info(f"2. mlflow metric {key} : {value}")
                mlflow.log_metric(key, value)

            # Log the trained model
            logger.info(f"3. mlflow log_model {model_name} : {model}")

            mlflow.sklearn.log_model(model, model_name.lower().replace(" ", "_"))
            logger.info(f"Logged {model_name} model, metrics, and confusion matrix in MLflow")

    def log_into_mlflow(self, model_name, param_grid, metrics, model):
        """Logs hyperparameters, metrics, model, and artifacts (confusion matrix) into MLflow."""
        # Log hyperparameters
        with mlflow.start_run():
            logger.info(f"0. mlflow param_grid {param_grid}")
            #mlflow.log_params(param_grid)
            flat_params = {}
            for param, values in param_grid.items():
                if isinstance(values, list):  # Convert lists to strings to avoid MLflow errors
                    flat_params[param] = str(values)
                else:
                    flat_params[param] = values
            logger.info(f"0. mlflow param_grid (flattened) {flat_params}")
            for key, value in flat_params.items():
                logger.info(f"Logging in mlflow {model_name} {key}:{value}")
                mlflow.log_param(key, value)  # Log individually instead of mlflow.log_params(flat_params)


    #        mlflow.log_params(flat_params)

            logger.info(f"1. mlflow metrics {metrics}")

            # Log evaluation metrics
            for key, value in metrics.items():
                logger.info(f"2. mlflow param_grid {key} : {value}")
                mlflow.log_metric(key, value)

            # Log the trained model
            logger.info(f"3. mlflow log_model {model_name} : {model}")

            mlflow.sklearn.log_model(model, model_name.lower().replace(" ", "_"))
            logger.info(f"Logged {model_name} model, metrics, and confusion matrix in MLflow")


    def save_best_models(self):
        """Saves the best models locally."""
        save_dir = "best_models"
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Saving Best Models")
        for model_name, model in self.best_models.items():
            model_path = os.path.join(save_dir, f"{model_name.replace(' ', '_')}.joblib")
            logger.info(f"Model Path {model_path}")
            joblib.dump(model, model_path)
            logger.info(f"Saved Best Model:{model_name} model at {model_path}")
            mlflow.log_artifact(model_path)
        
        # Save the ensemble model
        ensemble_path = os.path.join(save_dir, "VotingClassifier.joblib")
        logger.info(f"ensemble_path Model Path {ensemble_path}")

        joblib.dump(self.ensemble_clf, ensemble_path)
        logger.info(f"Saved ensemble model at {ensemble_path}")
        mlflow.log_artifact(ensemble_path)
    
    def track_experiment(self):
        mlflow.set_experiment("MNIST_Model_Tuning")
        logger.info("00. MNIST_Model_Tuning Experiment Started")

        logger.info("01. Load and Preprocess Data")
        self.load_and_preprocess_data()
        logger.info("02. Run GridSearch")
        self.run_gridsearch()
        logger.info("03. Save Best Models")
        self.save_best_models()

            #logger.info(f"10. Log best_model {self.best_model} in MLFLOW")




if __name__ == "__main__":
    mnist_tuning = MNISTModelTuning()
    mnist_tuning.track_experiment()
