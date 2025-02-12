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

    def load_and_preprocess_data(self):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        self.X_train = X_train.reshape(-1, 28 * 28) / 255.0
        self.X_test = X_test.reshape(-1, 28 * 28) / 255.0
        self.y_train, self.y_test = y_train, y_test
        logger.info("Data loaded and preprocessed successfully.")

    def run_gridsearch(self):
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }

        self.grid_search = GridSearchCV(
            estimator=self.rf_model, param_grid=param_grid,
            scoring='accuracy', cv=5, n_jobs=-1, verbose=1
        )

        self.grid_search.fit(self.X_train, self.y_train)
        self.best_model = self.grid_search.best_estimator_
        logger.info(f"Best Parameters: {self.grid_search.best_params_}")

    def evaluate_models(self):
        models = {
            'Random Forest': self.best_model,
            'KNN': self.knn_model.fit(self.X_train, self.y_train),
            'Decision Tree': self.dt_model.fit(self.X_train, self.y_train)
        }

        for name, model in models.items():
            predictions = model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, predictions)
            logger.info(f"{name} Accuracy: {accuracy:.4f}")
            logger.info(classification_report(self.y_test, predictions))

    def train_cnn(self):
        cnn_model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ])

        cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        cnn_model.fit(self.X_train.reshape(-1, 28, 28, 1), self.y_train, epochs=5, batch_size=32, validation_split=0.2)

        cnn_accuracy = cnn_model.evaluate(self.X_test.reshape(-1, 28, 28, 1), self.y_test, verbose=0)[1]
        logger.info(f"CNN Accuracy: {cnn_accuracy:.4f}")
        return cnn_model

    def track_experiment(self):
        mlflow.set_experiment("MNIST_Model_Tuning")
        with mlflow.start_run():
            self.load_and_preprocess_data()
            mlflow.log_params({
                "n_estimators_range": [50, 100, 150],
                "max_depth_range": [None, 10, 20],
                "min_samples_split_range": [2, 5, 10]
            })
            self.run_gridsearch()
            mlflow.log_params(self.grid_search.best_params_)
            mlflow.log_metric("best_cv_accuracy", self.grid_search.best_score_)

            self.evaluate_models()
            test_accuracy = accuracy_score(self.y_test, self.best_model.predict(self.X_test))
            mlflow.log_metric("test_accuracy", test_accuracy)

            mlflow.sklearn.log_model(self.best_model, "best_random_forest_model")

            cnn_model = self.train_cnn()
            mlflow.keras.log_model(cnn_model, "cnn_model")


if __name__ == "__main__":
    mnist_tuning = MNISTModelTuning()
    mnist_tuning.track_experiment()
