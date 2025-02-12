import logging

import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from tensorflow import keras
import logging


class RandomForestCV:
    def __init__(self):
        self.X = None
        self.y = None

        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.y_pred = None

        self.rfc = RandomForestClassifier()
        print("Custom Random Forest Classifier initialized.")

        self.param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }

        self.grid_search = None

    def load_split_dataset(self):
        try:
            (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
        except Exception as e:
            print(e)
            exit()

    def execute_gridsearchcv(self):
        try:
            print("Starting GridSearchCV...")
            self.grid_search = GridSearchCV(
                estimator=self.rfc,
                param_grid=self.param_grid,
                scoring='accuracy',
                cv=5,  # 5-fold cross-validation    2. Grid Search Implementation:
                n_jobs=-1,  # Use all CPU cores
                verbose=1  # Display progress
            )
            self.grid_search.fit(self.X_train, self.y_train)
            print("GridSearchCV completed.")
        except Exception as e:
            logging.error(f"Error during GridSearchCV: {e}")
            exit()

    # Step 4: Get the best hyperparameters and evaluate
    def eval_and_report(self):
        try:
            print(f"Best Hyperparameters:{self.grid_search.best_params_}")
            print(f"Best Cross-Validation Score:{self.grid_search.best_score_}")

            # Evaluate on the test set
            best_model = self.grid_search.best_estimator_
            self.y_pred = best_model.predict(self.X_test)

            print(f"Actual : Predicted")
            for i in range(len(self.y_test)):
                print(f"{self.y_test[i]}     : {self.y_pred[i]}")
            accuracy = accuracy_score(self.y_test, self.y_pred)
            print("Test Set Accuracy:", accuracy)

            # Print classification report
            print("Classification Report:")
            print(classification_report(self.y_test, self.y_pred))
        except Exception as e:
            print(f"Error during evaluation: {e}")
            exit()

# Set up experiment tracking
mlflow.set_experiment('Experiment_Hyperparameter_Tuning')

# Start an ML flow run
with mlflow.start_run():
    gscvi = RandomForestCV()
    gscvi.load_split_dataset()

    # Log hyperparameter grid
    mlflow.log_params({
        "n_estimators_range": gscvi.param_grid['n_estimators'],
        "max_depth_range": gscvi.param_grid['max_depth'],
        "min_samples_split_range": gscvi.param_grid['min_samples_split']
    })

    # Execute GridSearchCV
    gscvi.execute_gridsearchcv()

    # Log the best hyperparameters and the best score
    mlflow.log_params(gscvi.grid_search.best_params_)
    mlflow.log_metric("best_cv_accuracy", gscvi.grid_search.best_score_)

    # Log cross-validation scores
    for i, score in enumerate(gscvi.grid_search.cv_results_['mean_test_score']):
        mlflow.log_metric(f'cv_score_{i + 1}', score)
        print(f"cv_score_{i + 1}, {score}")

    # Log the best model
    input_example = gscvi.X_train[0].reshape(1, -1)  # Example input for reproducibility
    mlflow.sklearn.log_model(gscvi.grid_search.best_estimator_, "best_random_forest_model",
                             input_example=input_example)

    # Log test accuracy
    gscvi.eval_and_report()
    test_accuracy = accuracy_score(gscvi.y_test, gscvi.y_pred)
    mlflow.log_metric("test_accuracy", test_accuracy)

