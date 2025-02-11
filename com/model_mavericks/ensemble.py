from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from logger_config import logger

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data, mnist.target.astype(int)  # Convert target to int
logger.info(f'Dataset loaded')
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logger.info(f'Train / Test Dataset separated')
# Standardize for KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
logger.info(f'Train / Test Dataset scaled')

# Define individual models test
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
logger.info(f'RandomForestClassifier {rf_clf}')

dt_clf = DecisionTreeClassifier(random_state=42)
logger.info(f'DecisionTreeClassifier {dt_clf}')

knn_clf = KNeighborsClassifier(n_neighbors=5)  # Replace SVM with KNN
logger.info(f'KNeighborsClassifier {knn_clf}')

# Create ensemble model with VotingClassifier
ensemble_clf = VotingClassifier(
    estimators=[('rf', rf_clf), ('dt', dt_clf), ('knn', knn_clf)],
    voting='hard'  # Majority voting
)
logger.info(f'VotingClassifier {ensemble_clf}')

# Train ensemble model
ensemble_clf.fit(X_train_scaled, y_train)
logger.info(f'VotingClassifier {ensemble_clf}')

# Predict on test data
y_pred = ensemble_clf.predict(X_test_scaled)
logger.info(f'y_pred {y_pred}')

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
logger.info(f'Ensemble Model Accuracy: {accuracy:.4f}')
