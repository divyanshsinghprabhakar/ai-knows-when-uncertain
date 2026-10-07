"""Train and save the Random Forest classification model."""

from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# Load dataset
data = load_breast_cancer()

X = data.data
y = data.target


# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)


# Create and train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# Save trained model
project_folder = Path(__file__).resolve().parent.parent
model_path = project_folder / "models" / "model.pkl"

joblib.dump(model, model_path)

print("Model trained successfully!")
print("Model saved to models/model.pkl")