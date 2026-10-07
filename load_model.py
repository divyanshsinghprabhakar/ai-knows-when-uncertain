"""Load and verify the saved Random Forest model."""

from pathlib import Path

import joblib


# Get the project folder
project_folder = Path(__file__).resolve().parent

# Load the saved model
model_path = project_folder / "models" / "model.pkl"

model = joblib.load(model_path)

print("Model loaded successfully!")
print("Model type:", type(model).__name__)