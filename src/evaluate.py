"""Evaluate model confidence, selective prediction, and calibration."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.model_selection import train_test_split


# Project paths
project_folder = Path(__file__).resolve().parent.parent
model_path = project_folder / "models" / "model.pkl"
results_folder = project_folder / "results"
results_folder.mkdir(exist_ok=True)


# Load dataset
data = load_breast_cancer()

X = data.data
y = data.target


# Create the same train/validation/test split
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


# Load trained model
model = joblib.load(model_path)


# Make predictions on the test set
test_predictions = model.predict(X_test)
test_probabilities = model.predict_proba(X_test)
test_confidence = test_probabilities.max(axis=1)

print(f"Average confidence: {test_confidence.mean():.2%}")
print(f"Minimum confidence: {test_confidence.min():.2%}")
print(f"Maximum confidence: {test_confidence.max():.2%}")


# Make predictions on the validation set
val_predictions = model.predict(X_val)
val_probabilities = model.predict_proba(X_val)
val_confidence = val_probabilities.max(axis=1)


# Evaluate different confidence thresholds on validation data
thresholds = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

print("\nValidation threshold experiment:")

for threshold in thresholds:
    accepted = val_confidence >= threshold
    coverage = accepted.mean()

    if accepted.sum() > 0:
        selective_accuracy = accuracy_score(
            y_val[accepted],
            val_predictions[accepted]
        )

        print(
            f"Threshold={threshold:.2f} | "
            f"Coverage={coverage:.2%} | "
            f"Accuracy={selective_accuracy:.2%}"
        )


# Final threshold selected using validation results
selected_threshold = 0.80

print(f"\nSelected threshold: {selected_threshold:.2f}")


# Evaluate selected threshold on the untouched test set
accepted = test_confidence >= selected_threshold
coverage = accepted.mean()

if accepted.sum() > 0:
    selective_accuracy = accuracy_score(
        y_test[accepted],
        test_predictions[accepted]
    )

    print("\nFinal test evaluation:")
    print(f"Test coverage: {coverage:.2%}")
    print(f"Accuracy on accepted predictions: {selective_accuracy:.2%}")
    print(f"Rejected predictions: {(~accepted).sum()}")


# Calculate overall test accuracy
accuracy = accuracy_score(y_test, test_predictions)

print(f"Test accuracy: {accuracy:.2%}")


# Calculate Brier score
brier_score = brier_score_loss(
    y_test,
    test_probabilities[:, 1]
)

print(f"Brier score: {brier_score:.4f}")


# Calibration analysis
fraction_of_positives, mean_predicted_value = calibration_curve(
    y_test,
    test_probabilities[:, 1],
    n_bins=10
)

plt.figure(figsize=(7, 7))

plt.plot(
    mean_predicted_value,
    fraction_of_positives,
    marker="o",
    label="Random Forest"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration"
)

plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.title("Model Calibration Curve")
plt.legend()
plt.grid()

calibration_path = results_folder / "calibration_curve.png"
plt.savefig(calibration_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"Calibration curve saved to {calibration_path}")