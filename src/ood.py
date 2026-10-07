"""Detect out-of-distribution inputs and combine them with confidence checks."""

from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


# Project paths
project_folder = Path(__file__).resolve().parent.parent
model_path = project_folder / "models" / "model.pkl"


# Load dataset
data = load_breast_cancer()

X = data.data
y = data.target


# Create the same train/validation/test split used during evaluation
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


# Calculate the reference distribution using training data only
feature_means = X_train.mean(axis=0)
feature_stds = X_train.std(axis=0)


# Create synthetic OOD samples
# These samples are deliberately shifted away from the training distribution.
rng = np.random.default_rng(42)

X_ood = rng.normal(
    loc=feature_means,
    scale=5 * feature_stds,
    size=(100, X.shape[1])
)


# Standardize samples using the training distribution
X_val_scaled = (
    (X_val - feature_means)
    / (feature_stds + 1e-8)
)

X_test_scaled = (
    (X_test - feature_means)
    / (feature_stds + 1e-8)
)

X_ood_scaled = (
    (X_ood - feature_means)
    / (feature_stds + 1e-8)
)


# Calculate distance from the training distribution
val_distance = np.sqrt(
    (X_val_scaled ** 2).mean(axis=1)
)

test_distance = np.sqrt(
    (X_test_scaled ** 2).mean(axis=1)
)

ood_distance = np.sqrt(
    (X_ood_scaled ** 2).mean(axis=1)
)


# Select OOD threshold using normal validation data
# Only the most unusual 1% of validation samples are flagged.
ood_threshold = np.percentile(val_distance, 99)


# Evaluate OOD detection on synthetic OOD samples
ood_detected = ood_distance > ood_threshold


print("\nNormal test OOD check:")

test_flagged_as_ood = test_distance > ood_threshold

print("Normal test samples:", len(X_test))
print("Incorrectly flagged as OOD:", test_flagged_as_ood.sum())
print(
    "False OOD rate:",
    f"{test_flagged_as_ood.mean():.2%}"
)


print("\nOOD detection results:")
print("OOD samples detected:", ood_detected.sum())
print("OOD samples missed:", (~ood_detected).sum())
print(
    "OOD detection rate:",
    f"{ood_detected.mean():.2%}"
)


print("\nSelected OOD threshold:", f"{ood_threshold:.2f}")


print("\nValidation distance:")
print("Average validation distance:", f"{val_distance.mean():.2f}")
print("Maximum validation distance:", f"{val_distance.max():.2f}")


print("\nOOD distance:")
print("Average OOD distance:", f"{ood_distance.mean():.2f}")
print("Minimum OOD distance:", f"{ood_distance.min():.2f}")
print("Maximum OOD distance:", f"{ood_distance.max():.2f}")


print("\nOOD samples created:", X_ood.shape)

print("\nTraining distribution calculated.")
print("Number of features:", len(feature_means))
print("Dataset shape:", X.shape)

print("\nModel loaded successfully!")
print("Model type:", type(model).__name__)


# Final decision system
# A sample is sent for review if:
# 1. It looks out-of-distribution, OR
# 2. The model is not confident enough.

confidence_threshold = 0.80

test_probabilities = model.predict_proba(X_test)
test_confidence = test_probabilities.max(axis=1)


final_decision = np.where(
    test_distance > ood_threshold,
    "REVIEW - OOD",
    np.where(
        test_confidence < confidence_threshold,
        "REVIEW - LOW CONFIDENCE",
        "ACCEPT"
    )
)


print("\nFinal decision system:")
print("Accepted:", np.sum(final_decision == "ACCEPT"))
print(
    "OOD reviews:",
    np.sum(final_decision == "REVIEW - OOD")
)
print(
    "Low-confidence reviews:",
    np.sum(final_decision == "REVIEW - LOW CONFIDENCE")
)