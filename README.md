# AI That Knows When It Doesn't Know

An uncertainty-aware machine learning system that doesn't blindly trust its own predictions.

The system combines prediction confidence, calibration analysis, and out-of-distribution (OOD) detection to decide whether a prediction should be accepted or sent for human review.

## Why This Project?

Machine learning models can be highly confident even when they are wrong.

A prediction such as:

> "I'm 98% confident."

does not automatically mean the prediction is reliable.

This project explores a different approach: instead of forcing the model to predict everything, the system can recognize when a prediction may be unreliable and request human review.

The system asks three questions:

1. How confident is the model?
2. Are its confidence scores calibrated?
3. Does the input look like something the model has seen before?

If a sample fails these checks, it is flagged for review instead of being blindly accepted.

## System Architecture

```text
Input
  ↓
Random Forest Classifier
  ↓
Prediction + Confidence
  ↓
┌──────────────────────────┐
│ OOD Detection            │
│ Is the input unfamiliar? │
└────────────┬─────────────┘
             │
       Yes → REVIEW - OOD
             │ No
             ↓
┌──────────────────────────┐
│ Confidence Check         │
│ Confidence < 80%?        │
└────────────┬─────────────┘
             │
       Yes → REVIEW
             │ No
             ↓
          ACCEPT
Results
Classification
Metric	Result
Test accuracy	88.37%
Average confidence	92.33%
Brier score	0.0566
Selective Prediction

A confidence threshold of 0.80 was selected using the validation set.

On the untouched test set:

Metric	Result
Test coverage	83.72%
Accuracy on accepted predictions	100.00%
Rejected predictions	14 / 86
OOD Detection

The OOD threshold was selected from the 99th percentile of validation distances, giving a threshold of 3.22.

Metric	Result
Synthetic OOD samples	100
OOD samples detected	100
OOD detection rate	100.00%
Normal test samples	86
False OOD detections	1
False OOD rate	1.16%
Final Decision System

On the normal test set:

Decision	Samples
ACCEPT	71
REVIEW - LOW CONFIDENCE	14
REVIEW - OOD	1
Calibration Curve

The calibration curve compares predicted probabilities with observed outcomes.

Tech Stack
Python
NumPy — numerical computation
Pandas — data handling
scikit-learn — machine learning
Matplotlib — visualization
Joblib — model persistence
Git & GitHub — version control and project management
Project Structure
ai-knows-when-uncertain/
│
├── models/
│   └── model.pkl
│
├── results/
│   └── calibration_curve.png
│
├── src/
│   ├── train.py
│   ├── evaluate.py
│   └── ood.py
│
├── load_model.py
├── requirements.txt
├── .gitignore
└── README.md
What Each File Does
train.py — trains and saves the Random Forest model.
evaluate.py — evaluates accuracy, confidence, selective prediction, and calibration.
ood.py — performs baseline out-of-distribution detection and the final decision logic.
load_model.py — demonstrates loading the saved model.
requirements.txt — lists the Python dependencies required to run the project.
results/calibration_curve.png — visualization of model calibration.
Methodology
1. Data Splitting

The dataset is divided into three parts:

70% training
15% validation
15% test

The validation set is used for selecting thresholds, while the test set remains untouched until final evaluation.

2. Confidence-Based Selective Prediction

The Random Forest produces class probabilities for each prediction.

The highest class probability is treated as the model's confidence.

A threshold of 0.80 was selected using validation data.

Predictions below this threshold are sent for human review.

3. Calibration

Model confidence is evaluated using:

Calibration curve
Brier score

This helps determine whether predicted probabilities correspond reasonably well to actual outcomes.

4. Out-of-Distribution Detection

A simple distribution-distance baseline is used to identify inputs that are substantially different from the training distribution.

The reference distribution is calculated using the training data only.

The OOD threshold is selected from the 99th percentile of validation distances rather than being chosen arbitrarily.

5. Final Decision

The system combines both signals:

OOD detection
      +
Confidence threshold
      ↓
Accept or Request Human Review
Limitations

This project is currently a baseline implementation and has several limitations:

The OOD samples are synthetically generated rather than collected from a real external distribution.
The OOD detector uses a simple standardized feature-distance method.
The test set is relatively small, so calibration estimates can be noisy.
The current confidence threshold is optimized for this particular dataset and should not be assumed to generalize to other datasets.
The system currently uses a Random Forest classifier and has not yet been compared against other uncertainty-estimation approaches.

These limitations provide opportunities for future improvements.

Future Improvements
Evaluate OOD detection using real external datasets.
Compare multiple OOD detection techniques.
Add Expected Calibration Error (ECE).
Compare calibrated and uncalibrated probability estimates.
Experiment with neural-network uncertainty methods.
Build a FastAPI inference service.
Create a web interface for interactive predictions.
Add automated tests and experiment tracking.
Containerize the system with Docker.
Installation

Clone the repository and move into the project directory:

git clone https://github.com/divyanshsinghprabhakar/ai-knows-when-uncertain.git
cd ai-knows-when-uncertain

Create a virtual environment:

python -m venv .venv

Activate it on Windows PowerShell:

.\.venv\Scripts\Activate.ps1

Install the dependencies:

pip install -r requirements.txt
Usage
Train the model
python src/train.py
Evaluate the model
python src/evaluate.py
Run OOD detection
python src/ood.py

The trained model is saved to:

models/model.pkl