# main.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve

# --- 1. Load Dataset ---
# The Breast Cancer Wisconsin dataset is a binary classification problem.
# Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.
# They describe characteristics of the cell nuclei present in the image.
# The goal is to predict whether the mass is malignant (1) or benign (0).
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = pd.Series(cancer.target)

print("--- Dataset Head ---")
print(X.head())
print("\n--- Target Distribution ---")
print(y.value_counts())
print("\n")


# --- 2. Train/Test Split and Feature Standardization --- [cite: 9]
# We split the data into training (80%) and testing (20%) sets.
# StandardScaler removes the mean and scales the features to unit variance.
# This is crucial for logistic regression to prevent features with larger scales from dominating the model.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --- 3. Fit a Logistic Regression Model --- [cite: 10]
# We create an instance of the LogisticRegression model and fit it to our scaled training data.
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)


# --- 4. Evaluate the Model --- [cite: 11]
print("--- Model Evaluation ---")
# Make predictions on the test set
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] # Probabilities for the positive class (1)

# Confusion Matrix
# A table showing the performance of the classification model.
# [[True Negative, False Positive], [False Negative, True Positive]]
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{cm}\n")

# Precision
# Of all the positive predictions, how many were actually correct? (TP / (TP + FP))
precision = precision_score(y_test, y_pred)
print(f"Precision: {precision:.4f}")

# Recall (Sensitivity)
# Of all the actual positive cases, how many did the model correctly identify? (TP / (TP + FN))
recall = recall_score(y_test, y_pred)
print(f"Recall: {recall:.4f}\n")

# ROC-AUC Score
# The Area Under the Receiver Operating Characteristic Curve.
# It measures the model's ability to distinguish between classes. A score of 1.0 is perfect.
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC Score: {roc_auc:.4f}\n")


# --- 5. Threshold Tuning and Sigmoid Function Explanation --- [cite: 12]

# The Sigmoid Function
# Logistic regression uses the sigmoid function to map any real-valued number into a value between 0 and 1.
# This output is interpreted as a probability.
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace(-10, 10, 100)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(z, sigmoid(z))
plt.title("Sigmoid Function")
plt.xlabel("z (Linear Combination of Inputs)")
plt.ylabel("Probability")
plt.grid(True)

# ROC Curve Visualization
# Plots the True Positive Rate (Recall) against the False Positive Rate at various threshold settings.
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Threshold Tuning
# By default, the probability threshold is 0.5. If prob > 0.5, predict 1, else 0.
# We can change this threshold to optimize for either precision or recall.
print("--- Threshold Tuning ---")
print("Default threshold (0.5) predictions are used above.")

# Example: Using a higher threshold (e.g., 0.8) to increase precision.
# This means we are more "confident" before classifying as positive (malignant).
high_precision_threshold = 0.8
y_pred_high_precision = (y_pred_proba >= high_precision_threshold).astype(int)

print(f"\nMetrics with a threshold of {high_precision_threshold}:")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_high_precision)}")
print(f"Precision: {precision_score(y_test, y_pred_high_precision):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_high_precision):.4f}")

# Example: Using a lower threshold (e.g., 0.3) to increase recall.
# This means we want to catch as many positive cases as possible, even at the risk of more false positives.
high_recall_threshold = 0.3
y_pred_high_recall = (y_pred_proba >= high_recall_threshold).astype(int)

print(f"\nMetrics with a threshold of {high_recall_threshold}:")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_high_recall)}")
print(f"Precision: {precision_score(y_test, y_pred_high_recall):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_high_recall):.4f}")
