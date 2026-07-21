# ==========================================================
# Titanic Survival Prediction
# CodeSoft Internship - Task 1
# Author: Shravan Shirodkar
# ==========================================================

# -------------------------------------------------
# Import Required Libraries
# -------------------------------------------------

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

# -------------------------------------------------
# Create Images Folder
# -------------------------------------------------

base_dir = os.path.dirname(os.path.abspath(__file__))

image_folder = os.path.join(base_dir, "images")

os.makedirs(image_folder, exist_ok=True)

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

dataset_path = r"C:\Users\Admin\OneDrive\Desktop\CODE SOFT\TASKNO1\Titanic-Dataset.csv"

df = pd.read_csv(dataset_path)

print("="*60)
print("Dataset Loaded Successfully")
print("="*60)

print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# -------------------------------------------------
# Survival Count
# -------------------------------------------------

plt.figure(figsize=(6,4))

sns.countplot(x="Survived", data=df)

plt.title("Survival Count")

plt.tight_layout()

plt.savefig(os.path.join(image_folder,"survival_count.png"))

plt.close()

# -------------------------------------------------
# Survival by Gender
# -------------------------------------------------

plt.figure(figsize=(6,4))

sns.countplot(
    x="Sex",
    hue="Survived",
    data=df
)

plt.title("Survival by Gender")

plt.tight_layout()

plt.savefig(os.path.join(image_folder,"survival_by_gender.png"))

plt.close()

# -------------------------------------------------
# Survival by Passenger Class
# -------------------------------------------------

plt.figure(figsize=(6,4))

sns.countplot(
    x="Pclass",
    hue="Survived",
    data=df
)

plt.title("Survival by Passenger Class")

plt.tight_layout()

plt.savefig(os.path.join(image_folder,"survival_by_passenger_class.png"))

plt.close()

# -------------------------------------------------
# Age Distribution
# -------------------------------------------------

plt.figure(figsize=(8,5))

sns.histplot(
    df["Age"],
    bins=30,
    kde=True
)

plt.title("Age Distribution")

plt.tight_layout()

plt.savefig(os.path.join(image_folder,"age_distribution.png"))

plt.close()

print("\nEDA Completed")

# -------------------------------------------------
# Data Cleaning
# -------------------------------------------------

print("\nCleaning Dataset...")

df["Age"].fillna(df["Age"].median(), inplace=True)

df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

df.drop("Cabin", axis=1, inplace=True)

df.drop(
    ["PassengerId","Name","Ticket"],
    axis=1,
    inplace=True
)

# -------------------------------------------------
# Encode Categorical Variables
# -------------------------------------------------

df = pd.get_dummies(
    df,
    drop_first=True
)

print("\nProcessed Dataset")

print(df.head())

# -------------------------------------------------
# Features and Target
# -------------------------------------------------

X = df.drop("Survived", axis=1)

y = df["Survived"]

# -------------------------------------------------
# Train Test Split
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42

)

print("\nTraining Samples :", X_train.shape)

print("Testing Samples :", X_test.shape)

# -------------------------------------------------
# Random Forest Model
# -------------------------------------------------

model = RandomForestClassifier(

    n_estimators=100,

    random_state=42

)

print("\nTraining Model...")

model.fit(X_train, y_train)

print("Training Completed")
# -------------------------------------------------
# Make Predictions
# -------------------------------------------------

print("\nMaking Predictions...")

y_pred = model.predict(X_test)

# -------------------------------------------------
# Model Accuracy
# -------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)

print(f"Accuracy : {accuracy*100:.2f}%")

# -------------------------------------------------
# Classification Report
# -------------------------------------------------

print("\nClassification Report")

print(classification_report(y_test, y_pred))

# -------------------------------------------------
# Confusion Matrix
# -------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(os.path.join(image_folder,"confusion_matrix.png"))

plt.close()

# -------------------------------------------------
# ROC Curve
# -------------------------------------------------

probability = model.predict_proba(X_test)[:,1]

fpr, tpr, threshold = roc_curve(
    y_test,
    probability
)

roc_score = roc_auc_score(
    y_test,
    probability
)

print(f"\nROC AUC Score : {roc_score:.4f}")

plt.figure(figsize=(6,5))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_score:.3f}"
)

plt.plot([0,1],[0,1],"r--")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.tight_layout()

plt.savefig(os.path.join(image_folder,"roc_curve.png"))

plt.close()

# -------------------------------------------------
# Feature Importance
# -------------------------------------------------

importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": model.feature_importances_

})

importance = importance.sort_values(

    by="Importance",

    ascending=False

)

print("\nFeature Importance")

print(importance)

plt.figure(figsize=(9,6))

sns.barplot(

    x="Importance",

    y="Feature",

    data=importance

)

plt.title("Feature Importance")

plt.tight_layout()

plt.savefig(os.path.join(image_folder,"feature_importance.png"))

plt.close()

# -------------------------------------------------
# Sample Passenger Prediction
# -------------------------------------------------

print("\n" + "="*60)
print("SAMPLE PASSENGER PREDICTION")
print("="*60)

sample_passenger = X.iloc[[0]]

prediction = model.predict(sample_passenger)

print(
    "Prediction :",
    "Survived" if prediction[0] == 1 else "Did Not Survive"
)

# -------------------------------------------------
# Save Predictions
# -------------------------------------------------

prediction_df = X_test.copy()

prediction_df["Actual"] = y_test.values

prediction_df["Predicted"] = y_pred

prediction_df.to_csv(

    "titanic_predictions.csv",

    index=False

)

print("\nPrediction file saved as:")
print("titanic_predictions.csv")

# -------------------------------------------------
# Final Summary
# -------------------------------------------------

print("\n" + "="*60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("="*60)

print(f"Total Passengers : {len(df)}")

print(f"Training Samples : {len(X_train)}")

print(f"Testing Samples  : {len(X_test)}")

print(f"Accuracy         : {accuracy*100:.2f}%")

print(f"ROC AUC Score    : {roc_score:.4f}")

print("\nGenerated Files:")

print("✔ images/survival_count.png")
print("✔ images/survival_by_gender.png")
print("✔ images/survival_by_passenger_class.png")
print("✔ images/age_distribution.png")
print("✔ images/confusion_matrix.png")
print("✔ images/roc_curve.png")
print("✔ images/feature_importance.png")
print("✔ titanic_predictions.csv")

print("\nThank you for using Titanic Survival Prediction!")
