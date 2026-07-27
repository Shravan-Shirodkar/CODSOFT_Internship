# ==========================================================
# Titanic Survival Prediction
# CodeSoft Internship - Task 1
# Author: Shravan Shirodkar
# ==========================================================

# ==========================================================
# 1. IMPORT REQUIRED LIBRARIES
# ==========================================================

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
    roc_curve,
)

# ==========================================================
# 2. PROJECT SETUP
# ==========================================================

print("=" * 70)
print("       TITANIC SURVIVAL PREDICTION")
print("       CodeSoft Internship - Task 1")
print("=" * 70)

# Current Project Folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset Path
DATASET_PATH = os.path.join(BASE_DIR, "Titanic-Dataset.csv")

# Folder to Store Graphs
IMAGE_FOLDER = os.path.join(BASE_DIR, "images")

# Create Images Folder if it doesn't exist
os.makedirs(IMAGE_FOLDER, exist_ok=True)

print("\nProject Folder:")
print(BASE_DIR)

print("\nDataset Path:")
print(DATASET_PATH)

# ==========================================================
# 3. LOAD DATASET
# ==========================================================

print("\n" + "=" * 70)
print("Loading Dataset...")
print("=" * 70)

try:
    df = pd.read_csv(DATASET_PATH)
    print("\nDataset Loaded Successfully!")
except FileNotFoundError:
    print("\nERROR: Titanic-Dataset.csv was not found.")
    print("Place the dataset in the same folder as this Python file.")
    exit()

# ==========================================================
# 4. BASIC DATASET INFORMATION
# ==========================================================

print("\n" + "=" * 70)
print("FIRST FIVE ROWS")
print("=" * 70)

print(df.head())

print("\n" + "=" * 70)
print("DATASET SHAPE")
print("=" * 70)

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

print("\n" + "=" * 70)
print("COLUMN NAMES")
print("=" * 70)

print(df.columns.tolist())

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

df.info()

print("\n" + "=" * 70)
print("STATISTICAL SUMMARY")
print("=" * 70)

print(df.describe())

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

print(df.isnull().sum())

print("\nDataset inspection completed successfully!")
# ==========================================================
# 5. EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================================

print("\n" + "=" * 70)
print("EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 70)

# ==========================================================
# Survival Count
# ==========================================================

print("\nDisplaying Graph 1 : Survival Count")

plt.figure(figsize=(6,4))

sns.countplot(
    x="Survived",
    data=df,
    palette="Set2"
)

plt.title("Survival Count")
plt.xlabel("Survived")
plt.ylabel("Number of Passengers")

plt.tight_layout()

plt.savefig(
    os.path.join(
        IMAGE_FOLDER,
        "survival_count.png"
    )
)

plt.show(block=True)

plt.close()

print("Graph Saved : images/survival_count.png")

# ==========================================================
# Survival by Gender
# ==========================================================

print("\nDisplaying Graph 2 : Survival by Gender")

plt.figure(figsize=(7,5))

sns.countplot(
    x="Sex",
    hue="Survived",
    data=df,
    palette="viridis"
)

plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")

plt.legend(
    title="Survived",
    labels=["No","Yes"]
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        IMAGE_FOLDER,
        "survival_by_gender.png"
    )
)

plt.show(block=True)

plt.close()

print("Graph Saved : images/survival_by_gender.png")

# ==========================================================
# Survival by Passenger Class
# ==========================================================

print("\nDisplaying Graph 3 : Survival by Passenger Class")

plt.figure(figsize=(7,5))

sns.countplot(
    x="Pclass",
    hue="Survived",
    data=df,
    palette="coolwarm"
)

plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")

plt.legend(
    title="Survived",
    labels=["No","Yes"]
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        IMAGE_FOLDER,
        "survival_by_passenger_class.png"
    )
)

plt.show(block=True)

plt.close()

print("Graph Saved : images/survival_by_passenger_class.png")

# ==========================================================
# Age Distribution
# ==========================================================

print("\nDisplaying Graph 4 : Age Distribution")

plt.figure(figsize=(8,5))

sns.histplot(
    df["Age"],
    bins=30,
    kde=True,
    color="steelblue"
)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    os.path.join(
        IMAGE_FOLDER,
        "age_distribution.png"
    )
)

plt.show(block=True)

plt.close()

print("Graph Saved : images/age_distribution.png")

# ==========================================================
# Correlation Heatmap
# ==========================================================

print("\nDisplaying Graph 5 : Correlation Heatmap")

plt.figure(figsize=(10,8))

numeric_df = df.select_dtypes(include=["number"])

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    os.path.join(
        IMAGE_FOLDER,
        "correlation_heatmap.png"
    )
)

plt.show(block=True)

plt.close()

print("Graph Saved : images/correlation_heatmap.png")

print("\nEDA Completed Successfully!")
# ==========================================================
# 6. DATA CLEANING
# ==========================================================

print("\n" + "=" * 70)
print("DATA CLEANING")
print("=" * 70)

# -----------------------------
# Missing Values Before Cleaning
# -----------------------------

print("\nMissing Values Before Cleaning")

print(df.isnull().sum())

# -----------------------------
# Fill Missing Age Values
# -----------------------------

df["Age"].fillna(
    df["Age"].median(),
    inplace=True
)

# -----------------------------
# Fill Missing Embarked Values
# -----------------------------

df["Embarked"].fillna(
    df["Embarked"].mode()[0],
    inplace=True
)

# -----------------------------
# Drop Cabin Column
# -----------------------------

df.drop(
    "Cabin",
    axis=1,
    inplace=True
)

# -----------------------------
# Drop Unnecessary Columns
# -----------------------------

df.drop(
    columns=[
        "PassengerId",
        "Name",
        "Ticket"
    ],
    inplace=True
)

print("\nMissing Values After Cleaning")

print(df.isnull().sum())

print("\nDataset Cleaned Successfully!")

# ==========================================================
# 7. ENCODE CATEGORICAL VARIABLES
# ==========================================================

print("\n" + "=" * 70)
print("ENCODING CATEGORICAL VARIABLES")
print("=" * 70)

df = pd.get_dummies(
    df,
    drop_first=True
)

print("\nEncoded Dataset")

print(df.head())

print("\nEncoded Dataset Shape")

print(df.shape)

# ==========================================================
# 8. FEATURES AND TARGET
# ==========================================================

print("\n" + "=" * 70)
print("PREPARING FEATURES AND TARGET")
print("=" * 70)

X = df.drop(
    "Survived",
    axis=1
)

y = df["Survived"]

print("\nFeatures Shape")

print(X.shape)

print("\nTarget Shape")

print(y.shape)

# ==========================================================
# 9. TRAIN TEST SPLIT
# ==========================================================

print("\n" + "=" * 70)
print("TRAIN TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print("\nTraining Samples")

print(X_train.shape)

print("\nTesting Samples")

print(X_test.shape)

print("\nTraining Labels")

print(y_train.shape)

print("\nTesting Labels")

print(y_test.shape)

print("\nData Split Completed Successfully!")

print("\n" + "=" * 70)
print("FIRST HALF OF PROJECT COMPLETED")
print("=" * 70)
# ==========================================================
# 10. TRAIN RANDOM FOREST MODEL
# ==========================================================

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST MODEL")
print("=" * 70)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# ==========================================================
# 11. MAKE PREDICTIONS
# ==========================================================

print("\n" + "=" * 70)
print("MAKING PREDICTIONS")
print("=" * 70)

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]

print("Predictions Generated Successfully!")

# ==========================================================
# 12. MODEL EVALUATION
# ==========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

roc_score = roc_auc_score(
    y_test,
    y_prob
)

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(f"Accuracy      : {accuracy*100:.2f}%")
print(f"ROC AUC Score : {roc_score:.4f}")

print("\nClassification Report")
print()

print(classification_report(
    y_test,
    y_pred
))

# ==========================================================
# 13. CONFUSION MATRIX
# ==========================================================

print("\nDisplaying Graph : Confusion Matrix")

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("Actual Label")

plt.tight_layout()

plt.savefig(
    os.path.join(
        IMAGE_FOLDER,
        "confusion_matrix.png"
    )
)

plt.show(block=True)

plt.close()

print("Graph Saved : images/confusion_matrix.png")

# ==========================================================
# 14. ROC CURVE
# ==========================================================

print("\nDisplaying Graph : ROC Curve")

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(6,5))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"AUC = {roc_score:.3f}"
)

plt.plot(
    [0,1],
    [0,1],
    "r--",
    linewidth=2
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("Receiver Operating Characteristic (ROC) Curve")

plt.legend(loc="lower right")

plt.tight_layout()

plt.savefig(
    os.path.join(
        IMAGE_FOLDER,
        "roc_curve.png"
    )
)

plt.show(block=True)

plt.close()

print("Graph Saved : images/roc_curve.png")

print("\n" + "=" * 70)
print("MODEL EVALUATION COMPLETED")
print("=" * 70)
# ==========================================================
# 15. FEATURE IMPORTANCE
# ==========================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop Important Features\n")
print(importance)

plt.figure(figsize=(10,6))

sns.barplot(
    data=importance,
    x="Importance",
    y="Feature",
    palette="viridis"
)

plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")

plt.tight_layout()

plt.savefig(
    os.path.join(
        IMAGE_FOLDER,
        "feature_importance.png"
    )
)

plt.show(block=True)

plt.close()

print("Graph Saved : images/feature_importance.png")

# ==========================================================
# 16. SAMPLE PASSENGER PREDICTION
# ==========================================================

print("\n" + "=" * 70)
print("SAMPLE PASSENGER PREDICTION")
print("=" * 70)

sample_passenger = X.iloc[[0]]

prediction = model.predict(sample_passenger)[0]

if prediction == 1:
    result = "Survived"
else:
    result = "Did Not Survive"

print("\nPrediction Result")
print("----------------------------")
print(result)

# ==========================================================
# 17. SAVE TEST PREDICTIONS
# ==========================================================

prediction_df = X_test.copy()

prediction_df["Actual"] = y_test.values
prediction_df["Predicted"] = y_pred

prediction_file = os.path.join(
    BASE_DIR,
    "titanic_predictions.csv"
)

prediction_df.to_csv(
    prediction_file,
    index=False
)

print("\nPrediction File Saved Successfully!")
print(prediction_file)

# ==========================================================
# 18. SAVE FEATURE IMPORTANCE CSV
# ==========================================================

importance.to_csv(
    os.path.join(
        BASE_DIR,
        "feature_importance.csv"
    ),
    index=False
)

print("Feature Importance CSV Saved!")

# ==========================================================
# 19. FINAL PROJECT SUMMARY
# ==========================================================

print("\n" + "=" * 70)
print("PROJECT SUMMARY")
print("=" * 70)

print(f"Total Passengers      : {len(df)}")
print(f"Training Samples      : {len(X_train)}")
print(f"Testing Samples       : {len(X_test)}")
print(f"Model Accuracy        : {accuracy*100:.2f}%")
print(f"ROC AUC Score         : {roc_score:.4f}")

print("\nGenerated Files")
print("-" * 70)

print("images/survival_count.png")
print("images/survival_by_gender.png")
print("images/survival_by_passenger_class.png")
print("images/age_distribution.png")
print("images/correlation_heatmap.png")
print("images/confusion_matrix.png")
print("images/roc_curve.png")
print("images/feature_importance.png")
print("titanic_predictions.csv")
print("feature_importance.csv")

print("\n" + "=" * 70)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 70)

print("""
Project Workflow

1. Loaded Titanic Dataset
2. Explored Dataset
3. Visualized Data
4. Cleaned Missing Values
5. Encoded Categorical Features
6. Split Dataset
7. Trained Random Forest Model
8. Evaluated Model
9. Generated Graphs
10. Saved Predictions
11. Completed Project
""")

print("Thank you for using Titanic Survival Prediction!")
