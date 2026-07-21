# ==========================================================
# Iris Flower Classification
# CodeSoft Internship - Task 3
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
    classification_report,
    confusion_matrix
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

dataset_path = r"C:\Users\Admin\OneDrive\Desktop\CODE SOFT\TASKNO3\IRIS.csv"

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

print("\nStatistical Summary")
print(df.describe())

# -------------------------------------------------
# Species Distribution
# -------------------------------------------------

plt.figure(figsize=(7,5))

sns.countplot(x="species", data=df)

plt.title("Species Distribution")

plt.tight_layout()

plt.savefig(os.path.join(image_folder,
                         "species_distribution.png"))

plt.close()

# -------------------------------------------------
# Pair Plot
# -------------------------------------------------

pair = sns.pairplot(
    df,
    hue="species"
)

pair.fig.suptitle(
    "Pair Plot of Iris Dataset",
    y=1.02
)

pair.savefig(
    os.path.join(
        image_folder,
        "pairplot.png"
    )
)

plt.close()

plt.figure(figsize=(8,6))

sns.boxplot(
    data=df.drop("species",axis=1)
)

plt.title("Feature Distribution")

plt.tight_layout()

plt.savefig(
    os.path.join(
        image_folder,
        "feature_distribution.png"
    )
)

plt.close()
# -------------------------------------------------
# Correlation Heatmap
# -------------------------------------------------

plt.figure(figsize=(8,6))

sns.heatmap(
    df.drop("species",axis=1).corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    os.path.join(
        image_folder,
        "correlation_heatmap.png"
    )
)

plt.close()

print("\nEDA Completed")

# -------------------------------------------------
# Features and Target
# -------------------------------------------------

X = df.drop("species", axis=1)

y = df["species"]

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

    n_estimators=200,

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
    cmap="Greens",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted Species")

plt.ylabel("Actual Species")

plt.tight_layout()

plt.savefig(
    os.path.join(
        image_folder,
        "confusion_matrix.png"
    )
)

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

plt.figure(figsize=(8,5))

sns.barplot(

    x="Importance",

    y="Feature",

    data=importance

)

plt.title("Feature Importance")

plt.tight_layout()

plt.savefig(
    os.path.join(
        image_folder,
        "feature_importance.png"
    )
)

plt.close()

# -------------------------------------------------
# Sample Flower Prediction
# -------------------------------------------------

print("\n" + "="*60)
print("SAMPLE FLOWER PREDICTION")
print("="*60)

sample_flower = pd.DataFrame({

    "sepal_length":[5.1],

    "sepal_width":[3.5],

    "petal_length":[1.4],

    "petal_width":[0.2]

})

prediction = model.predict(sample_flower)

print(f"\nPredicted Species : {prediction[0]}")

# -------------------------------------------------
# Save Predictions
# -------------------------------------------------

prediction_df = X_test.copy()

prediction_df["Actual Species"] = y_test.values

prediction_df["Predicted Species"] = y_pred

prediction_df.to_csv(

    "iris_predictions.csv",

    index=False

)

print("\nPrediction file saved as:")
print("iris_predictions.csv")

# -------------------------------------------------
# Final Summary
# -------------------------------------------------

print("\n" + "="*60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("="*60)

print(f"Total Flowers     : {len(df)}")

print(f"Training Samples  : {len(X_train)}")

print(f"Testing Samples   : {len(X_test)}")

print(f"Model Accuracy    : {accuracy*100:.2f}%")

print("\nGenerated Files:")

print("✔ images/species_distribution.png")
print("✔ images/pairplot.png")
print("✔ images/correlation_heatmap.png")
print("✔ images/confusion_matrix.png")
print("✔ images/feature_importance.png")
print("✔ iris_predictions.csv")

print("\nThank you for using Iris Flower Classification!")
