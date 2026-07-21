# ==========================================================
# Movie Rating Prediction with Python
# CodeSoft Internship - Task 2
# Author: Shravan Shirodkar
# ==========================================================

# ----------------------------
# Import Required Libraries
# ----------------------------

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.ensemble import RandomForestRegressor

# -------------------------------------------------
# Create Images Folder
# -------------------------------------------------

if not os.path.exists("images"):
    os.makedirs("images")

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

dataset_path = r"C:\Users\Admin\OneDrive\Desktop\CODE SOFT\TASKNO2\IMDb Movies India.csv"

df = pd.read_csv(dataset_path, encoding="latin1")

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
# Data Cleaning
# -------------------------------------------------

print("\nCleaning Dataset...")

# Keep only rows where rating exists
df = df.dropna(subset=["Rating"])

# Fill missing values

df["Genre"] = df["Genre"].fillna("Unknown")
df["Director"] = df["Director"].fillna("Unknown")
df["Actor 1"] = df["Actor 1"].fillna("Unknown")
df["Actor 2"] = df["Actor 2"].fillna("Unknown")
df["Actor 3"] = df["Actor 3"].fillna("Unknown")

df["Votes"] = df["Votes"].fillna("0")
df["Duration"] = df["Duration"].fillna("0")
df["Year"] = df["Year"].fillna("0")

# -------------------------------------------------
# Convert Numeric Columns
# -------------------------------------------------

df["Votes"] = (
    df["Votes"]
    .astype(str)
    .str.replace(",", "")
)

df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce")

df["Duration"] = (
    df["Duration"]
    .astype(str)
    .str.extract("(\d+)")
)

df["Duration"] = pd.to_numeric(df["Duration"], errors="coerce")

df["Year"] = (
    df["Year"]
    .astype(str)
    .str.extract("(\d{4})")
)

df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

# Replace NaN values

df["Votes"].fillna(df["Votes"].median(), inplace=True)
df["Duration"].fillna(df["Duration"].median(), inplace=True)
df["Year"].fillna(df["Year"].median(), inplace=True)

print("\nDataset After Cleaning")

print(df.info())

# -------------------------------------------------
# Rating Distribution
# -------------------------------------------------

plt.figure(figsize=(8,5))

sns.histplot(df["Rating"], bins=20, color="royalblue")

plt.title("Movie Rating Distribution")

plt.xlabel("Rating")

plt.ylabel("Count")

plt.tight_layout()

plt.savefig("images/rating_distribution.png")

plt.close()

# -------------------------------------------------
# Top 10 Genres
# -------------------------------------------------

plt.figure(figsize=(10,6))

genre_counts = df["Genre"].value_counts().head(10)

sns.barplot(
    x=genre_counts.values,
    y=genre_counts.index
)

plt.title("Top 10 Movie Genres")

plt.xlabel("Count")

plt.ylabel("Genre")

plt.tight_layout()

plt.savefig("images/top_genres.png")

plt.close()

# -------------------------------------------------
# Correlation Heatmap
# -------------------------------------------------

temp = df.copy()

le = LabelEncoder()

for col in ["Genre","Director","Actor 1","Actor 2","Actor 3"]:

    temp[col] = le.fit_transform(temp[col])

corr = temp[[
    "Year",
    "Duration",
    "Votes",
    "Genre",
    "Director",
    "Actor 1",
    "Actor 2",
    "Actor 3",
    "Rating"
]].corr()

plt.figure(figsize=(10,8))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("images/correlation_heatmap.png")

plt.close()

print("\nEDA Completed")

# -------------------------------------------------
# Label Encoding
# -------------------------------------------------

genre_encoder = LabelEncoder()
director_encoder = LabelEncoder()
actor1_encoder = LabelEncoder()
actor2_encoder = LabelEncoder()
actor3_encoder = LabelEncoder()

df["Genre"] = genre_encoder.fit_transform(df["Genre"])
df["Director"] = director_encoder.fit_transform(df["Director"])
df["Actor 1"] = actor1_encoder.fit_transform(df["Actor 1"])
df["Actor 2"] = actor2_encoder.fit_transform(df["Actor 2"])
df["Actor 3"] = actor3_encoder.fit_transform(df["Actor 3"])

# -------------------------------------------------
# Features and Target
# -------------------------------------------------

X = df[[
    "Genre",
    "Year",
    "Duration",
    "Votes",
    "Director",
    "Actor 1",
    "Actor 2",
    "Actor 3"
]]

y = df["Rating"]

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

model = RandomForestRegressor(

    n_estimators=200,

    random_state=42,

    n_jobs=-1

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
# Model Evaluation
# -------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)

print(f"Mean Absolute Error (MAE) : {mae:.4f}")
print(f"Mean Squared Error (MSE)  : {mse:.4f}")
print(f"Root Mean Squared Error   : {rmse:.4f}")
print(f"R² Score                  : {r2:.4f}")

# -------------------------------------------------
# Actual vs Predicted Plot
# -------------------------------------------------

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.6
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    linewidth=2
)

plt.title("Actual vs Predicted Ratings")

plt.xlabel("Actual Rating")

plt.ylabel("Predicted Rating")

plt.tight_layout()

plt.savefig("images/actual_vs_predicted.png")

plt.close()

# -------------------------------------------------
# Residual Plot
# -------------------------------------------------

residuals = y_test - y_pred

plt.figure(figsize=(8,6))

plt.scatter(
    y_pred,
    residuals,
    alpha=0.6
)

plt.axhline(
    y=0,
    color="red",
    linestyle="--"
)

plt.title("Residual Plot")

plt.xlabel("Predicted Rating")

plt.ylabel("Residual")

plt.tight_layout()

plt.savefig("images/residual_plot.png")

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

plt.figure(figsize=(10,6))

sns.barplot(

    x="Importance",

    y="Feature",

    data=importance

)

plt.title("Feature Importance")

plt.tight_layout()

plt.savefig("images/feature_importance.png")

plt.close()

# -------------------------------------------------
# Sample Movie Prediction
# -------------------------------------------------

print("\n" + "="*60)
print("SAMPLE MOVIE PREDICTION")
print("="*60)

sample_movie = pd.DataFrame({

    "Genre":[df["Genre"].mode()[0]],

    "Year":[2024],

    "Duration":[150],

    "Votes":[250000],

    "Director":[df["Director"].mode()[0]],

    "Actor 1":[df["Actor 1"].mode()[0]],

    "Actor 2":[df["Actor 2"].mode()[0]],

    "Actor 3":[df["Actor 3"].mode()[0]]

})

predicted_rating = model.predict(sample_movie)

print(f"\nPredicted Movie Rating : {predicted_rating[0]:.2f}/10")

# -------------------------------------------------
# Save Predictions
# -------------------------------------------------

prediction_df = pd.DataFrame({

    "Actual Rating": y_test.values,

    "Predicted Rating": y_pred

})

prediction_df.to_csv(

    "movie_rating_predictions.csv",

    index=False

)

print("\nPrediction file saved as:")
print("movie_rating_predictions.csv")

# -------------------------------------------------
# Final Summary
# -------------------------------------------------

print("\n" + "="*60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("="*60)

print(f"Total Movies Used : {len(df)}")
print(f"Training Samples  : {len(X_train)}")
print(f"Testing Samples   : {len(X_test)}")

print("\nGenerated Files:")

print("✔ images/rating_distribution.png")
print("✔ images/top_genres.png")
print("✔ images/correlation_heatmap.png")
print("✔ images/actual_vs_predicted.png")
print("✔ images/residual_plot.png")
print("✔ images/feature_importance.png")
print("✔ movie_rating_predictions.csv")

print("\nThank you for using Movie Rating Prediction!")
