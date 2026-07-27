# ==========================================================
# Movie Rating Prediction with Python
# CodeSoft Internship - Task 2
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

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
# ==========================================================
# 2. PROJECT SETUP
# ==========================================================

print("=" * 70)
print("        MOVIE RATING PREDICTION WITH PYTHON")
print("          CodeSoft Internship - Task 2")
print("=" * 70)

# Current Project Folder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset Path

DATASET_PATH = os.path.join(
    BASE_DIR,
    "IMDb Movies India.csv"
)

# Images Folder

IMAGE_FOLDER = os.path.join(
    BASE_DIR,
    "images"
)

# Create Images Folder

os.makedirs(
    IMAGE_FOLDER,
    exist_ok=True
)

print("\nProject Folder")

print(BASE_DIR)

print("\nDataset Path")

print(DATASET_PATH)

print("\nImages Folder Ready!")
# ==========================================================
# 3. LOAD DATASET
# ==========================================================

print("\n" + "=" * 70)
print("LOADING DATASET")
print("=" * 70)

try:

    df = pd.read_csv(
        DATASET_PATH,
        encoding="latin1"
    )

    print("\nDataset Loaded Successfully!")

except FileNotFoundError:

    print("\nERROR : IMDb Movies India.csv not found.")
    print("Place the dataset in the same folder as this Python file.")

    exit()

# ==========================================================
# 4. DATASET PREVIEW
# ==========================================================

print("\n" + "=" * 70)
print("FIRST FIVE ROWS")
print("=" * 70)

print(df.head())

print("\n" + "=" * 70)
print("LAST FIVE ROWS")
print("=" * 70)

print(df.tail())

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

print(df.describe(include="all"))

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
# Rating Distribution
# ==========================================================

print("\nDisplaying Graph 1 : Movie Rating Distribution")

plt.figure(figsize=(8,5))

sns.histplot(
    df["Rating"],
    bins=20,
    kde=True,
    color="royalblue"
)

plt.title("Movie Rating Distribution")

plt.xlabel("Movie Rating")

plt.ylabel("Number of Movies")

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        IMAGE_FOLDER,
        "rating_distribution.png"
    )
)

# Show graph on screen
plt.show(block=True)

plt.close()

print("Graph Saved : images/rating_distribution.png")
# ==========================================================
# Top 10 Movie Genres
# ==========================================================

print("\nDisplaying Graph 2 : Top 10 Movie Genres")

genre_counts = (
    df["Genre"]
    .value_counts()
    .head(10)
)

plt.figure(figsize=(10,6))

sns.barplot(
    x=genre_counts.values,
    y=genre_counts.index,
    palette="viridis"
)

plt.title("Top 10 Movie Genres")

plt.xlabel("Number of Movies")

plt.ylabel("Genre")

plt.grid(axis="x", alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        IMAGE_FOLDER,
        "top_genres.png"
    )
)

plt.show(block=True)

plt.close()

print("Graph Saved : images/top_genres.png")
# ==========================================================
# Correlation Heatmap
# ==========================================================

print("\nDisplaying Graph 3 : Correlation Heatmap")

temp = df.copy()

encoder = LabelEncoder()

categorical_columns = [
    "Genre",
    "Director",
    "Actor 1",
    "Actor 2",
    "Actor 3"
]

for column in categorical_columns:

    temp[column] = encoder.fit_transform(
        temp[column].astype(str)
    )

# Convert Numeric Columns

temp["Votes"] = (
    temp["Votes"]
    .astype(str)
    .str.replace(",", "")
)

temp["Votes"] = pd.to_numeric(
    temp["Votes"],
    errors="coerce"
)

temp["Duration"] = (
    temp["Duration"]
    .astype(str)
    .str.extract("(\d+)")
)

temp["Duration"] = pd.to_numeric(
    temp["Duration"],
    errors="coerce"
)

temp["Year"] = (
    temp["Year"]
    .astype(str)
    .str.extract("(\d{4})")
)

temp["Year"] = pd.to_numeric(
    temp["Year"],
    errors="coerce"
)

temp.fillna(temp.median(numeric_only=True), inplace=True)

corr = temp[
    [
        "Year",
        "Duration",
        "Votes",
        "Genre",
        "Director",
        "Actor 1",
        "Actor 2",
        "Actor 3",
        "Rating"
    ]
].corr()

plt.figure(figsize=(10,8))

sns.heatmap(
    corr,
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
# ==========================================================
# Movies Released Per Year
# ==========================================================

print("\nDisplaying Graph 4 : Movies Released Per Year")

year_data = (
    df["Year"]
    .astype(str)
    .str.extract("(\d{4})")[0]
)

year_data = pd.to_numeric(
    year_data,
    errors="coerce"
)

plt.figure(figsize=(10,5))

sns.histplot(
    year_data,
    bins=30,
    color="orange"
)

plt.title("Movies Released Per Year")

plt.xlabel("Year")

plt.ylabel("Number of Movies")

plt.tight_layout()

plt.savefig(
    os.path.join(
        IMAGE_FOLDER,
        "movies_per_year.png"
    )
)

plt.show(block=True)

plt.close()

print("Graph Saved : images/movies_per_year.png")
# ==========================================================
# Top 10 Directors
# ==========================================================

print("\nDisplaying Graph 5 : Top Directors")

director_counts = (
    df["Director"]
    .value_counts()
    .head(10)
)

plt.figure(figsize=(10,6))

sns.barplot(
    x=director_counts.values,
    y=director_counts.index,
    palette="magma"
)

plt.title("Top 10 Directors")

plt.xlabel("Number of Movies")

plt.ylabel("Director")

plt.tight_layout()

plt.savefig(
    os.path.join(
        IMAGE_FOLDER,
        "top_directors.png"
    )
)

plt.show(block=True)

plt.close()

print("Graph Saved : images/top_directors.png")

print("\nEDA Completed Successfully!")
# ==========================================================
# 6. DATA CLEANING
# ==========================================================

print("\n" + "=" * 70)
print("DATA CLEANING")
print("=" * 70)

print("\nMissing Values Before Cleaning\n")

print(df.isnull().sum())

# ----------------------------------------------------------
# Remove rows where Rating is missing
# ----------------------------------------------------------

df = df.dropna(subset=["Rating"])

print("\nRows with missing ratings removed.")

# ----------------------------------------------------------
# Fill Missing Categorical Values
# ----------------------------------------------------------

categorical_columns = [
    "Genre",
    "Director",
    "Actor 1",
    "Actor 2",
    "Actor 3"
]

for column in categorical_columns:

    df[column] = df[column].fillna("Unknown")

# ----------------------------------------------------------
# Fill Missing Numeric Values
# ----------------------------------------------------------

df["Votes"] = df["Votes"].fillna("0")

df["Duration"] = df["Duration"].fillna("0")

df["Year"] = df["Year"].fillna("0")

# ----------------------------------------------------------
# Convert Votes Column
# ----------------------------------------------------------

df["Votes"] = (
    df["Votes"]
    .astype(str)
    .str.replace(",", "")
)

df["Votes"] = pd.to_numeric(
    df["Votes"],
    errors="coerce"
)

# ----------------------------------------------------------
# Convert Duration Column
# ----------------------------------------------------------

df["Duration"] = (
    df["Duration"]
    .astype(str)
    .str.extract("(\d+)")
)

df["Duration"] = pd.to_numeric(
    df["Duration"],
    errors="coerce"
)

# ----------------------------------------------------------
# Convert Year Column
# ----------------------------------------------------------

df["Year"] = (
    df["Year"]
    .astype(str)
    .str.extract("(\d{4})")
)

df["Year"] = pd.to_numeric(
    df["Year"],
    errors="coerce"
)

# ----------------------------------------------------------
# Fill Remaining Missing Values
# ----------------------------------------------------------

df["Votes"].fillna(
    df["Votes"].median(),
    inplace=True
)

df["Duration"].fillna(
    df["Duration"].median(),
    inplace=True
)

df["Year"].fillna(
    df["Year"].median(),
    inplace=True
)

# ----------------------------------------------------------
# Remove Duplicate Records
# ----------------------------------------------------------

duplicates = df.duplicated().sum()

print(f"\nDuplicate Rows : {duplicates}")

if duplicates > 0:

    df.drop_duplicates(inplace=True)

    print("Duplicate rows removed successfully.")

else:

    print("No duplicate rows found.")

print("\nMissing Values After Cleaning\n")

print(df.isnull().sum())

print("\nDataset Cleaned Successfully!")

print("\nDataset Shape After Cleaning")

print(df.shape)
# ==========================================================
# 7. LABEL ENCODING
# ==========================================================

print("\n" + "=" * 70)
print("LABEL ENCODING")
print("=" * 70)

genre_encoder = LabelEncoder()
director_encoder = LabelEncoder()
actor1_encoder = LabelEncoder()
actor2_encoder = LabelEncoder()
actor3_encoder = LabelEncoder()

df["Genre"] = genre_encoder.fit_transform(df["Genre"])

df["Director"] = director_encoder.fit_transform(
    df["Director"]
)

df["Actor 1"] = actor1_encoder.fit_transform(
    df["Actor 1"]
)

df["Actor 2"] = actor2_encoder.fit_transform(
    df["Actor 2"]
)

df["Actor 3"] = actor3_encoder.fit_transform(
    df["Actor 3"]
)

print("\nCategorical Columns Encoded Successfully!")

# ==========================================================
# 8. FEATURES AND TARGET
# ==========================================================

print("\n" + "=" * 70)
print("FEATURES AND TARGET")
print("=" * 70)

feature_columns = [
    "Genre",
    "Year",
    "Duration",
    "Votes",
    "Director",
    "Actor 1",
    "Actor 2",
    "Actor 3"
]

X = df[feature_columns]

y = df["Rating"]

print("\nFeatures Shape")

print(X.shape)

print("\nTarget Shape")

print(y.shape)

print("\nSelected Features")

for feature in feature_columns:
    print("•", feature)

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

    random_state=42

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
# 10. TRAIN RANDOM FOREST REGRESSOR
# ==========================================================

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST REGRESSOR")
print("=" * 70)

model = RandomForestRegressor(

    n_estimators=200,

    random_state=42,

    n_jobs=-1

)

print("\nTraining Model...")

model.fit(
    X_train,
    y_train
)

print("Model Trained Successfully!")

# ==========================================================
# 11. MAKE PREDICTIONS
# ==========================================================

print("\n" + "=" * 70)
print("MAKING PREDICTIONS")
print("=" * 70)

y_pred = model.predict(X_test)

print("Predictions Generated Successfully!")

# ==========================================================
# 12. MODEL EVALUATION
# ==========================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(f"Mean Absolute Error (MAE) : {mae:.4f}")

print(f"Mean Squared Error (MSE)  : {mse:.4f}")

print(f"Root Mean Squared Error   : {rmse:.4f}")

print(f"R² Score                  : {r2:.4f}")

print("\nModel Evaluation Completed Successfully!")
# ==========================================================
# 13. ACTUAL VS PREDICTED RATINGS
# ==========================================================

print("\nDisplaying Graph : Actual vs Predicted Ratings")

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.6,
    color="royalblue"
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    linewidth=2
)

plt.title("Actual vs Predicted Movie Ratings")

plt.xlabel("Actual Rating")

plt.ylabel("Predicted Rating")

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        IMAGE_FOLDER,
        "actual_vs_predicted.png"
    )
)

# Show graph
plt.show(block=True)

plt.close()

print("Graph Saved : images/actual_vs_predicted.png")
# ==========================================================
# 14. RESIDUAL PLOT
# ==========================================================

print("\nDisplaying Graph : Residual Plot")

residuals = y_test - y_pred

plt.figure(figsize=(8,6))

plt.scatter(
    y_pred,
    residuals,
    alpha=0.6,
    color="darkorange"
)

plt.axhline(
    y=0,
    color="red",
    linestyle="--",
    linewidth=2
)

plt.title("Residual Plot")

plt.xlabel("Predicted Rating")

plt.ylabel("Residual")

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        IMAGE_FOLDER,
        "residual_plot.png"
    )
)

# Show graph
plt.show(block=True)

plt.close()

print("Graph Saved : images/residual_plot.png")

print("\nEvaluation Graphs Completed Successfully!")
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

print("\nFeature Importance\n")

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

plt.grid(alpha=0.3)

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
# 16. SAMPLE MOVIE PREDICTION
# ==========================================================

print("\n" + "=" * 70)
print("SAMPLE MOVIE PREDICTION")
print("=" * 70)

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

# ==========================================================
# 17. SAVE PREDICTIONS
# ==========================================================

prediction_df = pd.DataFrame({

    "Actual Rating": y_test.values,

    "Predicted Rating": y_pred

})

prediction_file = os.path.join(

    BASE_DIR,

    "movie_rating_predictions.csv"

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

importance_file = os.path.join(

    BASE_DIR,

    "feature_importance.csv"

)

importance.to_csv(

    importance_file,

    index=False

)

print("Feature Importance CSV Saved Successfully!")

# ==========================================================
# 19. FINAL PROJECT SUMMARY
# ==========================================================

print("\n" + "=" * 70)
print("PROJECT SUMMARY")
print("=" * 70)

print(f"Total Movies Used      : {len(df)}")
print(f"Training Samples       : {len(X_train)}")
print(f"Testing Samples        : {len(X_test)}")

print(f"\nMAE                    : {mae:.4f}")
print(f"MSE                    : {mse:.4f}")
print(f"RMSE                   : {rmse:.4f}")
print(f"R² Score               : {r2:.4f}")

print("\nGenerated Files")
print("-" * 70)

print("images/rating_distribution.png")
print("images/top_genres.png")
print("images/correlation_heatmap.png")
print("images/movies_per_year.png")
print("images/top_directors.png")
print("images/actual_vs_predicted.png")
print("images/residual_plot.png")
print("images/feature_importance.png")
print("movie_rating_predictions.csv")
print("feature_importance.csv")

print("\n" + "=" * 70)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 70)

print("""
Project Workflow

1. Loaded IMDb Movie Dataset
2. Explored Dataset
3. Visualized Data
4. Cleaned Dataset
5. Encoded Categorical Features
6. Split Dataset into Train and Test Sets
7. Trained Random Forest Regressor
8. Evaluated Model Performance
9. Generated Visualization Graphs
10. Saved Predictions
11. Completed Project Successfully
""")

print("Thank you for using Movie Rating Prediction!")
