import os
import json
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# =========================================================
# 1. File paths
# =========================================================

DATASET_PATH = "dataset/crop_yield.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "yield_prediction_model.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "yield_metrics.json")


# =========================================================
# 2. Load dataset
# =========================================================

print("\nLoading crop yield dataset...")

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully!")
print("Rows:", len(df))
print("Columns:", list(df.columns))


# =========================================================
# 3. Clean column names
# =========================================================

df.columns = df.columns.str.strip()


# =========================================================
# 4. Remove missing values
# =========================================================

required_columns = [
    "Crop",
    "Crop_Year",
    "Season",
    "State",
    "Area",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide",
    "Yield"
]

df = df.dropna(subset=required_columns)

print("\nAfter removing missing values:")
print("Rows:", len(df))


# =========================================================
# 5. Select features and target
# =========================================================

features = [
    "Crop",
    "Crop_Year",
    "Season",
    "State",
    "Area",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide"
]

target = "Yield"

X = df[features]
y = df[target]


# =========================================================
# 6. Feature types
# =========================================================

categorical_features = [
    "Crop",
    "Season",
    "State"
]

numeric_features = [
    "Crop_Year",
    "Area",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide"
]


# =========================================================
# 7. Preprocessing
# =========================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)


# =========================================================
# 8. Random Forest model
# =========================================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    max_depth=20
)


# =========================================================
# 9. Create complete ML pipeline
# =========================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# =========================================================
# 10. Train-test split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining data:", len(X_train))
print("Testing data:", len(X_test))


# =========================================================
# 11. Train model
# =========================================================

print("\nTraining Random Forest model...")
print("Please wait...")

pipeline.fit(X_train, y_train)

print("Model training completed!")


# =========================================================
# 12. Evaluate model
# =========================================================

y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R2 Score: {r2:.4f}")


# =========================================================
# 13. Create models folder
# =========================================================

os.makedirs(MODEL_DIR, exist_ok=True)


# =========================================================
# 14. Save trained model
# =========================================================

joblib.dump(pipeline, MODEL_PATH)

print("\nModel saved successfully:")
print(MODEL_PATH)


# =========================================================
# 15. Save model metrics
# =========================================================

metrics = {
    "model": "Random Forest Regressor",
    "dataset_rows": int(len(df)),
    "training_rows": int(len(X_train)),
    "testing_rows": int(len(X_test)),
    "mean_absolute_error": float(mae),
    "r2_score": float(r2)
}

with open(METRICS_PATH, "w") as file:
    json.dump(metrics, file, indent=4)

print("Metrics saved successfully:")
print(METRICS_PATH)


# =========================================================
# 16. Test prediction
# =========================================================

sample_data = pd.DataFrame([
    {
        "Crop": "Wheat",
        "Crop_Year": 2020,
        "Season": "Rabi",
        "State": "Punjab",
        "Area": 5,
        "Annual_Rainfall": 750,
        "Fertilizer": 5000,
        "Pesticide": 200,
    }
])

sample_prediction = pipeline.predict(sample_data)[0]

print("\n==============================")
print("SAMPLE PREDICTION")
print("==============================")
print(f"Estimated Yield: {sample_prediction:.2f} tons/hectare")

total_production = sample_prediction * 5

print(f"Estimated Total Production: {total_production:.2f} tons")
print("==============================")

print("\nYield prediction model is ready!")