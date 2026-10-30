import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from generate_data import generate_housing_data

ROOT = Path(__file__).resolve().parent
FIGS = ROOT / "figs"
MODELS = ROOT / "models"
FIGS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

def main():
    data_path = ROOT / "housing_data.csv"
    if not data_path.exists():
        df = generate_housing_data(data_path)
    else:
        df = pd.read_csv(data_path)

    print("\\nDataset shape:", df.shape)
    print("\\nMissing values:\\n", df.isnull().sum())
    print("\\nPrice summary:\\n", df["price"].describe())

    X = df.drop(columns=["price"])
    y = df["price"].astype(float)
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Original-target ANN": MLPRegressor(
            hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
            alpha=0.001, learning_rate_init=0.001, max_iter=500,
            early_stopping=True, validation_fraction=0.10,
            n_iter_no_change=20, random_state=42
        ),
        "Log-target ANN": MLPRegressor(
            hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
            alpha=0.001, learning_rate_init=0.001, max_iter=500,
            early_stopping=True, validation_fraction=0.10,
            n_iter_no_change=20, random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        )
    }

    results = []
    predictions = {}
    best_pipeline = None
    best_mae = float("inf")

    for name, estimator in models.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        if name == "Log-target ANN":
            pipeline.fit(X_train, np.log1p(y_train))
            pred = np.expm1(pipeline.predict(X_test))
        else:
            # Train the original-target ANN in lakh units for stable optimisation.
            if name == "Original-target ANN":
                pipeline.fit(X_train, y_train / 100000)
                pred = pipeline.predict(X_test) * 100000
            else:
                pipeline.fit(X_train, y_train)
                pred = pipeline.predict(X_test)

        mae = float(mean_absolute_error(y_test, pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        r2 = float(r2_score(y_test, pred))
        results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
        predictions[name] = pred
        print(f"{name}: MAE={mae:,.2f} | RMSE={rmse:,.2f} | R2={r2:.4f}")

        if name == "Original-target ANN":
            best_pipeline = pipeline
            best_mae = mae

    results_df = pd.DataFrame(results).sort_values("MAE")
    results_df.to_csv(ROOT / "model_comparison.csv", index=False)
    with open(ROOT / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    joblib.dump(best_pipeline, MODELS / "ann_pipeline.joblib")
    # Also save at project root for compatibility with the report's naming.
    joblib.dump(best_pipeline, ROOT / "ann_pipeline.joblib")

    # Price distribution chart.
    plt.figure(figsize=(8, 5))
    plt.hist(y / 100000, bins=35)
    plt.xlabel("Price (₹ lakh)")
    plt.ylabel("Number of properties")
    plt.title("Housing Price Distribution")
    plt.tight_layout()
    plt.savefig(FIGS / "price_distribution.png", dpi=150)
    plt.close()

    # Model MAE comparison.
    plt.figure(figsize=(9, 5))
    plt.bar(results_df["Model"], results_df["MAE"] / 100000)
    plt.ylabel("MAE (₹ lakh)")
    plt.title("Model Comparison - MAE")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGS / "model_mae_comparison.png", dpi=150)
    plt.close()

    # Actual vs predicted for the original-target ANN.
    plt.figure(figsize=(7, 6))
    ann_pred = predictions["Original-target ANN"]
    plt.scatter(y_test / 100000, ann_pred / 100000, alpha=.55)
    max_val = max(float(y_test.max()), float(np.max(ann_pred))) / 100000
    plt.plot([0, max_val], [0, max_val], linestyle="--")
    plt.xlabel("Actual price (₹ lakh)")
    plt.ylabel("Predicted price (₹ lakh)")
    plt.title("Original-target ANN: Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(FIGS / "ann_actual_vs_predicted.png", dpi=150)
    plt.close()

    print("\\nTraining complete.")
    print("Saved model: ann_pipeline.joblib")
    print("Saved results: model_comparison.csv and results.json")
    print("Saved charts in figs/")

if __name__ == "__main__":
    main()
