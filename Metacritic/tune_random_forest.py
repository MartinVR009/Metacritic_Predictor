import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import numpy as np


def load_dataset():
    df = pd.read_csv("games.csv")
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["metascore"] = pd.to_numeric(df["metascore"], errors="coerce")
    df["user_score"] = pd.to_numeric(df["user_score"], errors="coerce")

    df2 = df.dropna(subset=["metascore"]).copy()
    df2["release_year"] = df2["release_date"].dt.year.fillna(-1).astype(int)
    df2["text"] = df2["title"].fillna("") + " \n " + df2["summary"].fillna("")

    X = df2[["user_score", "release_year", "platform", "text"]]
    y = df2["metascore"].astype(float)
    return train_test_split(X, y, test_size=0.4, random_state=42)


def build_pipeline():
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="<MISSING>")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    text_transformer = TfidfVectorizer(
        max_features=3000,
        stop_words="english"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, ["user_score", "release_year"]),
            ("cat", cat_transformer, ["platform"]),
            ("text", text_transformer, "text"),
        ]
    )

    model = Pipeline([
        ("pre", preprocessor),
        ("rf", RandomForestRegressor(random_state=42, n_jobs=-1))
    ])

    return model


def main():
    print("🔍 Loading dataset...")
    X_train, X_test, y_train, y_test = load_dataset()

    print("🔧 Building pipeline...")
    pipeline = build_pipeline()

    print("\n🎯 Running refined Random Forest hyperparameter tuning...\n")

    param_dist = {
        "rf__n_estimators": [150, 200, 300, 350, 400],
        "rf__max_depth": [None, 60, 80, 100, 120],
        "rf__min_samples_split": [2],
        "rf__min_samples_leaf": [1],
        "rf__max_features": [None, "sqrt"],  # None = all features
        "rf__bootstrap": [True],
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=12,             # small but meaningful search
        cv=3,
        verbose=2,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        random_state=42
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_params = search.best_params_

    print("\nBest parameters found:")
    print(json.dumps(best_params, indent=2))

    print("\nEvaluating refined RF model...")
    y_pred = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(((y_test - y_pred) ** 2).mean())
    r2 = r2_score(y_test, y_pred)

    print("\n===== TUNED RANDOM FOREST RESULTS =====")
    print(f"MAE : {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²  : {r2:.3f}")

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    save_path = models_dir / "rf_refined.joblib"

    joblib.dump(best_model, save_path)
    print(f"\nSaved refined model to: {save_path}")

    results_df = pd.DataFrame({
        "mae": [mae],
        "rmse": [rmse],
        "r2": [r2],
        "params": [json.dumps(best_params)],
    })
    results_df.to_csv("rf_refined_results.csv", index=False)
    print("Saved results to rf_refined_results.csv")


if __name__ == "__main__":
    main()
