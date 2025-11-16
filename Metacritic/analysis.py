import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = Path(__file__).parent / "games.csv"
    if not p.exists():
        print(f"Cannot find {p}")
        sys.exit(1)

    df = pd.read_csv(p)

    # Basic cleaning
    df["release_date"] = pd.to_datetime(df.get("release_date"), errors="coerce")
    df["metascore"] = pd.to_numeric(df.get("metascore"), errors="coerce")
    df["user_score"] = pd.to_numeric(df.get("user_score"), errors="coerce")

    # How many rows and how many non-null targets
    n_total = len(df)
    n_with_target = df["metascore"].notna().sum()

    print("Dataset overview")
    print("-----------------")
    print(f"Total rows: {n_total}")
    print(f"Rows with metascore (usable for supervised learning): {n_with_target}")
    print(f"Fraction usable: {n_with_target / max(1, n_total):.3f}")
    print()

    # Missing value summary for candidate features
    print("Missingness (selected columns)")
    print(df[["platform", "release_date", "user_score", "summary", "title"]].isna().mean().sort_values())
    print()

    # Target distribution
    print("Metascore distribution (describe)")
    print(df["metascore"].describe())
    print()

    # Platforms and counts
    print("Top platforms by count (platform : count)")
    print(df["platform"].value_counts().head(20))
    print()

    # Correlation with user_score (if present)
    if df["user_score"].notna().sum() > 0:
        corr = df[["metascore", "user_score"]].corr().iloc[0, 1]
        print(f"Pearson correlation metascore <-> user_score: {corr:.3f}")
    else:
        print("No user_score values to compute correlation.")
    print()

    # Quick rule of thumb about dataset size
    n_samples = int(n_with_target)
    if n_samples < 200:
        print("Small dataset (<200). Expect high variance; complex models will overfit.")
    elif n_samples < 1000:
        print("Modest dataset (200-1000). Simple models or strong regularization recommended.")
    else:
        print("Decent dataset (>=1000). You can try moderately complex models and include text features.")
    print()

    # If enough samples, run a baseline model (text + numeric)
    if n_samples < 50:
        print("Too few samples to train a meaningful model; skipping baseline training.")
        return

    # Train a small baseline model using scikit-learn
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.impute import SimpleImputer
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import FeatureUnion, Pipeline
        from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
        from sklearn.compose import ColumnTransformer
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    except Exception as e:
        print("scikit-learn not available or failed to import:", e)
        print("To run baseline training, install scikit-learn (pip install scikit-learn)")
        return

    # Prepare features
    df2 = df.dropna(subset=["metascore"]).copy()
    df2["release_year"] = df2["release_date"].dt.year.fillna(-1).astype(int)
    df2["text"] = (df2.get("title", "").fillna("") + " \n " + df2.get("summary", "").fillna(""))

    features = ["user_score", "release_year", "platform", "text"]
    X = df2[features]
    y = df2["metascore"].astype(float)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessors
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="<MISSING>")),
        # Avoid passing `sparse`/`sparse_output` to keep compatibility across sklearn versions
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    text_transformer = TfidfVectorizer(max_features=2000, stop_words="english")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, ["user_score", "release_year"]),
            ("cat", cat_transformer, ["platform"]),
            ("text", text_transformer, "text"),
        ],
        remainder="drop",
    )

    model = Pipeline([
        ("pre", preprocessor),
        ("rf", RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)),
    ])

    print("Training baseline RandomForestRegressor (user_score + platform + TF-IDF(text))...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("Baseline results on held-out test set:")
    print(f"  MAE:  {mae:.3f}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  R2:   {r2:.3f}")
    print()

    # Compare to simple baseline: predicting train mean
    mean_pred = y_train.mean()
    mae_mean = mean_absolute_error(y_test, np.full_like(y_test, mean_pred, dtype=float))
    print(f"Mean-predictor MAE for comparison: {mae_mean:.3f}")

    # Show a few examples
    n_show = min(8, len(y_test))
    print("Examples (true -> pred) -- first rows of test set")
    for t, pval in list(zip(y_test.values[:n_show], y_pred[:n_show])):
        print(f"  {t:.1f} -> {pval:.2f}")


if __name__ == "__main__":
    main()
