"""Retrain a fast, compatible model and save it to `models/model.joblib`.

This script uses a Ridge regressor with the same preprocessing pipeline to
avoid long training times and to ensure compatibility with the installed
scikit-learn version.
"""
from pathlib import Path
import joblib
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


def build_pipeline():
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="<MISSING>")),
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
        ("ridge", Ridge(alpha=1.0)),
    ])

    return model


def main():
    root = Path(__file__).parent
    csv_path = root / "games.csv"
    if not csv_path.exists():
        raise SystemExit(f"Cannot find {csv_path}")

    df = pd.read_csv(csv_path)
    df["release_date"] = pd.to_datetime(df.get("release_date"), errors="coerce")
    df["metascore"] = pd.to_numeric(df.get("metascore"), errors="coerce")
    df["user_score"] = pd.to_numeric(df.get("user_score"), errors="coerce")

    df2 = df.dropna(subset=["metascore"]).copy()
    df2["release_year"] = df2["release_date"].dt.year.fillna(-1).astype(int)
    df2["text"] = (df2.get("title", "").fillna("") + " \n " + df2.get("summary", "").fillna(""))

    X = df2[["user_score", "release_year", "platform", "text"]]
    y = df2["metascore"].astype(float)

    pipeline = build_pipeline()
    print("Training (fast Ridge) on full dataset...")
    pipeline.fit(X, y)

    model_path = root / "models" / "model.joblib"
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Wrote compatible model to {model_path}")


if __name__ == "__main__":
    main()
