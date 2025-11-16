# train_and_save.py
import pandas as pd
import joblib
from pathlib import Path
import argparse

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from shared_pipeline import build_preprocessor


def load_data(csv_path):
    df = pd.read_csv(csv_path)

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["metascore"] = pd.to_numeric(df["metascore"], errors="coerce")
    df["user_score"] = pd.to_numeric(df["user_score"], errors="coerce")

    df = df.dropna(subset=["metascore"]).copy()
    df["release_year"] = df["release_date"].dt.year.fillna(-1).astype(int)
    df["text"] = df["title"].fillna("") + " " + df["summary"].fillna("")

    X = df[["user_score", "release_year", "platform", "text"]]
    y = df["metascore"]

    return X, y


def build_model(model_name):
    pre = build_preprocessor()

    if model_name == "ridge":
        model = Ridge(alpha=1.0)

    elif model_name == "rf":
        model = RandomForestRegressor(
            n_estimators=120,
            n_jobs=-1,
            random_state=42
        )

    elif model_name == "xgb":
        from xgboost import XGBRegressor
        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
        )

    elif model_name == "lgbm":
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline([("pre", pre), ("model", model)])

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="ridge | random_forest | xgboost | lightgbm")
    args = parser.parse_args()

    X, y = load_data("games.csv")
    model = build_model(args.model)

    print(f"Training model: {args.model}")
    model.fit(X, y)

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    path = models_dir / f"{args.model}.joblib"
    joblib.dump(model, path)
    print(f"Model saved to {path}")


if __name__ == "__main__":
    main()
