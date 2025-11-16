import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def load_data(csv_path: Path):
    df = pd.read_csv(csv_path)

    df["release_date"] = pd.to_datetime(df.get("release_date"), errors="coerce")
    df["metascore"] = pd.to_numeric(df.get("metascore"), errors="coerce")
    df["user_score"] = pd.to_numeric(df.get("user_score"), errors="coerce")

    df = df.dropna(subset=["metascore"]).copy()

    df["release_year"] = df["release_date"].dt.year.fillna(-1).astype(int)
    df["text"] = (
        df.get("title", "").fillna("") +
        " \n " +
        df.get("summary", "").fillna("")
    )

    X = df[["user_score", "release_year", "platform", "text"]]
    y = df["metascore"].astype(float)

    return X, y, df


def evaluate(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"\n📦 Loading model: {model_path.name}")
    model = joblib.load(model_path)

    csv_path = Path(__file__).parent / "games.csv"
    X, y, df_full = load_data(csv_path)

    print("📊 Splitting dataset: 60% train / 40% test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.40, random_state=42
    )

    print("🔍 Running predictions...")
    y_pred = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n===== RESULTS =====")
    print(f"MAE :  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²  : {r2:.3f}")

    # Save result file
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    output_path = results_dir / f"{model_path.stem}_results.csv"
    results = pd.DataFrame({
        "true_metascore": y_test.values,
        "predicted_metascore": y_pred,
    })
    results.to_csv(output_path, index=False)

    print(f"\n📁 Results saved to: {output_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument(
        "--model",
        required=True,
        help="Model filename inside the models/ folder, e.g. ridge_model.joblib",
    )
    args = parser.parse_args()

    model_path = Path(__file__).parent / "models" / args.model
    evaluate(model_path)


if __name__ == "__main__":
    main()
