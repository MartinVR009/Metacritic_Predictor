import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def load_data():
    df = pd.read_csv("games.csv")

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["metascore"] = pd.to_numeric(df["metascore"], errors="coerce")
    df["user_score"] = pd.to_numeric(df["user_score"], errors="coerce")

    df = df.dropna(subset=["metascore"]).copy()
    df["release_year"] = df["release_date"].dt.year.fillna(-1).astype(int)
    df["text"] = df["title"].fillna("") + " " + df["summary"].fillna("")

    X = df[["user_score", "release_year", "platform", "text"]]
    y = df["metascore"]

    return X, y, df


def plot_pred_vs_actual(y_test, y_pred, outdir):
    plt.figure(figsize=(7,7))
    plt.scatter(y_test, y_pred, alpha=0.4)
    plt.plot([0,100], [0,100], "r--")
    plt.xlabel("Actual Metascore")
    plt.ylabel("Predicted Metascore")
    plt.title("Predicted vs Actual")
    plt.savefig(outdir / "pred_vs_actual.png")
    plt.close()


def plot_error_distribution(errors, outdir):
    plt.figure(figsize=(7,5))
    plt.hist(errors, bins=40, alpha=0.7)
    plt.title("Prediction Error Distribution")
    plt.xlabel("Error (Prediction - Actual)")
    plt.ylabel("Frequency")
    plt.savefig(outdir / "error_distribution.png")
    plt.close()


def plot_top_features(model, preprocessor, outdir, top_n=20):
    try:
        model_obj = model.named_steps["model"]
        pre = model.named_steps["pre"]
        vectorizer = pre.named_transformers_["text"].named_steps["tfidf"]

        feature_names = (
            ["user_score", "release_year"] +
            list(model.named_steps["pre"].named_transformers_["platform"].named_steps["encoder"].get_feature_names_out(["platform"])) +
            list(vectorizer.get_feature_names_out())
        )

        # Ridge → coef_
        if hasattr(model_obj, "coef_"):
            importances = model_obj.coef_[0] if model_obj.coef_.ndim > 1 else model_obj.coef_

        # Tree models → feature_importances_
        elif hasattr(model_obj, "feature_importances_"):
            importances = model_obj.feature_importances_

        else:
            print("No feature importances available.")
            return

        # Get top features
        idx = np.argsort(importances)[-top_n:]
        top_features = np.array(feature_names)[idx]
        top_values = importances[idx]

        plt.figure(figsize=(8, 10))
        plt.barh(top_features, top_values)
        plt.title(f"Top {top_n} Feature Importances")
        plt.tight_layout()
        plt.savefig(outdir / "top_features.png")
        plt.close()

    except Exception as e:
        print("Feature importance error:", e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="which model to load (.joblib)")
    args = parser.parse_args()

    model_path = Path(args.model)
    model = joblib.load(model_path)

    print(f"Loaded model: {model_path}")

    X, y, df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.40, random_state=42
    )

    y_pred = model.predict(X_test)
    errors = y_pred - y_test

    outdir = Path("model_explanations") / model_path.stem
    outdir.mkdir(parents=True, exist_ok=True)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    with open(outdir / "metrics.txt", "w") as f:
        f.write(f"MAE: {mae}\nRMSE: {rmse}\nR2: {r2}\n")

    print("Generating plots...")
    plot_pred_vs_actual(y_test, y_pred, outdir)
    plot_error_distribution(errors, outdir)
    plot_top_features(model, model.named_steps["pre"], outdir)

    print(f"All graphs saved to: {outdir}")


if __name__ == "__main__":
    main()
