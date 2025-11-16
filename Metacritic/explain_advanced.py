import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from pathlib import Path
import argparse
from scipy.sparse import issparse


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


def make_shap_plots(model, X_sample, outdir):
    try:
        print("Transforming input data for SHAP...")
        pre = model.named_steps["pre"]
        model_obj = model.named_steps["model"]

        X_trans = pre.transform(X_sample)

        # Convert sparse → dense for SHAP if needed
        if issparse(X_trans):
            X_trans_dense = X_trans.toarray()
        else:
            X_trans_dense = X_trans

        feature_names = pre.get_feature_names_out()

        print("▶ Building TreeExplainer...")
        explainer = shap.TreeExplainer(model_obj)
        shap_values = explainer.shap_values(X_trans_dense)

        # SHAP summary
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_trans_dense, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(outdir / "shap_summary.png")
        plt.close()

        # SHAP beeswarm
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_trans_dense, feature_names=feature_names,
                          plot_type="beeswarm", show=False)
        plt.tight_layout()
        plt.savefig(outdir / "shap_beeswarm.png")
        plt.close()

        # Single prediction force plot (dense only)
        force_fig = shap.force_plot(
            explainer.expected_value,
            shap_values[0],
            feature_names=feature_names,
            matplotlib=True
        )
        plt.savefig(outdir / "shap_force.png")
        plt.close()

        return shap_values, feature_names

    except Exception as e:
        print("SHAP FAILED:", e)
        return None, None


def build_wordclouds(shap_values, feature_names, outdir):
    if shap_values is None:
        print("Skipping wordclouds (no SHAP values)")
        return

    shap_mean = np.mean(shap_values, axis=0)

    # Choose only text/TFiDF features
    tfidf_features = [name for name in feature_names if "tfidf" in name]

    pos_words = {}
    neg_words = {}

    for name, val in zip(feature_names, shap_mean):
        if "tfidf" not in name:
            continue
        word = name.replace("pre__text__tfidf__", "")
        if val > 0:
            pos_words[word] = val
        elif val < 0:
            neg_words[word] = abs(val)

    wc_pos = WordCloud(width=800, height=600, background_color="white")
    wc_neg = WordCloud(width=800, height=600, background_color="black")

    if pos_words:
        plt.figure(figsize=(10, 6))
        plt.imshow(wc_pos.generate_from_frequencies(pos_words))
        plt.axis("off")
        plt.title("Words Increasing Score")
        plt.savefig(outdir / "wordcloud_positive.png")
        plt.close()

    if neg_words:
        plt.figure(figsize=(10, 6))
        plt.imshow(wc_neg.generate_from_frequencies(neg_words))
        plt.axis("off")
        plt.title("Words Decreasing Score")
        plt.savefig(outdir / "wordcloud_negative.png")
        plt.close()


def correlation_heatmap(df, outdir):
    corr = df[["user_score", "release_year", "metascore"]].corr()

    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.savefig(outdir / "feature_correlation_heatmap.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    model_path = Path(args.model)
    model = joblib.load(model_path)

    print(f"Loaded model: {model_path.name}")

    X, y, df = load_data()

    # Smaller sample to keep SHAP fast
    X_sample = X.sample(200, random_state=42).reset_index(drop=True)

    outdir = Path("model_explanations") / model_path.stem
    outdir.mkdir(parents=True, exist_ok=True)

    print("Generating SHAP plots...")
    shap_values, feature_names = make_shap_plots(model, X_sample, outdir)

    print("Generating word clouds...")
    build_wordclouds(shap_values, feature_names, outdir)

    print("Generating correlation heatmap...")
    correlation_heatmap(df, outdir)

    print(f"All plots saved in {outdir}")


if __name__ == "__main__":
    main()
