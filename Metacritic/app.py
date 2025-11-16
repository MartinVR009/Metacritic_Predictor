from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import joblib
import pandas as pd


class GameInfo(BaseModel):
    title: Optional[str] = None
    summary: Optional[str] = None
    platform: Optional[str] = None
    user_score: Optional[float] = None
    release_date: Optional[str] = None  # ISO date or any parsable by pandas


app = FastAPI(title="Metascore Predictor")

# Load model at startup
MODEL_PATH = Path(__file__).parent / "models" / "model_rf.joblib"
model = None


@app.on_event("startup")
def load_model():
    global model
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError
        model = joblib.load(MODEL_PATH)
        return
    except Exception as e:
        # Log the error and fall back to training a quick pipeline
        print(f"Warning: failed to load model from {MODEL_PATH}: {e}")
        print("Training a lightweight fallback model at startup (this may take a few seconds)...")

        # Train a minimal pipeline (fast Ridge) so the app can serve predictions
        try:
            from sklearn.linear_model import Ridge
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import OneHotEncoder, StandardScaler
            from sklearn.compose import ColumnTransformer
        except Exception as imp_err:
            raise RuntimeError("scikit-learn is required to build fallback model: " + str(imp_err))

        import pandas as pd
        df = pd.read_csv(Path(__file__).parent / "games.csv")
        df["release_date"] = pd.to_datetime(df.get("release_date"), errors="coerce")
        df["metascore"] = pd.to_numeric(df.get("metascore"), errors="coerce")
        df["user_score"] = pd.to_numeric(df.get("user_score"), errors="coerce")
        df2 = df.dropna(subset=["metascore"]).copy()
        df2["release_year"] = df2["release_date"].dt.year.fillna(-1).astype(int)
        df2["text"] = (df2.get("title", "").fillna("") + " \n " + df2.get("summary", "").fillna(""))

        X = df2[["user_score", "release_year", "platform", "text"]]
        y = df2["metascore"].astype(float)

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

        # Train on a small sample for speed (or full data if you prefer)
        sample = X.sample(n=min(2000, len(X)), random_state=42)
        y_sample = y.loc[sample.index]
        model.fit(sample, y_sample)
        print("Fallback model trained and ready.")
        # Do not overwrite saved model automatically
        return


@app.post("/predict")
def predict(game: GameInfo):
    # Build dataframe with the expected pipeline inputs
    title = game.title or ""
    summary = game.summary or ""
    text = title + " \n " + summary

    # derive release_year
    release_year = None
    if game.release_date:
        try:
            release_year = pd.to_datetime(game.release_date, errors="coerce").year
        except Exception:
            release_year = None

    df = pd.DataFrame([{"user_score": game.user_score, "release_year": release_year if release_year is not None else -1,
                        "platform": game.platform, "text": text}])

    preds = model.predict(df)
    # Return single float
    return {"predicted_metascore": float(preds[0])}
