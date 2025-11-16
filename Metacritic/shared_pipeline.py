# shared_pipeline.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def build_preprocessor():
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

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, ["user_score", "release_year"]),
            ("cat", cat_transformer, ["platform"]),
            ("text", text_transformer, "text"),
        ],
        remainder="drop",
    )
