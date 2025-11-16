import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "models" / "random_forest.joblib"
if not MODEL_PATH.exists():
    print("Model file not found. Run train_and_save.py first")
    raise SystemExit(1)

model = joblib.load(MODEL_PATH)

sample = {"user_score": 8.0, "release_year": 2022, "platform": "PC", "text": "Example Game\nA fun action adventure"}
df = pd.DataFrame([sample])
pred = model.predict(df)
print("Predicted metascore:", float(pred[0]))
