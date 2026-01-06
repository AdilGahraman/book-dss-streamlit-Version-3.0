import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODELS_DIR = Path("models")

class ModelStore:
    def __init__(self):
        self.meta = {}
        self.rating_model = None
        self.like_model = None
        self.kmeans = None
        self.cluster_map = None
        self.forecast_res = None
        self.monthly = None
        self.nlp_pipeline = None
        self.nlp_le = None

    def load(self):
        meta_path = MODELS_DIR / "meta.json"
        if meta_path.exists():
            self.meta = json.loads(meta_path.read_text(encoding="utf-8"))

        self.rating_model = joblib.load(MODELS_DIR / "rating_predictor.joblib")
        self.like_model = joblib.load(MODELS_DIR / "like_classifier.joblib")
        self.kmeans = joblib.load(MODELS_DIR / "user_clusters.joblib")

        cm = MODELS_DIR / "cluster_mapping.csv"
        if cm.exists():
            self.cluster_map = pd.read_csv(cm)

        mr = MODELS_DIR / "monthly_ratings.csv"
        if mr.exists():
            self.monthly = pd.read_csv(mr)

        fs = MODELS_DIR / "forecast_sarimax.joblib"
        if fs.exists():
            self.forecast_res = joblib.load(fs)

        nlp_p = MODELS_DIR / "nlp_pipeline.joblib"
        nlp_le = MODELS_DIR / "nlp_label_encoder.joblib"
        if nlp_p.exists() and nlp_le.exists():
            self.nlp_pipeline = joblib.load(nlp_p)
            self.nlp_le = joblib.load(nlp_le)

    def predict_rating(self, payload: dict):
        features = self.meta["rating"]["features"]
        X = pd.DataFrame([{k: payload.get(k, 0) for k in features}]).fillna(0)
        pred = float(self.rating_model.predict(X)[0])

        # explanation: feature importance (global)
        importances = getattr(self.rating_model, "feature_importances_", None)
        top = []
        if importances is not None:
            pairs = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)[:5]
            top = [{"feature": f, "importance": float(w)} for f, w in pairs]

        return {"prediction": pred, "top_features": top}

    def predict_like(self, payload: dict):
        features = self.meta["like"]["features"]
        X = pd.DataFrame([{k: payload.get(k, 0) for k in features}]).fillna(0)

        pred = int(self.like_model.predict(X)[0])
        proba = None
        if hasattr(self.like_model, "predict_proba"):
            proba = self.like_model.predict_proba(X)[0].tolist()

        importances = getattr(self.like_model, "feature_importances_", None)
        top = []
        if importances is not None:
            pairs = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)[:5]
            top = [{"feature": f, "importance": float(w)} for f, w in pairs]

        return {"prediction": pred, "proba": proba, "top_features": top}

    def predict_nlp(self, text: str):
        if self.nlp_pipeline is None or self.nlp_le is None:
            return {"error": "NLP model not trained"}

        pred_idx = int(self.nlp_pipeline.predict([text])[0])
        label = self.nlp_le.inverse_transform([pred_idx])[0]

        # explanation: show top TFIDF terms (approx)
        # (We can’t easily explain LinearSVC without extra work; this is good enough for demo)
        tfidf = self.nlp_pipeline.named_steps["tfidf"]
        vec = tfidf.transform([text])
        # top terms in input text
        if vec.nnz > 0:
            top_ids = vec.toarray()[0].argsort()[-8:][::-1]
            terms = tfidf.get_feature_names_out()
            top_terms = [terms[i] for i in top_ids if vec.toarray()[0][i] > 0][:8]
        else:
            top_terms = []

        return {"category": str(label), "keywords": top_terms}

    def forecast(self, months_ahead: int, future_rating_count=None):
        if self.forecast_res is None or self.monthly is None:
            return {"error": "Forecast model not trained or insufficient time-series data."}

        months_ahead = int(max(1, min(months_ahead, 24)))

        # exog for future: either user provided or use last known count
        last_count = float(self.monthly["rating_count"].iloc[-1])
        if future_rating_count is None:
            exog_f = np.array([[last_count]] * months_ahead, dtype=float)
        else:
            vals = [float(x) for x in future_rating_count][:months_ahead]
            if len(vals) < months_ahead:
                vals += [last_count] * (months_ahead - len(vals))
            exog_f = np.array(vals, dtype=float).reshape(-1, 1)

        fc = self.forecast_res.get_forecast(steps=months_ahead, exog=exog_f)
        mean = fc.predicted_mean.astype(float).tolist()
        conf = fc.conf_int().astype(float).values.tolist()  # [[low, high], ...]

        return {"months_ahead": months_ahead, "mean": mean, "conf_int": conf, "assumed_counts": exog_f.flatten().tolist()}
    
    def predict_cluster(self, payload: dict):
        features = self.meta["clusters"]["features"]

        X = pd.DataFrame([[
            float(payload[f]) for f in features
        ]], columns=features)

        cluster = int(self.kmeans.predict(X)[0])

        return {
            "cluster": cluster,
            "description": {
                0: "Enthusiastic Readers (high engagement)",
                1: "Critical Readers (selective)",
                2: "Casual Readers (low activity)"
            }.get(cluster, "Unknown")
        }

