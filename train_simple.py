import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score

from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

print("🎯 TRAINING ALL MODELS")
print("=" * 60)

os.makedirs("models", exist_ok=True)

# =============== LOAD DATA ===============
df = pd.read_csv("data/prepared_data.csv")
books = pd.read_csv("data/books_prepared.csv")
users = pd.read_csv("data/users_prepared.csv")

# Ensure Like column
if "Like" not in df.columns:
    df["Like"] = (df["Book-Rating"] >= 7).astype(int)

# Parse Timestamp safely
if "Timestamp" in df.columns:
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

# =============== MODEL 1: RATING PREDICTOR ===============
features_reg = [
    "Age",
    "user_avg_rating",
    "user_consistency",
    "book_avg_rating",
    "book_popularity",
    "book_rating_std",
    "age_book_gap",
    "rating_difference",
]

X_reg = df[features_reg].fillna(0)
y_reg = df["Book-Rating"].astype(float)

# Faster + still good for demo
rating_model = RandomForestRegressor(
    n_estimators=50,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rating_model.fit(X_reg, y_reg)

pred_reg = rating_model.predict(X_reg)
mae = mean_absolute_error(y_reg, pred_reg)

joblib.dump(rating_model, "models/rating_predictor.joblib")

# =============== MODEL 2: LIKE/DISLIKE CLASSIFIER ===============
features_clf = [
    "Age",
    "user_avg_rating",
    "user_consistency",
    "book_avg_rating",
    "book_popularity",
    "age_book_gap",
    "popularity_ratio",
]

X_clf = df[features_clf].fillna(0)
y_clf = df["Like"].astype(int)

like_model = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
)
like_model.fit(X_clf, y_clf)
acc = like_model.score(X_clf, y_clf)

joblib.dump(like_model, "models/like_classifier.joblib")

# =============== MODEL 3: USER CLUSTERS ===============
user_features = ["user_avg_rating", "user_activity", "user_consistency", "Age"]
user_df = df[["User-ID"] + user_features].drop_duplicates()
X_user = user_df[user_features].fillna(0)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
user_df["cluster"] = kmeans.fit_predict(X_user)

joblib.dump(kmeans, "models/user_clusters.joblib")
user_df[["User-ID", "cluster"]].to_csv("models/cluster_mapping.csv", index=False)

# =============== MODEL 4: FORECASTING (REAL ML: SARIMAX) ===============
forecast_info = {"enabled": False}

if "Timestamp" in df.columns and df["Timestamp"].notna().any():
    ts = df.dropna(subset=["Timestamp"]).copy()
    ts["month"] = ts["Timestamp"].dt.to_period("M").dt.to_timestamp()

    monthly = ts.groupby("month").agg(
        avg_rating=("Book-Rating", "mean"),
        rating_count=("Book-Rating", "count")
    ).reset_index()

    # Need enough points to fit
    if len(monthly) >= 12:
        y = monthly["avg_rating"].astype(float)
        exog = monthly[["rating_count"]].astype(float)

        # SARIMAX: stable and explainable
        model = SARIMAX(
            y,
            exog=exog,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False)

        joblib.dump(res, "models/forecast_sarimax.joblib")
        monthly.to_csv("models/monthly_ratings.csv", index=False)

        forecast_info = {
            "enabled": True,
            "points": int(len(monthly)),
            "start": str(monthly["month"].min()),
            "end": str(monthly["month"].max()),
        }
    else:
        monthly.to_csv("models/monthly_ratings.csv", index=False)

# =============== MODEL 5: NLP (REAL ML: TFIDF + LinearSVC) ===============
nlp_info = {"enabled": False}

if {"Category", "Book-Title", "Book-Author"}.issubset(books.columns):
    books = books.copy()
    books["text"] = (
        books["Book-Title"].fillna("").astype(str)
        + " "
        + books["Book-Author"].fillna("").astype(str)
        + " "
        + books["Publisher"].fillna("").astype(str)
    )

    # Label encode categories
    le = LabelEncoder()
    y_cat = le.fit_transform(books["Category"].fillna("Unknown").astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        books["text"], y_cat, test_size=0.2, random_state=42, stratify=y_cat
    )

    nlp_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
        ("clf", LinearSVC())
    ])

    nlp_pipeline.fit(X_train, y_train)
    test_acc = nlp_pipeline.score(X_test, y_test)

    joblib.dump(nlp_pipeline, "models/nlp_pipeline.joblib")
    joblib.dump(le, "models/nlp_label_encoder.joblib")

    nlp_info = {
        "enabled": True,
        "classes": list(le.classes_),
        "test_accuracy": float(test_acc),
    }

# =============== SAVE METADATA ===============
meta = {
    "rating": {"features": features_reg, "train_mae": float(mae)},
    "like": {"features": features_clf, "train_acc": float(acc)},
    "clusters": {"features": user_features, "k": 3},
    "forecast": forecast_info,
    "nlp": nlp_info,
}

with open("models/meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print("\n✅ TRAINING DONE")
print(json.dumps(meta, indent=2))
print("\n📁 Models saved to /models")
