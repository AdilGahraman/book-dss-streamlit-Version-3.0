import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="📚 Book DSS", page_icon="📚", layout="wide")

# ---------- Premium CSS ----------
st.markdown("""
<style>
/* Page background */
.stApp {
    background-color: #ffffff;
    color: #111827;
}

/* Main container */
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}

/* Headings */
h1, h2, h3 {
    color: #111827;
    letter-spacing: -0.02em;
}

/* Small helper text */
.small-muted {
    color: #6b7280;
    font-size: 0.9rem;
}

/* Cards */
.card {
    background: #ffffff;
    padding: 16px 16px;
    border-radius: 14px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 4px 12px rgba(0,0,0,0.04);
}

/* KPI numbers */
.kpi {
    font-size: 2.0rem;
    font-weight: 800;
    margin: 0;
    color: #111827;
}

.kpi-label {
    font-size: 0.95rem;
    color: #374151;
    margin: 0;
}

/* Divider */
.hr {
    height: 1px;
    background: #e5e7eb;
    margin: 12px 0;
}
</style>
""", unsafe_allow_html=True)


# ---------- Feature descriptions (one-liners) ----------
FEATURE_DESCRIPTIONS = {
    # User-related
    "Age": "Age of the user in years.",
    "user_avg_rating": "Average rating the user has given historically (user preference baseline).",
    "user_activity": "How many books the user has rated (engagement level).",
    "user_consistency": "How stable the user's ratings are over time (lower = more consistent).",

    # Book-related
    "book_avg_rating": "Average rating the book received from all users (overall quality signal).",
    "book_popularity": "Number of ratings the book received (popularity proxy).",
    "book_rating_std": "How much users disagree about the book (higher = more mixed opinions).",

    # Engineered interaction features
    "age_book_gap": "Difference between user's age and book’s typical audience (proxy feature).",
    "rating_difference": "User’s average rating minus book’s average rating (preference alignment).",
    "popularity_ratio": "Book popularity relative to user’s typical exposure (relative popularity).",

    # Forecasting
    "months_ahead": "How many future months you want to forecast.",
    "nlp_text": "Text input (title/description) used to predict a book category."
}

def feature_help(key: str):
    """Small helper to show descriptions consistently."""
    st.caption(FEATURE_DESCRIPTIONS.get(key, ""))

# ---------- Cached API wrappers ----------
@st.cache_data(ttl=5)
def api_get(path: str):
    return requests.get(f"{API_URL}{path}", timeout=5).json()

@st.cache_data(ttl=2)
def api_post(path: str, payload: dict):
    # keep timeout 8 as you had; raise if you want 15
    return requests.post(f"{API_URL}{path}", json=payload, timeout=8).json()

# ---------- Cached dataset loaders for EDA ----------
@st.cache_data
def load_local_data():
    df = pd.read_csv("data/prepared_data.csv")
    books = pd.read_csv("data/books_prepared.csv")
    users = pd.read_csv("data/users_prepared.csv")

    # Safe timestamp parsing for EDA
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    return df, books, users

# ---------- Sidebar ----------
with st.sidebar:
    st.title("📚 Book DSS")
    st.markdown('<div class="small-muted">FastAPI + Streamlit hybrid</div>', unsafe_allow_html=True)

    tab = st.radio(
        "Navigation",
        [
            "🧩 Project Overview",
            "📈 Rating",
            "🎯 Like",
            "👥 Clusters",
            "📅 Forecast",
            "🔤 NLP",
            "📊 EDA",
            "🧠 Status"
        ]
    )

    st.markdown("---")
    st.caption("API")
    if st.button("Ping API"):
        try:
            r = api_get("/health")
            st.success("API OK ✅" if r.get("ok") else "API issue")
        except Exception as e:
            st.error(f"API error: {e}")

# ---------- Header ----------
st.markdown("## 📚 Book Recommendation Intelligent Decision Support System")
st.markdown(
    '<div class="small-muted">'
    'Five Machine Learning Models integrated into a single IDSS for explainable, data-driven decisions'
    '</div>',
    unsafe_allow_html=True)
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# ---------- Session state helpers ----------
def set_default(key, val):
    if key not in st.session_state:
        st.session_state[key] = val

# Rating inputs
set_default("r_age", 30)
set_default("r_user_avg", 7.0)
set_default("r_user_cons", 2.0)
set_default("r_book_avg", 7.5)
set_default("r_pop", 100.0)
set_default("r_book_std", 3.0)
set_default("r_age_gap", 0.0)
set_default("r_diff", 0.0)
set_default("rating_result", None)

# Like inputs
set_default("l_age", 30)
set_default("l_user_avg", 7.0)
set_default("l_user_cons", 2.0)
set_default("l_book_avg", 7.5)
set_default("l_pop", 100.0)
set_default("l_age_gap", 0.0)
set_default("l_pop_ratio", 1.0)
set_default("like_result", None)

# Cluster inputs
set_default("c_avg", 7.0)
set_default("c_act", 10.0)
set_default("c_cons", 2.0)
set_default("c_age", 30)

# Forecast inputs
set_default("months_ahead", 6)
set_default("forecast_result", None)

# NLP
set_default("nlp_text", "fantasy novel about wizards and a magic school")
set_default("nlp_result", None)

# ---------- Prediction functions ----------
def predict_rating():
    payload = {
        "Age": float(st.session_state.r_age),
        "user_avg_rating": float(st.session_state.r_user_avg),
        "user_consistency": float(st.session_state.r_user_cons),
        "book_avg_rating": float(st.session_state.r_book_avg),
        "book_popularity": float(st.session_state.r_pop),
        "book_rating_std": float(st.session_state.r_book_std),
        "age_book_gap": float(st.session_state.r_age_gap),
        "rating_difference": float(st.session_state.r_diff),
    }
    st.session_state.rating_result = api_post("/predict/rating", payload)

def predict_like():
    payload = {
        "Age": float(st.session_state.l_age),
        "user_avg_rating": float(st.session_state.l_user_avg),
        "user_consistency": float(st.session_state.l_user_cons),
        "book_avg_rating": float(st.session_state.l_book_avg),
        "book_popularity": float(st.session_state.l_pop),
        "age_book_gap": float(st.session_state.l_age_gap),
        "popularity_ratio": float(st.session_state.l_pop_ratio),
    }
    st.session_state.like_result = api_post("/predict/like", payload)

def predict_cluster():
    payload = {
        "user_avg_rating": float(st.session_state.c_avg),
        "user_activity": float(st.session_state.c_act),
        "user_consistency": float(st.session_state.c_cons),
        "Age": float(st.session_state.c_age),
    }
    return api_post("/predict/cluster", payload)

def predict_forecast():
    payload = {"months_ahead": int(st.session_state.months_ahead)}
    st.session_state.forecast_result = api_post("/forecast", payload)

def predict_nlp():
    payload = {"text": st.session_state.nlp_text}
    st.session_state.nlp_result = api_post("/predict/nlp", payload)

# ---------- Tabs ----------
# ======================================================
# 🧩 PROJECT OVERVIEW PAGE (NEW – DSS DESIGN)
# ======================================================
if tab == "🧩 Project Overview":

    st.subheader("🧩 Project Overview")
    st.info(
        "This Intelligent Decision Support System (IDSS) integrates **five machine learning models** "
        "to transform raw book-rating data into **actionable, explainable decisions**. "
        "Each model addresses a different decision-making task and contributes to the overall system intelligence."
    )

    st.markdown("### 🎯 Modeling Categories Overview")
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # -------- MODEL 1 --------
    st.markdown("## 1️⃣ Predictive Modeling — Rating Prediction")
    st.markdown("""
    <div class="card">
    <b>Goal:</b> Predict how a user is likely to rate a specific book.<br><br>

    <b>Technique:</b> Random Forest Regression (ensemble of decision trees).<br><br>

    <b>Input Features:</b>
    <ul>
      <li>User age</li>
      <li>User average rating</li>
      <li>User rating consistency</li>
      <li>Book average rating</li>
      <li>Book popularity</li>
      <li>Book rating standard deviation</li>
      <li>Age–book gap</li>
      <li>User–book rating difference</li>
    </ul>

    <b>IDSS Logic:</b>
    <ul>
      <li><span class="badge ok">Prediction Error ≤ 1.0</span> High confidence recommendation</li>
      <li><span class="badge warn">Prediction Error 1.0–1.5</span> Moderate confidence</li>
      <li><span class="badge info">Prediction Error > 1.5</span> Use with caution</li>
    </ul>

    <b>Output:</b> Numerical rating prediction (0–10) with feature importance explanation.
    </div>
    """, unsafe_allow_html=True)

    # -------- MODEL 2 --------
    st.markdown("## 2️⃣ Classification Modeling — Like / Dislike")
    st.markdown("""
    <div class="card">
    <b>Goal:</b> Classify whether a user will like or dislike a book.<br><br>

    <b>Technique:</b> Random Forest Classifier (binary classification).<br><br>

    <b>Input Features:</b>
    <ul>
      <li>User age</li>
      <li>User average rating</li>
      <li>User consistency</li>
      <li>Book average rating</li>
      <li>Book popularity</li>
      <li>Age–book gap</li>
      <li>Popularity ratio</li>
    </ul>

    <b>IDSS Logic:</b>
    <ul>
      <li><span class="badge ok">Probability ≥ 70%</span> Strong positive preference</li>
      <li><span class="badge warn">Probability 50–70%</span> Uncertain preference</li>
      <li><span class="badge info">Probability < 50%</span> Likely dislike</li>
    </ul>

    <b>Output:</b> Like / Dislike decision with confidence score.
    </div>
    """, unsafe_allow_html=True)

    # -------- MODEL 3 --------
    st.markdown("## 3️⃣ Clustering Modeling — User Segmentation")
    st.markdown("""
    <div class="card">
    <b>Goal:</b> Segment users into behavioral groups for targeted strategies.<br><br>

    <b>Technique:</b> K-Means Clustering (unsupervised learning).<br><br>

    <b>Input Features:</b>
    <ul>
      <li>User average rating</li>
      <li>User activity level</li>
      <li>User rating consistency</li>
      <li>User age</li>
    </ul>

    <b>IDSS Logic:</b>
    <ul>
      <li><span class="badge ok">Distinct cluster separation</span> Strong segmentation</li>
      <li><span class="badge warn">Moderate overlap</span> Acceptable segmentation</li>
      <li><span class="badge info">High overlap</span> Weak segmentation</li>
    </ul>

    <b>Output:</b> Cluster label with behavioral description (e.g., Enthusiastic, Critical, Casual readers).
    </div>
    """, unsafe_allow_html=True)

    # -------- MODEL 4 --------
    st.markdown("## 4️⃣ Forecasting Modeling — Rating Trends")
    st.markdown("""
    <div class="card">
    <b>Goal:</b> Forecast future trends in average book ratings over time.<br><br>

    <b>Technique:</b> SARIMAX Time-Series Forecasting.<br><br>

    <b>Input Features:</b>
    <ul>
      <li>Historical monthly average ratings</li>
      <li>Temporal trends and seasonality</li>
    </ul>

    <b>IDSS Logic:</b>
    <ul>
      <li><span class="badge ok">Narrow confidence interval</span> Reliable forecast</li>
      <li><span class="badge warn">Wide interval</span> Increased uncertainty</li>
    </ul>

    <b>Output:</b> Future rating estimates with confidence bounds.
    </div>
    """, unsafe_allow_html=True)

    # -------- MODEL 5 --------
    st.markdown("## 5️⃣ NLP Modeling — Book Category Prediction")
    st.markdown("""
    <div class="card">
    <b>Goal:</b> Automatically classify books into categories using text data.<br><br>

    <b>Technique:</b> TF-IDF Vectorization + Linear Support Vector Classifier (LinearSVC).<br><br>

    <b>Input Features:</b>
    <ul>
      <li>Book title</li>
      <li>Book description or keywords</li>
    </ul>

    <b>IDSS Logic:</b>
    <ul>
      <li><span class="badge ok">Clear keyword dominance</span> High confidence classification</li>
      <li><span class="badge warn">Mixed vocabulary</span> Moderate confidence</li>
    </ul>

    <b>Output:</b> Predicted book category with influential keywords.
    </div>
    """, unsafe_allow_html=True)

elif tab == "📈 Rating":
    colA, colB = st.columns([1.1, 0.9], gap="large")

    with colA:
        st.subheader("📈 Rating Prediction (Real ML)")
        st.caption("Move sliders to generate a prediction. For stability, use Recalculate if API is busy.")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**👤 Reader**")
            st.slider("Age", 10, 80, key="r_age", on_change=predict_rating)
            feature_help("Age")

            st.slider("User Avg Rating", 0.0, 10.0, key="r_user_avg", step=0.1, on_change=predict_rating)
            feature_help("user_avg_rating")

            st.slider("User Consistency", 0.0, 5.0, key="r_user_cons", step=0.1, on_change=predict_rating)
            feature_help("user_consistency")

        with c2:
            st.markdown("**📚 Book**")
            st.slider("Book Avg Rating", 0.0, 10.0, key="r_book_avg", step=0.1, on_change=predict_rating)
            feature_help("book_avg_rating")

            st.slider("Popularity", 1.0, 1000.0, key="r_pop", step=1.0, on_change=predict_rating)
            feature_help("book_popularity")

            st.slider("Book Rating Std", 0.0, 5.0, key="r_book_std", step=0.1, on_change=predict_rating)
            feature_help("book_rating_std")

        c3, c4 = st.columns(2)
        with c3:
            st.slider("Age-Book Gap", -50.0, 50.0, key="r_age_gap", step=1.0, on_change=predict_rating)
            feature_help("age_book_gap")
        with c4:
            st.slider("Rating Difference", -10.0, 10.0, key="r_diff", step=0.1, on_change=predict_rating)
            feature_help("rating_difference")

        if st.button("Recalculate", type="primary"):
            predict_rating()

    with colB:
        st.subheader("📊 Result & Explanation")

        if st.session_state.rating_result is None:
            st.info("Move sliders to get an instant prediction.")
        else:
            res = st.session_state.rating_result
            if not res.get("ok"):
                st.error(str(res))
            else:
                pred = res["data"]["prediction"]
                st.markdown(f"""
                <div class="card">
                  <p class="kpi-label">Predicted Rating</p>
                  <p class="kpi">{pred:.2f} / 10</p>
                </div>
                """, unsafe_allow_html=True)

                top = res["data"].get("top_features", [])
                st.markdown("**Why (global importance):**")
                if top:
                    df_imp = pd.DataFrame(top)
                    st.dataframe(df_imp, use_container_width=True, hide_index=True)
                else:
                    st.caption("No feature importance available.")

elif tab == "🎯 Like":
    colA, colB = st.columns([1.1, 0.9], gap="large")

    with colA:
        st.subheader("🎯 Like / Dislike (Real ML)")
        st.caption("Instant prediction + confidence. Use Recalculate if you move sliders quickly.")

        c1, c2 = st.columns(2)
        with c1:
            st.slider("Age", 10, 80, key="l_age", on_change=predict_like)
            feature_help("Age")

            st.slider("User Avg Rating", 0.0, 10.0, key="l_user_avg", step=0.1, on_change=predict_like)
            feature_help("user_avg_rating")

            st.slider("User Consistency", 0.0, 5.0, key="l_user_cons", step=0.1, on_change=predict_like)
            feature_help("user_consistency")

        with c2:
            st.slider("Book Avg Rating", 0.0, 10.0, key="l_book_avg", step=0.1, on_change=predict_like)
            feature_help("book_avg_rating")

            st.slider("Popularity", 1.0, 1000.0, key="l_pop", step=1.0, on_change=predict_like)
            feature_help("book_popularity")

            st.slider("Age-Book Gap", -50.0, 50.0, key="l_age_gap", step=1.0, on_change=predict_like)
            feature_help("age_book_gap")

        st.slider("Popularity Ratio", 0.0, 200.0, key="l_pop_ratio", step=0.5, on_change=predict_like)
        feature_help("popularity_ratio")

        if st.button("Recalculate Like", type="primary"):
            predict_like()

    with colB:
        st.subheader("📊 Result & Explanation")
        if st.session_state.like_result is None:
            st.info("Move sliders to get instant classification.")
        else:
            res = st.session_state.like_result
            if not res.get("ok"):
                st.error(str(res))
            else:
                pred = res["data"]["prediction"]
                proba = res["data"].get("proba")
                label = "✅ LIKE" if pred == 1 else "❌ DISLIKE"

                conf = None
                if proba:
                    conf = max(proba)

                st.markdown(f"""
                <div class="card">
                  <p class="kpi-label">Prediction</p>
                  <p class="kpi">{label}</p>
                </div>
                """, unsafe_allow_html=True)

                if conf is not None:
                    st.metric("Confidence", f"{conf:.1%}")

                st.markdown("**Top drivers (global importance):**")
                top = res["data"].get("top_features", [])
                if top:
                    st.dataframe(pd.DataFrame(top), use_container_width=True, hide_index=True)

elif tab == "👥 Clusters":
    st.subheader("👥 User Segmentation (K-Means)")
    st.caption("This assigns a user profile to a cluster segment based on behavior patterns.")

    st.slider("User Avg Rating", 0.0, 10.0, key="c_avg", step=0.1)
    feature_help("user_avg_rating")

    st.slider("User Activity", 0.0, 50.0, key="c_act", step=1.0)
    feature_help("user_activity")

    st.slider("User Consistency", 0.0, 5.0, key="c_cons", step=0.1)
    feature_help("user_consistency")

    st.slider("Age", 10, 80, key="c_age")
    feature_help("Age")

    if st.button("Assign Cluster", type="primary"):
        res = predict_cluster()
        if res.get("ok"):
            st.success(f"Cluster {res['data']['cluster']}")
            st.info(res["data"]["description"])
        else:
            st.error(str(res))

elif tab == "📅 Forecast":
    st.subheader("📅 Forecasting (Real ML: SARIMAX)")
    st.caption("Forecast monthly average rating, with confidence intervals.")

    st.slider("Months ahead", 1, 24, key="months_ahead", on_change=predict_forecast)
    feature_help("months_ahead")

    if st.button("Forecast now", type="primary"):
        predict_forecast()

    res = st.session_state.forecast_result
    if res is None:
        st.info("Move the slider to forecast instantly.")
    else:
        if not res.get("ok"):
            st.error(res["data"].get("error", str(res)))
        else:
            mean = res["data"]["mean"]
            conf = res["data"]["conf_int"]

            x = list(range(1, len(mean) + 1))
            low = [c[0] for c in conf]
            high = [c[1] for c in conf]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=mean, mode="lines+markers", name="Forecast"))
            fig.add_trace(go.Scatter(x=x, y=low, mode="lines", name="Lower", line=dict(dash="dash")))
            fig.add_trace(go.Scatter(x=x, y=high, mode="lines", name="Upper", line=dict(dash="dash")))
            fig.update_layout(
                title="Forecasted Monthly Average Rating",
                xaxis_title="Month Ahead",
                yaxis_title="Avg Rating"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Model explanation:** SARIMAX captures trend + seasonality + noise. Confidence bands show uncertainty.")

elif tab == "🔤 NLP":
    st.subheader("🔤 NLP Category Prediction (Real ML)")
    st.caption("TF-IDF text vectorization + LinearSVC classification.")

    st.text_area("Title / description", key="nlp_text", height=120, on_change=predict_nlp)
    feature_help("nlp_text")

    if st.button("Predict Category", type="primary"):
        predict_nlp()

    res = st.session_state.nlp_result
    if res is None:
        st.info("Type in the text box to get instant category prediction.")
    else:
        if not res.get("ok"):
            st.error(res["data"].get("error", str(res)))
        else:
            category = res["data"]["category"]
            keywords = res["data"].get("keywords", [])

            st.markdown(f"""
            <div class="card">
              <p class="kpi-label">Predicted Category</p>
              <p class="kpi">{category}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Keywords detected:**")
            if keywords:
                st.write(", ".join(keywords))
            else:
                st.caption("No strong keywords found (text too short or generic).")

            st.markdown("**Model explanation:** TF-IDF emphasizes informative terms; LinearSVC separates categories efficiently for fast demos.")

elif tab == "📊 EDA":
    st.subheader("📊 Exploratory Data Analysis")
    st.caption("EDA helps validate data quality and supports model choices (DSS requirement).")

    df, books, users = load_local_data()

    eda_tabs = st.tabs(["👤 Users", "📚 Books", "⭐ Ratings"])

    # -------- USERS EDA --------
    with eda_tabs[0]:
        c1, c2 = st.columns(2)

        with c1:
            fig = px.histogram(users, x="Age", nbins=25, title="User Age Distribution")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.histogram(users, x="user_activity", nbins=25, title="User Activity Distribution")
            st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(
            users,
            x="user_activity",
            y="user_avg_rating",
            title="User Activity vs User Average Rating",
            opacity=0.6
        )
        st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(
            users,
            x="user_consistency",
            y="user_avg_rating",
            title="User Consistency vs User Average Rating",
            opacity=0.6
        )
        st.plotly_chart(fig, use_container_width=True)

    # -------- BOOKS EDA --------
    with eda_tabs[1]:
        c1, c2 = st.columns(2)

        with c1:
            fig = px.histogram(books, x="book_avg_rating", nbins=25, title="Book Average Rating Distribution")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            # popularity is skewed; show log scale via transform for readability
            safe_pop = books["book_popularity"].replace(0, np.nan).dropna()
            if len(safe_pop) > 0:
                tmp = books.copy()
                tmp["log_popularity"] = np.log10(tmp["book_popularity"].replace(0, np.nan))
                fig = px.histogram(tmp, x="log_popularity", nbins=25, title="Book Popularity (log10 scale)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Popularity chart skipped (no valid popularity values).")

        fig = px.scatter(
            books,
            x="book_popularity",
            y="book_avg_rating",
            title="Book Popularity vs Book Avg Rating",
            opacity=0.5
        )
        st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(
            books,
            x="book_rating_std",
            y="book_avg_rating",
            title="Rating Std vs Book Avg Rating (Controversy vs Rating)",
            opacity=0.5
        )
        st.plotly_chart(fig, use_container_width=True)

    # -------- RATINGS EDA --------
    with eda_tabs[2]:
        fig = px.histogram(df, x="Book-Rating", nbins=11, title="Distribution of Given Ratings")
        st.plotly_chart(fig, use_container_width=True)

        if "Category" in df.columns:
            fig = px.box(df, x="Category", y="Book-Rating", title="Ratings by Category")
            st.plotly_chart(fig, use_container_width=True)

        if "Timestamp" in df.columns and df["Timestamp"].notna().any():
            tmp = df.dropna(subset=["Timestamp"]).copy()
            tmp["month"] = tmp["Timestamp"].dt.to_period("M").dt.to_timestamp()
            monthly = tmp.groupby("month")["Book-Rating"].mean().reset_index()

            fig = px.line(monthly, x="month", y="Book-Rating", title="Average Rating Over Time (Monthly)")
            st.plotly_chart(fig, use_container_width=True)

            monthly_count = tmp.groupby("month")["Book-Rating"].count().reset_index(name="count")
            fig = px.bar(monthly_count, x="month", y="count", title="Ratings Volume Over Time (Monthly)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No usable Timestamp data for time-based EDA charts.")

else:
    st.subheader("🧠 System Status")
    try:
        r = api_get("/health")
        st.json(r)
    except Exception as e:
        st.error(f"API not reachable: {e}")
        st.markdown("Run API first: `uvicorn api.main:app`")
