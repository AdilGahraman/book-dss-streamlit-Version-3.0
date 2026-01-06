from fastapi import FastAPI
from api.schemas import RatingRequest, LikeRequest, ClusterRequest, NlpRequest, ForecastRequest, ApiResponse
from api.ml import ModelStore

app = FastAPI(title="Book DSS API", version="1.0.0")

store = ModelStore()

@app.on_event("startup")
def startup():
    store.load()

@app.get("/health", response_model=ApiResponse)
def health():
    return {"ok": True, "data": {"loaded": True, "meta": store.meta}}

@app.post("/predict/rating", response_model=ApiResponse)
def predict_rating(req: RatingRequest):
    result = store.predict_rating(req.model_dump())
    return {"ok": True, "data": result}

@app.post("/predict/like", response_model=ApiResponse)
def predict_like(req: LikeRequest):
    result = store.predict_like(req.model_dump())
    return {"ok": True, "data": result}

@app.post("/predict/nlp", response_model=ApiResponse)
def predict_nlp(req: NlpRequest):
    result = store.predict_nlp(req.text)
    ok = "error" not in result
    return {"ok": ok, "data": result}

@app.post("/forecast", response_model=ApiResponse)
def forecast(req: ForecastRequest):
    result = store.forecast(req.months_ahead, req.future_rating_count)
    ok = "error" not in result
    return {"ok": ok, "data": result}

@app.post("/predict/cluster", response_model=ApiResponse)
def predict_cluster(req: ClusterRequest):
    result = store.predict_cluster(req.model_dump())
    return {"ok": True, "data": result}
