"""Simple FastAPI server for serving saved models.

Run with:
    uvicorn serve:app --reload --port 8000
"""
import os
import io
import joblib
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Churn Model Serving")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


class PredictRequest(BaseModel):
    rows: List[dict]
    model_filename: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(req: PredictRequest):
    # Load model
    model_filename = req.model_filename
    if model_filename is None:
        raise HTTPException(status_code=400, detail="model_filename is required")
    model_path = os.path.join(MODELS_DIR, model_filename)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="model not found")

    model = joblib.load(model_path)
    df = pd.DataFrame(req.rows)
    try:
        preds = model.predict(df)
        proba = model.predict_proba(df)[:, 1] if hasattr(model, "predict_proba") else None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    out = df.copy()
    out["pred_churn"] = preds
    if proba is not None:
        out["pred_churn_proba"] = proba.tolist()
    return out.to_dict(orient="records")


@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...), model_filename: str = ""):
    if not model_filename:
        raise HTTPException(status_code=400, detail="model_filename query param is required")
    model_path = os.path.join(MODELS_DIR, model_filename)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="model not found")
    model = joblib.load(model_path)
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    try:
        preds = model.predict(df)
        proba = model.predict_proba(df)[:, 1] if hasattr(model, "predict_proba") else None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    out = df.copy()
    out["pred_churn"] = preds
    if proba is not None:
        out["pred_churn_proba"] = proba.tolist()
    return out.to_dict(orient="records")
