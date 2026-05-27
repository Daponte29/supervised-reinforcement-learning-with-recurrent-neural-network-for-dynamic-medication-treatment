"""FastAPI inference server."""
from __future__ import annotations

import numpy as np
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from deploy.inference import load_engine

app = FastAPI(title="Model Serving API")

MODEL_PATH = os.getenv("MODEL_PATH", "deploy/artifacts/model.onnx")
engine = load_engine(MODEL_PATH)


class PredictRequest(BaseModel):
    # TODO: define your input schema
    data: list[list[float]]   # e.g. batch of feature vectors


class PredictResponse(BaseModel):
    predictions: list


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        x = np.array(req.data)
        preds = engine.predict(x)
        return PredictResponse(predictions=preds.tolist())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
