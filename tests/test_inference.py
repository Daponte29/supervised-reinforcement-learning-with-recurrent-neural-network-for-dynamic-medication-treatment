import pytest
import numpy as np


# TODO: point at a real exported model artifact to run these tests

# def test_onnx_inference_shape():
#     from deploy.inference import ONNXInferenceEngine
#     engine = ONNXInferenceEngine("deploy/artifacts/model.onnx")
#     x = np.zeros((1, 3, 224, 224), dtype=np.float32)
#     out = engine.predict(x)
#     assert out.shape[0] == 1


# def test_server_health(client):
#     response = client.get("/health")
#     assert response.status_code == 200
#     assert response.json() == {"status": "ok"}


# def test_server_predict(client):
#     payload = {"data": [[0.0] * FEATURE_DIM]}
#     response = client.post("/predict", json=payload)
#     assert response.status_code == 200
#     assert "predictions" in response.json()


def test_placeholder():
    pass  # remove once real inference tests are wired up
