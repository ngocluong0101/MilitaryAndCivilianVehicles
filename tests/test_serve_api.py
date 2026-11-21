import pytest
from fastapi.testclient import TestClient

from serve_helpers import StubDetector, png_bytes, write_fake_onnx
from vehicledet.serve.app import create_app


@pytest.fixture
def stub():
    return StubDetector()


@pytest.fixture
def client(stub):
    with TestClient(create_app(stub)) as c:
        yield c


def post(client, data, name="a.png", ctype="image/png", **params):
    return client.post("/predict", files={"file": (name, data, ctype)}, params=params)


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["backend"] == "stub" and "military tank" in body["classes"]


def test_predict_returns_boxes_and_timings(client, stub):
    r = post(client, png_bytes((64, 48)))
    assert r.status_code == 200
    body = r.json()
    assert body["image"] == {"width": 64, "height": 48}
    assert body["detections"] == [
        {
            "class_id": 2,
            "label": "military tank",
            "score": 0.91,
            "bbox": {"x1": 1.0, "y1": 2.0, "x2": 30.0, "y2": 40.0},
        }
    ]
    assert body["inference_ms"] == 1.5 and body["total_ms"] == 2.5 and body["backend"] == "stub"
    assert stub.calls == [((64, 48), 0.25, 0.7)]  # giá trị mặc định


def test_predict_passes_thresholds(client, stub):
    r = post(client, png_bytes(), conf=0.95, iou=0.4)
    assert r.status_code == 200 and r.json()["detections"] == []
    assert stub.calls[-1][1:] == (0.95, 0.4)


@pytest.mark.parametrize(
    "params", [{"conf": 1.5}, {"conf": -0.1}, {"iou": 0}, {"iou": 2}, {"conf": "x"}]
)
def test_predict_validates_thresholds(client, params):
    assert post(client, png_bytes(), **params).status_code == 422


def test_predict_requires_file(client):
    assert client.post("/predict").status_code == 422


def test_predict_rejects_non_image_and_empty(client):
    r = post(client, b"this is not an image", "a.txt", "text/plain")
    assert r.status_code == 400 and "ảnh" in r.json()["detail"]
    assert post(client, b"").status_code == 400
    assert post(client, png_bytes()[:30]).status_code == 400  # ảnh cắt cụt


def test_predict_rejects_oversized_file(stub):
    with TestClient(create_app(stub, max_upload_mb=0.001)) as c:  # ~1 KB
        assert post(c, b"\x89PNG" + b"0" * 5000).status_code == 413
        assert post(c, png_bytes((8, 8))).status_code == 200  # ảnh nhỏ vẫn qua


def test_service_degraded_without_model(monkeypatch):
    monkeypatch.delenv("MODEL_PATH", raising=False)
    with TestClient(create_app()) as c:
        h = c.get("/health").json()
        assert (
            h["status"] == "degraded" and h["model_loaded"] is False and "MODEL_PATH" in h["error"]
        )
        r = post(c, png_bytes())
        assert r.status_code == 503 and "chưa sẵn sàng" in r.json()["detail"]


def test_service_reports_bad_model_path(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "nope.onnx"))
    with TestClient(create_app()) as c:
        assert "FileNotFoundError" in c.get("/health").json()["error"]
        assert post(c, png_bytes()).status_code == 503


def test_loads_real_onnx_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(write_fake_onnx(tmp_path / "m.onnx")))
    monkeypatch.setenv("DEVICE", "cpu")
    with TestClient(create_app()) as c:
        assert c.get("/health").json()["backend"] == "onnxruntime (CPU)"
        r = post(c, png_bytes((640, 480)))
        assert r.status_code == 200
        assert [d["label"] for d in r.json()["detections"]] == ["military tank", "civilian car"]


def test_inference_failure_is_500_without_leaking_details(stub):
    def boom(*a, **k):
        raise RuntimeError("secret internal path /home/x")

    stub.predict = boom
    with TestClient(create_app(stub)) as c:
        r = post(c, png_bytes())
        assert r.status_code == 500 and "secret" not in r.text
