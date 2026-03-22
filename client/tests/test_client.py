import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from client.inference_client import InferenceClient


class TestInferenceClientInit:
    def test_init_defaults(self):
        c = InferenceClient()
        assert c._host == "localhost"
        assert c._port == 50051
        assert c._http_port == 8080
        assert c._jpeg_quality == 85

    def test_init_custom_params(self):
        c = InferenceClient(
            host="192.168.1.10",
            port=9000,
            http_port=9001,
            jpeg_quality=95,
            timeout_seconds=60.0,
        )
        assert c._host == "192.168.1.10"
        assert c._port == 9000
        assert c._http_port == 9001
        assert c._jpeg_quality == 95
        assert c._timeout_seconds == 60.0


class TestInferenceClientConnect:
    def test_connect_timeout_raises(self):
        c = InferenceClient(host="localhost", port=59999, timeout_seconds=2)
        with pytest.raises(ConnectionError):
            c.connect()


class TestInferenceClientImagePrep:
    def test_prepare_image_returns_bytes(self):
        c = InferenceClient()
        c._input_width = 640
        c._input_height = 640
        img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        result = c._prepare_image(img)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_prepare_image_small_image(self):
        c = InferenceClient()
        c._input_width = 640
        c._input_height = 640
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = c._prepare_image(img)
        assert isinstance(result, bytes)


class TestInferenceClientDeserialize:
    def test_deserialize_empty_response(self):
        c = InferenceClient()
        # Create a mock response with no detections
        mock_response = MagicMock()
        mock_response.detections = []
        mock_response.image_width = 640
        mock_response.image_height = 480

        result = c._deserialize_response(mock_response)
        import supervision as sv
        assert isinstance(result, sv.Detections)
        assert len(result) == 0

    def test_deserialize_response_with_detections(self):
        c = InferenceClient()
        det1 = MagicMock()
        det1.bbox = [10.0, 20.0, 100.0, 200.0]
        det1.confidence = 0.95
        det1.class_id = 0

        det2 = MagicMock()
        det2.bbox = [50.0, 60.0, 150.0, 250.0]
        det2.confidence = 0.85
        det2.class_id = 1

        mock_response = MagicMock()
        mock_response.detections = [det1, det2]
        mock_response.image_width = 640
        mock_response.image_height = 480

        result = c._deserialize_response(mock_response)
        import supervision as sv
        assert isinstance(result, sv.Detections)
        assert len(result) == 2
        assert result.xyxy.shape == (2, 4)
        assert result.confidence[0] == pytest.approx(0.95)
        assert result.class_id[1] == 1


class TestFetchConfig:
    def test_fetch_config_updates_dimensions(self):
        c = InferenceClient()
        mock_config = {
            "model_name": "yolo11n",
            "model_type": "yolo11",
            "backend": "pytorch",
            "input_width": 320,
            "input_height": 320,
            "version": "1.0.0",
            "device": "cpu",
        }
        with patch("client.inference_client.httpx") as mock_httpx:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = mock_config
            mock_httpx.get.return_value = mock_resp

            result = c.fetch_config()

        assert c._input_width == 320
        assert c._input_height == 320
        assert result == mock_config


class TestStreamPredict:
    def test_stream_predict_raises_if_not_connected(self):
        c = InferenceClient()
        with pytest.raises(RuntimeError):
            list(c.stream_predict(iter([])))

    def test_stream_predict_calls_stub(self):
        c = InferenceClient()
        c._stub = MagicMock()
        c._input_width = 640
        c._input_height = 640

        # Mock stub.StreamPredict to return empty iterator
        c._stub.StreamPredict.return_value = iter([])

        result = list(c.stream_predict(iter([])))
        assert result == []
        c._stub.StreamPredict.assert_called_once()

    def test_stream_predict_yields_detections(self):
        c = InferenceClient()
        c._stub = MagicMock()
        c._input_width = 640
        c._input_height = 640

        mock_det = MagicMock()
        mock_det.bbox = [10.0, 20.0, 100.0, 200.0]
        mock_det.confidence = 0.9
        mock_det.class_id = 0

        mock_response = MagicMock()
        mock_response.detections = [mock_det]
        mock_response.image_width = 640
        mock_response.image_height = 480

        c._stub.StreamPredict.return_value = iter([mock_response])

        frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)]
        results = list(c.stream_predict(iter(frames)))

        import supervision as sv
        assert len(results) == 1
        assert isinstance(results[0], sv.Detections)
        assert len(results[0]) == 1
