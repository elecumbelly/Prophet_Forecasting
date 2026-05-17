"""End-to-end tests for the Flask JSON API."""
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from prophet_forecasting_tool.app import _IpRateLimiter, create_app


@pytest.fixture
def mock_engine():
    engine = MagicMock()
    return engine


@pytest.fixture
def mock_service():
    svc = MagicMock()
    svc.get_forecast.return_value = {
        "series_name": "test_series",
        "forecast_df": pd.DataFrame({
            "ds": pd.date_range("2024-01-01", periods=5),
            "yhat": list(range(5)),
        }),
        "training_rows": 100,
        "training_start": "2023-01-01T00:00:00",
        "training_end": "2024-01-01T00:00:00",
        "outliers_removed": 0,
        "forecast_plot": "data:image/png;base64,AAA",
        "components_plot": "data:image/png;base64,BBB",
    }
    return svc


@pytest.fixture
def client(tmp_path, mock_engine, mock_service):
    settings = MagicMock()
    settings.DATABASE_URL = "sqlite:///:memory:"
    app = create_app(
        settings=settings,
        engine=mock_engine,
        forecasting_service=mock_service,
        output_root=tmp_path / "outputs_web",
    )
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestHealth:
    def test_ok(self, client):
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json == {"status": "ok"}


class TestReady:
    def test_db_reachable(self, client, mock_engine):
        conn = MagicMock()
        conn.__enter__ = MagicMock(return_value=conn)
        conn.__exit__ = MagicMock(return_value=False)
        mock_engine.connect.return_value = conn
        resp = client.get("/readyz")
        assert resp.status_code == 200
        assert resp.json["status"] == "ok"

    def test_db_unreachable(self, client, mock_engine):
        mock_engine.connect.side_effect = RuntimeError("connection refused")
        resp = client.get("/readyz")
        assert resp.status_code == 503
        assert resp.json["status"] == "degraded"


class TestGetColumns:
    @patch("prophet_forecasting_tool.app.inspect")
    def test_success(self, mock_inspect, client):
        inspector = MagicMock()
        inspector.has_table.return_value = True
        inspector.get_columns.return_value = [{"name": "ts"}, {"name": "y"}]
        mock_inspect.return_value = inspector

        resp = client.get("/get_columns/demo_metrics")
        assert resp.status_code == 200
        assert resp.json == {"columns": ["ts", "y"]}

    @patch("prophet_forecasting_tool.app.inspect")
    def test_missing_table(self, mock_inspect, client):
        inspector = MagicMock()
        inspector.has_table.return_value = False
        mock_inspect.return_value = inspector

        resp = client.get("/get_columns/no_such_table")
        assert resp.status_code == 404

    def test_invalid_identifier(self, client):
        resp = client.get("/get_columns/bad;table")
        assert resp.status_code == 400


class TestApiHistorical:
    @patch("prophet_forecasting_tool.app.load_time_series")
    @patch("prophet_forecasting_tool.app.inspect")
    def test_success(self, mock_inspect, mock_load, client):
        inspector = MagicMock()
        inspector.has_table.return_value = True
        inspector.get_columns.return_value = [
            {"name": "ts"}, {"name": "y"}, {"name": "answer_rate"}
        ]
        mock_inspect.return_value = inspector

        mock_load.return_value = pd.DataFrame({
            "ds": pd.date_range("2024-01-01", periods=3),
            "y": [10, 20, 30],
        })

        resp = client.post(
            "/api/historical_data",
            json={"table": "demo", "ts_column": "ts", "y_column": "y"},
        )
        assert resp.status_code == 200
        assert resp.json["row_count"] == 3

    def test_invalid_table(self, client):
        resp = client.post(
            "/api/historical_data",
            json={"table": "bad;table", "ts_column": "ts", "y_column": "y"},
        )
        assert resp.status_code == 400

    @patch("prophet_forecasting_tool.app.inspect")
    def test_missing_column(self, mock_inspect, client):
        inspector = MagicMock()
        inspector.has_table.return_value = True
        inspector.get_columns.return_value = [{"name": "ts"}, {"name": "y"}]
        mock_inspect.return_value = inspector

        resp = client.post(
            "/api/historical_data",
            json={"table": "demo", "ts_column": "not_a_col", "y_column": "y"},
        )
        assert resp.status_code == 400
        assert "not_a_col" in resp.json["error"]


class TestApiForecast:
    @patch("prophet_forecasting_tool.app.inspect")
    def test_success(self, mock_inspect, client, mock_service):
        inspector = MagicMock()
        inspector.has_table.return_value = True
        inspector.get_columns.return_value = [{"name": "ts"}, {"name": "y"}]
        mock_inspect.return_value = inspector

        resp = client.post(
            "/api/forecast",
            json={
                "table": "demo",
                "ts_column": "ts",
                "y_column": "y",
                "freq": "D",
                "horizon": "30D",
                "series_name": "test_series",
                "regressors": [],
                "training_window_duration": "180 days",
            },
        )
        assert resp.status_code == 200
        body = resp.json
        assert body["series_name"] == "test_series"
        assert body["row_count"] == 5
        assert body["forecast_plot"].startswith("data:image/png")
        # ds must be JSON-safe (ISO-8601 string, not a Timestamp object).
        first_row = body["preview"][0]
        assert isinstance(first_row["ds"], str)
        assert first_row["ds"].startswith("2024-01-01T")
        mock_service.get_forecast.assert_called_once()

    def test_invalid_horizon(self, client):
        resp = client.post(
            "/api/forecast",
            json={
                "table": "demo",
                "ts_column": "ts",
                "y_column": "y",
                "freq": "D",
                "horizon": "forever",
                "series_name": "test",
                "regressors": [],
            },
        )
        assert resp.status_code == 400

    def test_invalid_identifier(self, client):
        resp = client.post(
            "/api/forecast",
            json={
                "table": "bad;table",
                "ts_column": "ts",
                "y_column": "y",
                "freq": "D",
                "horizon": "30D",
                "series_name": "test",
                "regressors": [],
            },
        )
        assert resp.status_code == 400

    @patch("prophet_forecasting_tool.app.inspect")
    def test_invalid_freq(self, mock_inspect, client):
        inspector = MagicMock()
        inspector.has_table.return_value = True
        inspector.get_columns.return_value = [{"name": "ts"}, {"name": "y"}]
        mock_inspect.return_value = inspector

        resp = client.post(
            "/api/forecast",
            json={
                "table": "demo",
                "ts_column": "ts",
                "y_column": "y",
                "freq": "Q",
                "horizon": "30D",
                "series_name": "test",
                "regressors": [],
            },
        )
        assert resp.status_code == 400


class TestCors:
    def test_cors_allows_local_dev(self, client):
        resp = client.post(
            "/api/forecast",
            json={
                "table": "demo",
                "ts_column": "ts",
                "y_column": "y",
                "freq": "D",
                "horizon": "bad",  # 400 path; we just want to inspect headers
                "series_name": "t",
                "regressors": [],
            },
            headers={"Origin": "http://localhost:3001"},
        )
        # CORS reflects the allowlisted origin (not *).
        assert resp.headers.get("Access-Control-Allow-Origin") == "http://localhost:3001"

    def test_cors_denies_unknown_origin(self, client):
        resp = client.post(
            "/api/forecast",
            json={
                "table": "demo",
                "ts_column": "ts",
                "y_column": "y",
                "freq": "D",
                "horizon": "bad",
                "series_name": "t",
                "regressors": [],
            },
            headers={"Origin": "http://evil.example.com"},
        )
        assert "Access-Control-Allow-Origin" not in resp.headers


class TestServeOutputs:
    def test_rejects_path_traversal(self, client):
        resp = client.get("/outputs_web/../etc/passwd")
        # Flask should normalize and either 400 or 404; both are safe.
        assert resp.status_code in (400, 404)

    def test_rejects_unknown_extension(self, client):
        resp = client.get("/outputs_web/somefile.exe")
        assert resp.status_code == 404


class TestAutoTuneRateLimit:
    @patch("prophet_forecasting_tool.app.inspect")
    def test_blocks_after_threshold(self, mock_inspect, client, mock_service):
        inspector = MagicMock()
        inspector.has_table.return_value = True
        inspector.get_columns.return_value = [{"name": "ts"}, {"name": "y"}]
        mock_inspect.return_value = inspector

        payload = {
            "table": "demo",
            "ts_column": "ts",
            "y_column": "y",
            "freq": "D",
            "horizon": "30D",
            "series_name": "test",
            "regressors": [],
            "auto_tune": True,
        }

        # First 3 requests must pass (AUTOTUNE_LIMIT_COUNT = 3).
        for _ in range(3):
            resp = client.post("/api/forecast", json=payload)
            assert resp.status_code == 200

        # 4th request from the same client is rejected with 429.
        resp = client.post("/api/forecast", json=payload)
        assert resp.status_code == 429
        assert "rate limit" in resp.json["error"].lower()

    def test_empty_buckets_are_pruned(self):
        """Long-running services must not accumulate one bucket per unique IP forever."""
        import time

        limiter = _IpRateLimiter(max_requests=2, window_seconds=0.01)
        # First record an IP that we'll let expire.
        for i in range(50):
            limiter.allow(f"10.0.0.{i}")
        assert len(limiter._buckets) == 50

        # Sleep long enough for the window to expire.
        time.sleep(0.05)

        # Now hammer allow() from a single IP to trip the prune. Each call
        # also evicts the stale timestamp from "10.0.0.<i>" (which we never
        # touch again), leaving those buckets empty. The prune wipes them.
        for _ in range(_IpRateLimiter._PRUNE_EVERY + 1):
            limiter.allow("repeat")

        # All 50 one-shot IPs should have been pruned; only "repeat" remains.
        assert len(limiter._buckets) <= 1, (
            f"Bucket dict grew unbounded: {len(limiter._buckets)} entries"
        )

    @patch("prophet_forecasting_tool.app.inspect")
    def test_non_autotune_not_limited(self, mock_inspect, client, mock_service):
        inspector = MagicMock()
        inspector.has_table.return_value = True
        inspector.get_columns.return_value = [{"name": "ts"}, {"name": "y"}]
        mock_inspect.return_value = inspector

        payload = {
            "table": "demo",
            "ts_column": "ts",
            "y_column": "y",
            "freq": "D",
            "horizon": "30D",
            "series_name": "test",
            "regressors": [],
            "auto_tune": False,
        }
        # 10 non-auto-tune requests in a row must all pass.
        for _ in range(10):
            resp = client.post("/api/forecast", json=payload)
            assert resp.status_code == 200
