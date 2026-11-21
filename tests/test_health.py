from fastapi.testclient import TestClient

from app.api.schemas import HealthResponse
from app.core.config import get_settings
from app.main import app


def test_health_check_returns_200_and_valid_schema() -> None:
    settings = get_settings()
    with TestClient(app) as client:
        response = client.get(f"{settings.API_V1_PREFIX}/health")
        assert response.status_code == 200

        payload = response.json()
        validated = HealthResponse.model_validate(payload)
        assert validated.status == "healthy"
        assert validated.version == settings.APP_VERSION
        assert validated.environment == settings.ENVIRONMENT
