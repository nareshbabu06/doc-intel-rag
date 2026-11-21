from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    """Response schema for the system health check endpoint."""

    model_config = ConfigDict(frozen=True)

    status: str
    version: str
    environment: str
