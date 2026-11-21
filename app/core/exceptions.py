from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class DocIntelException(Exception):
    """Base domain exception for the DocIntel application."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)


class DocumentParsingError(DocIntelException):
    """Raised when a document fails extraction or structural parsing."""

    def __init__(
        self,
        message: str = "Failed to parse document",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message=message, status_code=422, details=details)


class DocumentNotFoundError(DocIntelException):
    """Raised when a requested document cannot be located."""

    def __init__(
        self,
        message: str = "Document not found",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message=message, status_code=404, details=details)


class FileSizeExceededError(DocIntelException):
    """Raised when an uploaded document exceeds the maximum allowed payload size."""

    def __init__(
        self,
        message: str = "File size exceeds configured limit",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message=message, status_code=413, details=details)


def register_exception_handlers(app: FastAPI) -> None:
    """Registers centralized domain and validation exception handlers on the FastAPI app."""

    @app.exception_handler(DocIntelException)
    async def doc_intel_exception_handler(
        request: Request, exc: DocIntelException
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.__class__.__name__,
                "message": exc.message,
                "details": exc.details,
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": "RequestValidationError",
                "message": "Input validation failed",
                "details": {"errors": exc.errors()},
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "message": "An unexpected server error occurred",
                "details": {"error_type": exc.__class__.__name__},
            },
        )
