"""Structured errors: every error response is ``{"error": {"code", "message"}, "detail"}``.

``detail`` keeps FastAPI's default key so existing clients of ``/api/track`` keep working.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

STATUS_CODES = {
    400: "bad_request",
    404: "not_found",
    409: "conflict",
    413: "too_large",
    415: "unsupported_media_type",
    422: "invalid_input",
    429: "busy",
    500: "internal_error",
    503: "unavailable",
}


class ApiError(HTTPException):
    def __init__(self, status: int, code: str, message: str, **extra: Any) -> None:
        super().__init__(status, message)
        self.code = code
        self.extra = extra


def body(status: int, message: str, code: str | None = None, **extra: Any) -> dict[str, Any]:
    return {
        "error": {"code": code or STATUS_CODES.get(status, "error"), "message": message, **extra},
        "detail": message,
    }


def install(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = getattr(exc, "code", None)
        extra = getattr(exc, "extra", {})
        return JSONResponse(body(exc.status_code, str(exc.detail), code, **extra), exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        errs = [{"loc": list(e.get("loc", ())), "msg": e.get("msg", "")} for e in exc.errors()]
        msg = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in errs)
        return JSONResponse(body(422, msg or "invalid request", "invalid_input", fields=errs), 422)
