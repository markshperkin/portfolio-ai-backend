import logging
import logging.config
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.api.resume import router as resume_router
from app.db import init_pool, close_pool

# Structured logging — internals to stdout only, never to the wire
logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "format": '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}'
        }
    },
    "handlers": {
        "stdout": {"class": "logging.StreamHandler", "formatter": "json"}
    },
    "root": {"level": "INFO", "handlers": ["stdout"]},
})

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_pool()
    yield
    await close_pool()


app = FastAPI(title="portfolio-ai-backend", docs_url=None, redoc_url=None, lifespan=lifespan)

app.include_router(health_router)
app.include_router(chat_router)
app.include_router(resume_router)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 404:
        return JSONResponse(status_code=404, content={"error": "not found"})
    return JSONResponse(status_code=exc.status_code, content={"error": "request failed"})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"error": "invalid request"})


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    # Log full detail server-side; return nothing useful to the client
    log.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"error": "internal server error"})
