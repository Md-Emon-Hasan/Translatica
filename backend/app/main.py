"""
FastAPI Application Entry Point
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from app.api.routes import router
from app.core.config import settings
from app.core.database import init_db
from app.core.model import model_manager
from app.core.rate_limit import limiter, rate_limit_exceeded_handler
from app.utils.logger import get_logger

logger = get_logger("main")

# Templates removed


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - load model on startup, cleanup on shutdown."""
    logger.info("Starting application...")

    # Load model in a background thread
    import threading

    def load_task():
        try:
            model_manager.load()
            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.error(f"CRITICAL: Failed to load model: {e}")

    threading.Thread(target=load_task, daemon=True).start()

    logger.info("Initializing database...")
    await init_db()

    yield
    logger.info("Shutting down application...")
    model_manager.cleanup()
    logger.info("Cleanup complete.")


app = FastAPI(
    title=settings.APP_NAME,
    description="AI English-Spanish translator",
    version="1.0.0",
    lifespan=lifespan,
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return a 422 with a readable message the frontend can display."""
    problems = []
    for error in exc.errors():
        field = ".".join(str(part) for part in error.get("loc", ()) if part != "body")
        message = error.get("msg", "Invalid value")
        message = message.replace("Value error, ", "")
        problems.append({"field": field or "body", "message": message})

    detail = "; ".join(
        f"{problem['field']}: {problem['message']}" for problem in problems
    )
    logger.warning(f"Validation failed: {detail}")

    return JSONResponse(
        status_code=422,
        content={"detail": detail, "error": detail, "errors": problems},
    )


# Mount static files FIRST (before router, so 'static' route is available)
# Static files mount removed as frontend is now separate

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
