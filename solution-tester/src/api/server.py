import logging
import time
import json
import dataclasses
import uuid
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
import asyncio

from src.config import settings
from src.utils import create_health_response, create_error_response
from src.api.datatypes import ExecutionRequest
import src.executor as executor
import src.store.recorder as recorder
from src.logging import (
    request_id,
    configure_logging,
    LoggingContextRoute,
)

# Configure logging (request_id-aware)
configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""

    # Startup
    logger.info("Starting Tester Service...")

    yield
    # Shutdown
    logger.info("Shutting down Tester Service...")


# Create FastAPI app
app = FastAPI(
    title="CodeAssistant Tester Service",
    description="Evaluates proposed solutions against test cases",
    version="1.0.0",
    lifespan=lifespan,
)

# Apply route class to auto-populate logging context
app.router.route_class = LoggingContextRoute

origins = [
    "http://localhost:3001",
    "http://localhost:3000",
    # or "http://localhost:YOUR_NEW_WEB_PORT" 
    #example 
    #"http://localhost:3050"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Attach a request id to context and response headers for correlation."""
    rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    token = request_id.set(rid)
    try:
        response = await call_next(request)
    finally:
        request_id.reset(token)
    response.headers["X-Request-ID"] = rid
    return response


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "CodeAssistant Tester Service",
        "version": "1.0.0",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""

    try:
        return create_health_response(status="healthy")
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return create_health_response("unhealthy")


@app.post("/execute")
async def execute(request: Request):
    """Execution request endpoint to evaluate test cases."""

    try:
        # Receive and unpack the ExecutionRequest
        request_data = await request.json()

        try:
            request = ExecutionRequest(**request_data)
            episode_id = request.episode_id
            timestep = request.timestep

            if request.store_activity:
                recorder.store_request(episode_id, timestep, request_data)

            result = await executor.go(request)
            result_dict = dataclasses.asdict(result)

            if request.store_activity:
                recorder.store_response(episode_id, timestep, json.dumps(result_dict))

            return JSONResponse(content=result_dict)

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Execution error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=create_error_response(error="Internal server error", details=str(exc)),
    )
