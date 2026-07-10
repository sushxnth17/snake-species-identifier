from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time
import logging
import uuid

import os
import tensorflow as tf
from typing import List
from datetime import datetime
from backend.config import settings, Settings
from backend.schemas import PredictionResponse, ErrorResponse, ModelInfoResponse, MetricsResponse, HealthResponse
from backend.logging_config import setup_structured_logging, request_id_var
from backend import model_loader
from backend import predictor
from backend.metadata import get_snake_metadata
from backend.metrics import DiagnosticsMetrics, metrics_tracker
from backend.dependencies import get_settings, get_logger, get_metrics_tracker, get_model, get_class_names, get_calibrator

from backend.validation import validate_uploaded_image, read_and_validate_size
from backend.rate_limit import check_rate_limit_dependency, rate_limiter_cleanup_task
from backend.middleware import SecurityAndLoggingMiddleware
# Setup Logging
setup_structured_logging(settings.logging_level)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the model and class names exactly once.
    logger.info("Initializing application resources during startup lifespan...", extra={"event": "startup_init"})
    start_time = time.perf_counter()
    try:
        model_loader.load_model()
        logger.info("Model loaded successfully.", extra={"event": "model_loaded", "model_path": settings.model_path})
        
        model_loader.load_class_names()
        logger.info("Class names loaded successfully.", extra={"event": "class_names_loaded", "class_names_path": settings.class_names_path})
        
        model_loader.load_calibration()
        logger.info("Calibration info loaded successfully.", extra={"event": "calibration_loaded", "calibration_path": settings.calibration_info_path})
        
        startup_duration = time.perf_counter() - start_time
        logger.info(
            "Backend started successfully.",
            extra={
                "event": "backend_started",
                "startup_duration_seconds": startup_duration
            }
        )
    except Exception as e:
        startup_duration = time.perf_counter() - start_time
        logger.error(
            f"Startup failed to initialize TensorFlow model. API endpoints will fall back to mock predictions. Details: {e}",
            exc_info=e,
            extra={
                "event": "startup_failed",
                "startup_duration_seconds": startup_duration
            }
        )

    # Start background cleanup task for rate limiter
    cleanup_task = None
    if settings.rate_limit_enabled:
        import asyncio
        cleanup_task = asyncio.create_task(
            rate_limiter_cleanup_task(settings.rate_limit_window)
        )
        logger.info("Rate limiter background cleanup task started.", extra={"event": "rate_limit_cleanup_started"})

    yield
    # Shutdown logic
    if cleanup_task is not None:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
        logger.info("Rate limiter background cleanup task stopped.", extra={"event": "rate_limit_cleanup_stopped"})
        
    logger.info("Cleaning up application resources during shutdown lifespan...", extra={"event": "shutdown_cleanup"})

app = FastAPI(
    title=settings.api_title,
    description=settings.app_description,
    version=settings.api_version,
    debug=settings.debug,
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security, Logging, and Abuse Detection Middleware
app.add_middleware(SecurityAndLoggingMiddleware)


# --- Global Exception Handlers ---

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(
        f"HTTPException: status_code={exc.status_code}, detail={exc.detail}",
        exc_info=exc,
        extra={
            "event": "error",
            "endpoint": request.url.path,
            "status_code": exc.status_code
        }
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.status_code, "message": exc.detail}},
        headers=getattr(exc, "headers", None)
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    error_msg = "; ".join([f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in errors])
    logger.error(
        f"RequestValidationError: {error_msg}",
        exc_info=exc,
        extra={
            "event": "error",
            "endpoint": request.url.path,
            "validation_errors": errors
        }
    )
    return JSONResponse(
        status_code=422,
        content={"error": {"code": 422, "message": f"Validation Error: {error_msg}"}}
    )

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    logger.error(
        f"ValueError: {exc}",
        exc_info=exc,
        extra={
            "event": "error",
            "endpoint": request.url.path
        }
    )
    return JSONResponse(
        status_code=400,
        content={"error": {"code": 400, "message": f"Bad Request: {str(exc)}"}}
    )

@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request: Request, exc: FileNotFoundError):
    logger.error(
        f"FileNotFoundError: {exc}",
        exc_info=exc,
        extra={
            "event": "error",
            "endpoint": request.url.path
        }
    )
    return JSONResponse(
        status_code=503,
        content={"error": {"code": 503, "message": f"Service Unavailable: Required resource missing ({str(exc)})"}}
    )

@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    logger.error(
        f"RuntimeError: {exc}",
        exc_info=exc,
        extra={
            "event": "error",
            "endpoint": request.url.path
        }
    )
    return JSONResponse(
        status_code=500,
        content={"error": {"code": 500, "message": f"Internal Server Error: {str(exc)}"}}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(
        f"Unhandled exception: {exc}",
        exc_info=exc,
        extra={
            "event": "error",
            "endpoint": request.url.path
        }
    )
    return JSONResponse(
        status_code=500,
        content={"error": {"code": 500, "message": "An unexpected error occurred on the server."}}
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Check System Health",
    description="Verifies the operational status of the API service, CORS bindings, and checks whether the TensorFlow classification model has successfully loaded into memory.",
    response_description="A JSON status object indicating health and model load state.",
    responses={
        200: {
            "description": "API is healthy and running.",
            "content": {
                "application/json": {
                    "example": {
                        "api_status": "healthy",
                        "model_status": "loaded",
                        "version": "1.0.0",
                        "uptime_seconds": 120.45,
                        "timestamp": "2026-07-05T21:40:00Z",
                        "status": "healthy",
                        "model_loaded": True
                    }
                }
            }
        }
    }
)
def health_check(
    request: Request,
    settings: Settings = Depends(get_settings),
    logger: logging.Logger = Depends(get_logger),
    metrics: DiagnosticsMetrics = Depends(get_metrics_tracker)
):
    """
    Health check endpoint to verify backend status, model readiness, uptime, and version.
    """
    logger.info(
        "Health endpoint accessed",
        extra={
            "event": "health_check",
            "endpoint": request.url.path
        }
    )
    model_loaded = model_loader._model is not None
    model_status = "loaded" if model_loaded else "not_loaded"
    uptime_sec = round(metrics.uptime, 2)
    timestamp_str = datetime.utcnow().isoformat() + "Z"
    
    return {
        "api_status": "healthy",
        "model_status": model_status,
        "version": settings.api_version,
        "uptime_seconds": uptime_sec,
        "timestamp": timestamp_str,
        # Backward compatibility legacy fields
        "status": "healthy",
        "model_loaded": model_loaded
    }

@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Get Classification Model Information",
    description="Returns metadata about the active machine learning model, including model format, name, supported classes, input size, and whether the weights are loaded.",
    response_description="A structured object detailing model attributes and readiness."
)
def get_model_info(
    request: Request,
    settings: Settings = Depends(get_settings),
    logger: logging.Logger = Depends(get_logger),
    class_names: List[str] = Depends(get_class_names)
):
    """
    Get detailed information about the configured machine learning model.
    """
    logger.info(
        "Model info endpoint accessed",
        extra={
            "event": "model_info_check",
            "endpoint": request.url.path
        }
    )
    
    model_loaded = model_loader._model is not None
    model_filename = os.path.basename(settings.model_path)
    
    model_ext = os.path.splitext(model_filename)[1].lower()
    model_format = "Keras" if model_ext == ".keras" else ("HDF5" if model_ext in (".h5", ".hdf5") else "SavedModel")
    
    return {
        "model_name": model_filename,
        "model_format": model_format,
        "supported_classes": class_names,
        "image_size": settings.image_size,
        "confidence_threshold": settings.confidence_threshold,
        "model_loaded_status": model_loaded
    }

@app.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Get Application Performance Metrics",
    description="Returns metrics tracking overall request counts, successful predictions, failed predictions, uptime, and average inference latency.",
    response_description="A JSON object holding latency and outcome metrics."
)
def get_metrics(
    request: Request,
    logger: logging.Logger = Depends(get_logger),
    metrics: DiagnosticsMetrics = Depends(get_metrics_tracker)
):
    """
    Retrieve application diagnostics and prediction performance counters.
    """
    logger.info(
        "Metrics endpoint accessed",
        extra={
            "event": "metrics_check",
            "endpoint": request.url.path
        }
    )
    return {
        "total_predictions": metrics.total_predictions,
        "average_inference_time_ms": metrics.average_inference_time,
        "uptime_seconds": round(metrics.uptime, 2),
        "successful_predictions": metrics.successful_predictions,
        "failed_predictions": metrics.failed_predictions
    }

@app.post(
    "/predict",
    dependencies=[Depends(check_rate_limit_dependency)],
    response_model=PredictionResponse,
    summary="Predict Snake Species from Image",
    description=(
        "Uploads a picture of a snake and runs deep-learning classification.\n\n"
        "**Validations Executed**:\n"
        "- **File Existence**: Verifies the payload is not empty.\n"
        f"- **MIME Type**: Restricts uploads to standard images ({', '.join(settings.allowed_mime_types)}).\n"
        f"- **Max File Size**: Caps uploads at {settings.max_upload_size / (1024 * 1024):.1f}MB.\n\n"
        "**Inference Pipeline**:\n"
        "1. Decodes and scales input using MobileNetV2 preprocessing parameters.\n"
        "2. Performs prediction on the loaded neural network model.\n"
        "3. Fetches taxonomic description, danger class, and emergency first-aid guidelines."
    ),
    response_description="Enriched snake prediction detailing species, confidence, scientific name, and first-aid recommendations.",
    responses={
        200: {
            "model": PredictionResponse,
            "description": "Inference successfully completed."
        },
        400: {
            "model": ErrorResponse,
            "description": "Bad Request. Uploaded file is empty, corrupted, or has invalid data format."
        },
        413: {
            "model": ErrorResponse,
            "description": f"Payload Too Large. Uploaded image file size exceeds the {settings.max_upload_size / (1024 * 1024):.1f}MB limit."
        },
        415: {
            "model": ErrorResponse,
            "description": "Unsupported Media Type. Uploaded file format or MIME type is not allowed."
        },
        500: {
            "model": ErrorResponse,
            "description": "Internal Server Error. Encountered unhandled exception during processing or inference."
        },
        503: {
            "model": ErrorResponse,
            "description": "Service Unavailable. Model has not been initialized or required files are missing."
        }
    }
)
async def predict_species(
    request: Request,
    file: UploadFile = File(..., description="The binary image of the snake to classify (JPEG, PNG, or WebP format)."),
    settings: Settings = Depends(get_settings),
    logger: logging.Logger = Depends(get_logger),
    metrics: DiagnosticsMetrics = Depends(get_metrics_tracker),
    model: tf.keras.Model = Depends(get_model),
    class_names: List[str] = Depends(get_class_names),
    calibrator = Depends(get_calibrator)
):
    """
    Accepts an uploaded image of a snake and predicts its species.
    Validates file existence, MIME type, and size limits before running inference.
    """
    start_time = time.perf_counter()
    try:
        # Read and validate file size on-the-fly to prevent memory exhaustion
        image_bytes = await read_and_validate_size(file, settings.max_upload_size, request)
        
        # Validate the uploaded image file using the dedicated validation module
        validate_uploaded_image(file, image_bytes, settings)
            
        # 4. Preprocess image
        preprocess_start = time.perf_counter()
        preprocessed = await run_in_threadpool(predictor.preprocess_image, image_bytes)
        preprocess_duration = time.perf_counter() - preprocess_start
        
        # 5. Run prediction
        inference_start = time.perf_counter()
        raw_predictions = await run_in_threadpool(predictor.predict, model, preprocessed)
        inference_duration = time.perf_counter() - inference_start
        
        # 6. Extract predicted class and confidence
        predicted_species, confidence = predictor.format_prediction_results(raw_predictions, class_names)
        
        # Classify prediction confidence level using calibrator
        confidence_level = calibrator.classify_confidence(confidence)
        
        # 7. Retrieve Safety and Taxonomic Metadata
        metadata_start = time.perf_counter()
        meta_dict = get_snake_metadata(predicted_species)
        metadata_duration = time.perf_counter() - metadata_start
        
        total_duration = time.perf_counter() - start_time
        
        # Round timings to milliseconds
        preprocess_time_ms = round(preprocess_duration * 1000, 2)
        inference_time_ms = round(inference_duration * 1000, 2)
        metadata_lookup_time_ms = round(metadata_duration * 1000, 2)
        total_request_duration_ms = round(total_duration * 1000, 2)
        
        # Record successful prediction metrics
        metrics.record_prediction(inference_time_ms, success=True)
        
        # Log structured prediction request details
        logger.info(
            f"Prediction completed for {file.filename}: {predicted_species} ({confidence:.4f}, {confidence_level})",
            extra={
                "event": "prediction_request",
                "uploaded_filename": file.filename,
                "file_size_bytes": len(image_bytes),
                "prediction": predicted_species,
                "confidence": confidence,
                "confidence_level": confidence_level,
                "preprocessing_time_ms": preprocess_time_ms,
                "inference_time_ms": inference_time_ms,
                "metadata_lookup_time_ms": metadata_lookup_time_ms,
                "total_request_duration_ms": total_request_duration_ms
            }
        )
        
        return PredictionResponse(
            species=predicted_species,
            confidence=confidence,
            confidence_level=confidence_level,
            metadata=meta_dict,
            inference_time_ms=inference_time_ms
        )
    except Exception as e:
        metrics.record_prediction(0.0, success=False)
        raise e