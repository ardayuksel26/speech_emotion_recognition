"""
API Endpoints for Sentence Analysis

Provides REST API endpoints for Turkish sentence emotion analysis.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from typing import Optional, Dict, List, Any
import librosa
import numpy as np
import tempfile
import os
import asyncio
import uuid
import traceback
from datetime import datetime

from .audio_segmenter import AudioSegmenter
from .feature_extractor import FeatureExtractor
from .word_predictor import WordLevelPredictor, PredictionResult
from .aggregation_engine import AggregationEngine
from .job_queue import JobQueueManager, JobStatus
from .logging_config import setup_logging, log_with_context, log_error, log_performance


# Response Models
class AnalyzeSentenceResponse(BaseModel):
    """Response for analyze-sentence endpoint"""
    job_id: str
    status: str
    message: str


class WordPredictionResponse(BaseModel):
    """Word-level prediction in API response"""
    word_index: int
    start_time: float
    end_time: float
    emotion: str
    confidence: float
    probabilities: Dict[str, float]
    is_uncertain: bool


class AnalysisMetadataResponse(BaseModel):
    """Analysis metadata in API response"""
    num_words: int
    audio_duration: float
    sample_rate: int
    aggregation_strategy: str
    has_mixed_emotions: bool


class SentenceAnalysisResult(BaseModel):
    """Complete sentence analysis result"""
    primary_emotion: str
    confidence: float
    probabilities: Dict[str, float]
    is_mixed: bool
    secondary_emotion: Optional[str]
    word_predictions: List[WordPredictionResponse]
    metadata: AnalysisMetadataResponse
    processing_time: float


class JobStatusResponse(BaseModel):
    """Response for job status endpoint"""
    job_id: str
    status: str
    progress: float
    result: Optional[SentenceAnalysisResult] = None
    error: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response format with bilingual messages"""
    code: str
    message: str
    message_tr: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str
    request_id: Optional[str] = None


# Error code mappings
ERROR_CODES = {
    "INVALID_AUDIO_FORMAT": {
        "message": "The uploaded file must be in WAV format",
        "message_tr": "Yüklenen dosya WAV formatında olmalıdır",
        "status_code": 400
    },
    "INVALID_AUDIO_PROPERTIES": {
        "message": "Audio properties are invalid",
        "message_tr": "Ses dosyası özellikleri geçersiz",
        "status_code": 400
    },
    "INVALID_SAMPLE_RATE": {
        "message": "Sample rate must be between 8kHz and 48kHz",
        "message_tr": "Örnekleme hızı 8kHz ile 48kHz arasında olmalıdır",
        "status_code": 400
    },
    "INVALID_DURATION": {
        "message": "Audio duration must not exceed 30 seconds",
        "message_tr": "Ses dosyası süresi 30 saniyeyi geçmemelidir",
        "status_code": 400
    },
    "INVALID_AGGREGATION_STRATEGY": {
        "message": "Invalid aggregation strategy. Must be one of: weighted_average, majority_voting, temporal_weighted, confidence_threshold",
        "message_tr": "Geçersiz birleştirme stratejisi. Şunlardan biri olmalıdır: weighted_average, majority_voting, temporal_weighted, confidence_threshold",
        "status_code": 400
    },
    "RATE_LIMIT_EXCEEDED": {
        "message": "Rate limit exceeded",
        "message_tr": "İstek limiti aşıldı",
        "status_code": 429
    },
    "QUEUE_FULL": {
        "message": "Queue is full, please try again later",
        "message_tr": "Kuyruk dolu, lütfen daha sonra tekrar deneyin",
        "status_code": 503
    },
    "JOB_NOT_FOUND": {
        "message": "Job not found",
        "message_tr": "İş bulunamadı",
        "status_code": 404
    },
    "INTERNAL_ERROR": {
        "message": "Internal server error",
        "message_tr": "Sunucu hatası",
        "status_code": 500
    },
    "PROCESSING_ERROR": {
        "message": "Error during processing",
        "message_tr": "İşleme sırasında hata oluştu",
        "status_code": 500
    },
    "MODEL_LOAD_ERROR": {
        "message": "Failed to load model",
        "message_tr": "Model yüklenemedi",
        "status_code": 503
    },
    "AUDIO_LOAD_ERROR": {
        "message": "Failed to load audio file",
        "message_tr": "Ses dosyası yüklenemedi",
        "status_code": 400
    },
    "SEGMENTATION_ERROR": {
        "message": "Failed to segment audio",
        "message_tr": "Ses bölümlendirilemedi",
        "status_code": 500
    },
    "FEATURE_EXTRACTION_ERROR": {
        "message": "Failed to extract features",
        "message_tr": "Özellik çıkarılamadı",
        "status_code": 500
    },
    "PREDICTION_ERROR": {
        "message": "Failed to predict emotion",
        "message_tr": "Duygu tahmin edilemedi",
        "status_code": 500
    },
    "AGGREGATION_ERROR": {
        "message": "Failed to aggregate predictions",
        "message_tr": "Tahminler birleştirilemedi",
        "status_code": 500
    }
}


def create_error_response(
    error_code: str,
    request_id: Optional[str] = None,
    details: Optional[Dict] = None
) -> Dict:
    """
    Create standardized error response
    
    Args:
        error_code: Error code from ERROR_CODES
        request_id: Optional request ID
        details: Optional additional details
        
    Returns:
        Error response dictionary
    """
    if error_code not in ERROR_CODES:
        error_code = "INTERNAL_ERROR"
    
    error_info = ERROR_CODES[error_code]
    
    response = {
        "code": error_code,
        "message": error_info["message"],
        "message_tr": error_info["message_tr"],
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    
    if request_id:
        response["request_id"] = request_id
    
    if details:
        response["details"] = details
    
    return response


# Audio validation functions
def validate_audio_format(file: UploadFile) -> tuple[bool, Optional[str], Optional[Dict]]:
    """
    Validate audio file format
    
    Args:
        file: Uploaded file
        
    Returns:
        Tuple of (is_valid, error_message, error_details)
    """
    # Check if filename exists
    if not file.filename:
        return False, "Filename is required", {"received": None, "expected": "*.wav"}
    
    # Check file extension
    if not file.filename.lower().endswith('.wav'):
        return False, "The uploaded file must be in WAV format", {
            "received_format": file.filename.split('.')[-1] if '.' in file.filename else "unknown",
            "expected_format": "wav"
        }
    
    return True, None, None


def validate_audio_properties(audio: np.ndarray, sr: int) -> tuple[bool, Optional[str], Optional[Dict]]:
    """
    Validate audio properties (sample rate, duration)
    
    Args:
        audio: Audio signal
        sr: Sample rate
        
    Returns:
        Tuple of (is_valid, error_message, error_details)
    """
    error_details = {}
    
    # Check sample rate (8kHz to 48kHz)
    if sr < 8000 or sr > 48000:
        error_details["sample_rate"] = {
            "received": sr,
            "min_allowed": 8000,
            "max_allowed": 48000
        }
        return False, "Sample rate must be between 8kHz and 48kHz", error_details
    
    # Check duration (max 30 seconds)
    duration = len(audio) / sr
    if duration > 30.0:
        error_details["duration"] = {
            "received": round(duration, 2),
            "max_allowed": 30.0
        }
        return False, "Audio duration must not exceed 30 seconds", error_details
    
    # Check minimum duration (at least 0.1 seconds)
    if duration < 0.1:
        error_details["duration"] = {
            "received": round(duration, 2),
            "min_allowed": 0.1
        }
        return False, "Audio duration must be at least 0.1 seconds", error_details
    
    # Check for empty audio
    if len(audio) == 0:
        return False, "Audio file is empty", {"received_length": 0}
    
    # Check for excessive silence (>50% of duration) - warning only
    try:
        rms = librosa.feature.rms(y=audio)[0]
        silence_threshold = 0.01
        silence_ratio = np.sum(rms < silence_threshold) / len(rms)
        
        if silence_ratio > 0.5:
            # Warning, but not rejection - return as valid with warning info
            return True, None, {"warning": f"Audio contains {silence_ratio:.1%} silence"}
    except Exception:
        # If RMS calculation fails, continue anyway
        pass
    
    return True, None, None


def validate_aggregation_strategy(strategy: str) -> tuple[bool, Optional[str]]:
    """
    Validate aggregation strategy
    
    Args:
        strategy: Aggregation strategy name
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    valid_strategies = ["weighted_average", "majority_voting", "temporal_weighted", "confidence_threshold"]
    
    if strategy not in valid_strategies:
        return False, f"Invalid aggregation strategy. Must be one of: {', '.join(valid_strategies)}"
    
    return True, None


# Sentence analysis processor
async def process_sentence_analysis(
    job_id: str,
    audio_path: str,
    job_queue: JobQueueManager,
    segmenter: AudioSegmenter,
    feature_extractor: FeatureExtractor,
    predictor: WordLevelPredictor,
    aggregator: AggregationEngine,
    aggregation_strategy: str = "weighted_average",
    logger=None,
    request_id: Optional[str] = None
):
    """
    Process sentence analysis asynchronously with comprehensive logging and error handling
    
    Args:
        job_id: Job identifier
        audio_path: Path to temporary audio file
        job_queue: JobQueueManager instance
        segmenter: AudioSegmenter instance
        feature_extractor: FeatureExtractor instance
        predictor: WordLevelPredictor instance
        aggregator: AggregationEngine instance
        aggregation_strategy: Aggregation strategy to use
        logger: Logger instance
        request_id: Optional request ID for logging
    """
    start_time = datetime.now()
    logger = logger or setup_logging()
    
    try:
        # Mark as processing
        log_with_context(
            logger, "info",
            f"Starting sentence analysis for job {job_id}",
            request_id=request_id,
            job_id=job_id,
            context={"aggregation_strategy": aggregation_strategy}
        )
        job_queue.start_processing(job_id)
        
        # Load audio
        try:
            log_with_context(
                logger, "debug",
                f"Loading audio file: {audio_path}",
                request_id=request_id,
                job_id=job_id
            )
            audio, sr = librosa.load(audio_path, sr=None)
            audio_duration = len(audio) / sr
            
            log_with_context(
                logger, "info",
                f"Audio loaded: duration={audio_duration:.2f}s, sample_rate={sr}Hz",
                request_id=request_id,
                job_id=job_id,
                metrics={"audio_duration": audio_duration, "sample_rate": sr}
            )
        except Exception as e:
            error_msg = f"Failed to load audio file: {str(e)}"
            log_error(logger, error_msg, e, request_id=request_id, job_id=job_id)
            job_queue.fail_job(job_id, error_msg)
            return
        
        job_queue.update_progress(job_id, 0.2)
        
        # Segment audio
        try:
            log_with_context(
                logger, "debug",
                "Segmenting audio into words",
                request_id=request_id,
                job_id=job_id
            )
            segments = segmenter.segment_audio(audio, sr)
            
            log_with_context(
                logger, "info",
                f"Audio segmented into {len(segments)} segments",
                request_id=request_id,
                job_id=job_id,
                metrics={"num_segments": len(segments)}
            )
        except Exception as e:
            error_msg = f"Failed to segment audio: {str(e)}"
            log_error(logger, error_msg, e, request_id=request_id, job_id=job_id)
            job_queue.fail_job(job_id, error_msg)
            return
        
        job_queue.update_progress(job_id, 0.4)
        
        # Extract features and predict for each segment
        word_predictions = []
        segment_start_time = datetime.now()
        
        try:
            for idx, segment in enumerate(segments):
                try:
                    # Extract features
                    features = feature_extractor.extract_features(
                        segment.audio_data,
                        segment.sample_rate,
                        apply_scaling=True
                    )
                    
                    # Predict emotion
                    prediction = predictor.predict(features)
                    
                    # Store word prediction
                    word_predictions.append({
                        'word_index': segment.word_index,
                        'start_time': segment.start_time,
                        'end_time': segment.end_time,
                        'emotion': prediction.emotion,
                        'confidence': prediction.confidence,
                        'probabilities': prediction.probabilities,
                        'is_uncertain': prediction.is_uncertain
                    })
                    
                    log_with_context(
                        logger, "debug",
                        f"Segment {idx+1}/{len(segments)}: emotion={prediction.emotion}, confidence={prediction.confidence:.2f}",
                        request_id=request_id,
                        job_id=job_id,
                        context={"segment_index": idx, "emotion": prediction.emotion}
                    )
                except Exception as e:
                    log_error(
                        logger,
                        f"Failed to process segment {idx+1}: {str(e)}",
                        e,
                        request_id=request_id,
                        job_id=job_id,
                        context={"segment_index": idx}
                    )
                    # Continue with other segments, but mark as uncertain
                    word_predictions.append({
                        'word_index': idx,
                        'start_time': segment.start_time,
                        'end_time': segment.end_time,
                        'emotion': 'calm',  # Default fallback
                        'confidence': 0.25,
                        'probabilities': {'angry': 0.25, 'calm': 0.25, 'happy': 0.25, 'sad': 0.25},
                        'is_uncertain': True
                    })
            
            segment_time = (datetime.now() - segment_start_time).total_seconds()
            log_performance(
                logger,
                "segment_processing",
                segment_time,
                request_id=request_id,
                job_id=job_id,
                additional_metrics={"num_segments": len(segments)}
            )
        except Exception as e:
            error_msg = f"Failed during feature extraction or prediction: {str(e)}"
            log_error(logger, error_msg, e, request_id=request_id, job_id=job_id)
            job_queue.fail_job(job_id, error_msg)
            return
        
        job_queue.update_progress(job_id, 0.7)
        
        # Aggregate predictions
        try:
            log_with_context(
                logger, "debug",
                f"Aggregating predictions using strategy: {aggregation_strategy}",
                request_id=request_id,
                job_id=job_id
            )
            
            prediction_results = [
                PredictionResult(
                    emotion=wp['emotion'],
                    confidence=wp['confidence'],
                    probabilities=wp['probabilities'],
                    is_uncertain=wp['is_uncertain']
                )
                for wp in word_predictions
            ]
            
            sentence_prediction = aggregator.aggregate(
                prediction_results,
                strategy=aggregation_strategy
            )
            
            log_with_context(
                logger, "info",
                f"Aggregation complete: primary_emotion={sentence_prediction.primary_emotion}, confidence={sentence_prediction.confidence:.2f}",
                request_id=request_id,
                job_id=job_id,
                context={
                    "primary_emotion": sentence_prediction.primary_emotion,
                    "confidence": sentence_prediction.confidence,
                    "is_mixed": sentence_prediction.is_mixed
                }
            )
        except Exception as e:
            error_msg = f"Failed to aggregate predictions: {str(e)}"
            log_error(logger, error_msg, e, request_id=request_id, job_id=job_id)
            job_queue.fail_job(job_id, error_msg)
            return
        
        job_queue.update_progress(job_id, 0.9)
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Build result
        result = {
            'primary_emotion': sentence_prediction.primary_emotion,
            'confidence': sentence_prediction.confidence,
            'probabilities': sentence_prediction.probabilities,
            'is_mixed': sentence_prediction.is_mixed,
            'secondary_emotion': sentence_prediction.secondary_emotion,
            'word_predictions': word_predictions,
            'metadata': {
                'num_words': len(segments),
                'audio_duration': audio_duration,
                'sample_rate': sr,
                'aggregation_strategy': aggregation_strategy,
                'has_mixed_emotions': sentence_prediction.is_mixed
            },
            'processing_time': processing_time
        }
        
        # Mark as completed
        job_queue.complete_job(job_id, result)
        
        # Log success with metrics
        log_performance(
            logger,
            "sentence_analysis",
            processing_time,
            request_id=request_id,
            job_id=job_id,
            additional_metrics={
                "num_words": len(segments),
                "primary_emotion": sentence_prediction.primary_emotion,
                "confidence": sentence_prediction.confidence,
                "aggregation_strategy": aggregation_strategy
            }
        )
        
        log_with_context(
            logger, "info",
            f"Analysis completed successfully for job {job_id}",
            request_id=request_id,
            job_id=job_id,
            context={"processing_time": processing_time}
        )
        
    except Exception as e:
        # Mark as failed with detailed error logging
        error_msg = f"Processing failed: {str(e)}"
        log_error(
            logger,
            error_msg,
            e,
            request_id=request_id,
            job_id=job_id,
            context={"traceback": traceback.format_exc()}
        )
        job_queue.fail_job(job_id, error_msg)
    
    finally:
        # Clean up temporary file
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                log_with_context(
                    logger, "debug",
                    f"Cleaned up temporary file: {audio_path}",
                    request_id=request_id,
                    job_id=job_id
                )
            except Exception as e:
                log_error(
                    logger,
                    f"Failed to clean up temporary file: {audio_path}",
                    e,
                    request_id=request_id,
                    job_id=job_id
                )


def create_sentence_analysis_app(
    model_path: str,
    cors_origins: List[str] = ["*"],
    max_queue_size: int = 50,
    rate_limit_requests: int = 10,
    rate_limit_window: int = 60,
    processing_timeout: int = 30
) -> FastAPI:
    """
    Create FastAPI application with sentence analysis endpoints
    
    Args:
        model_path: Path to trained model file
        cors_origins: List of allowed CORS origins
        max_queue_size: Maximum queue size
        rate_limit_requests: Maximum requests per window
        rate_limit_window: Rate limit window in seconds
        processing_timeout: Processing timeout in seconds
        
    Returns:
        FastAPI application instance
    """
    app = FastAPI(title="Turkish Sentence Emotion Analysis API")
    
    # Setup logging
    logger = setup_logging(level="INFO", json_format=True)
    app.state.logger = logger
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize components
    try:
        job_queue = JobQueueManager(
            max_queue_size=max_queue_size,
            rate_limit_requests=rate_limit_requests,
            rate_limit_window=rate_limit_window,
            processing_timeout=processing_timeout
        )
        
        segmenter = AudioSegmenter()
        predictor = WordLevelPredictor(model_path)
        feature_extractor = FeatureExtractor()
        feature_extractor.set_scaler(predictor.model_artifacts['scaler'])
        aggregator = AggregationEngine()
        
        log_with_context(
            logger, "info",
            "Application initialized successfully",
            context={
                "model_path": model_path,
                "max_queue_size": max_queue_size,
                "rate_limit_requests": rate_limit_requests
            }
        )
    except Exception as e:
        log_error(logger, "Failed to initialize application components", e)
        raise
    
    # Store components in app state
    app.state.job_queue = job_queue
    app.state.segmenter = segmenter
    app.state.feature_extractor = feature_extractor
    app.state.predictor = predictor
    app.state.aggregator = aggregator
    
    @app.post("/api/analyze-sentence", response_model=AnalyzeSentenceResponse)
    async def analyze_sentence(
        request: Request,
        file: UploadFile = File(...),
        aggregation_strategy: str = Form("weighted_average")
    ):
        """
        Analyze Turkish sentence audio for emotion
        
        Args:
            file: Audio file (WAV format, 8-48kHz, max 30s)
            aggregation_strategy: Aggregation strategy to use
            
        Returns:
            Job ID for status polling
        """
        # Generate request ID for tracking
        request_id = str(uuid.uuid4())
        logger = app.state.logger
        
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        log_with_context(
            logger, "info",
            f"Received analyze-sentence request from {client_ip}",
            request_id=request_id,
            context={
                "filename": file.filename,
                "aggregation_strategy": aggregation_strategy,
                "client_ip": client_ip
            }
        )
        
        # Validate aggregation strategy
        is_valid_strategy, strategy_error = validate_aggregation_strategy(aggregation_strategy)
        if not is_valid_strategy:
            error_response = create_error_response(
                "INVALID_AGGREGATION_STRATEGY",
                request_id=request_id,
                details={"received": aggregation_strategy}
            )
            log_with_context(
                logger, "warning",
                f"Invalid aggregation strategy: {aggregation_strategy}",
                request_id=request_id
            )
            raise HTTPException(
                status_code=ERROR_CODES["INVALID_AGGREGATION_STRATEGY"]["status_code"],
                detail=error_response
            )
        
        # Validate audio format
        is_valid, error_msg, error_details = validate_audio_format(file)
        if not is_valid:
            error_response = create_error_response(
                "INVALID_AUDIO_FORMAT",
                request_id=request_id,
                details=error_details
            )
            log_with_context(
                logger, "warning",
                f"Invalid audio format: {error_msg}",
                request_id=request_id,
                context=error_details
            )
            raise HTTPException(
                status_code=ERROR_CODES["INVALID_AUDIO_FORMAT"]["status_code"],
                detail=error_response
            )
        
        # Create job (checks rate limiting and queue size)
        try:
            job_id = app.state.job_queue.create_job(client_ip)
            log_with_context(
                logger, "info",
                f"Job created: {job_id}",
                request_id=request_id,
                job_id=job_id
            )
        except ValueError as e:
            error_str = str(e)
            if "Rate limit" in error_str:
                error_response = create_error_response(
                    "RATE_LIMIT_EXCEEDED",
                    request_id=request_id
                )
                log_with_context(
                    logger, "warning",
                    f"Rate limit exceeded for {client_ip}",
                    request_id=request_id,
                    context={"client_ip": client_ip}
                )
                raise HTTPException(
                    status_code=ERROR_CODES["RATE_LIMIT_EXCEEDED"]["status_code"],
                    detail=error_response
                )
            elif "Queue is full" in error_str:
                error_response = create_error_response(
                    "QUEUE_FULL",
                    request_id=request_id
                )
                log_with_context(
                    logger, "warning",
                    "Queue is full",
                    request_id=request_id
                )
                raise HTTPException(
                    status_code=ERROR_CODES["QUEUE_FULL"]["status_code"],
                    detail=error_response
                )
            else:
                error_response = create_error_response(
                    "INTERNAL_ERROR",
                    request_id=request_id
                )
                log_error(logger, f"Unexpected error creating job: {error_str}", Exception(error_str), request_id=request_id)
                raise HTTPException(
                    status_code=500,
                    detail=error_response
                )
        
        # Save uploaded file temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_path = temp_file.name
        
        try:
            # Write uploaded file
            content = await file.read()
            temp_file.write(content)
            temp_file.close()
            
            log_with_context(
                logger, "debug",
                f"File saved temporarily: {temp_path}, size: {len(content)} bytes",
                request_id=request_id,
                job_id=job_id
            )
            
            # Load and validate audio properties
            try:
                audio, sr = librosa.load(temp_path, sr=None)
            except Exception as e:
                error_msg = f"Failed to load audio file: {str(e)}"
                log_error(logger, error_msg, e, request_id=request_id, job_id=job_id)
                os.remove(temp_path)
                app.state.job_queue.fail_job(job_id, error_msg)
                error_response = create_error_response(
                    "AUDIO_LOAD_ERROR",
                    request_id=request_id,
                    details={"error": str(e)}
                )
                raise HTTPException(
                    status_code=ERROR_CODES["AUDIO_LOAD_ERROR"]["status_code"],
                    detail=error_response
                )
            
            is_valid, error_msg, error_details = validate_audio_properties(audio, sr)
            
            if not is_valid:
                # Clean up and reject
                os.remove(temp_path)
                app.state.job_queue.fail_job(job_id, error_msg)
                log_with_context(
                    logger, "warning",
                    f"Invalid audio properties: {error_msg}",
                    request_id=request_id,
                    job_id=job_id,
                    context=error_details
                )
                
                # Determine specific error code
                if error_details and "sample_rate" in error_details:
                    error_code = "INVALID_SAMPLE_RATE"
                elif error_details and "duration" in error_details:
                    error_code = "INVALID_DURATION"
                else:
                    error_code = "INVALID_AUDIO_PROPERTIES"
                
                error_response = create_error_response(
                    error_code,
                    request_id=request_id,
                    details=error_details
                )
                raise HTTPException(
                    status_code=ERROR_CODES[error_code]["status_code"],
                    detail=error_response
                )
            
            # Start async processing
            asyncio.create_task(
                process_sentence_analysis(
                    job_id=job_id,
                    audio_path=temp_path,
                    job_queue=app.state.job_queue,
                    segmenter=app.state.segmenter,
                    feature_extractor=app.state.feature_extractor,
                    predictor=app.state.predictor,
                    aggregator=app.state.aggregator,
                    aggregation_strategy=aggregation_strategy,
                    logger=logger,
                    request_id=request_id
                )
            )
            
            log_with_context(
                logger, "info",
                f"Analysis job queued successfully: {job_id}",
                request_id=request_id,
                job_id=job_id
            )
            
            return AnalyzeSentenceResponse(
                job_id=job_id,
                status="queued",
                message="Analysis job created successfully"
            )
            
        except HTTPException:
            raise
        except Exception as e:
            # Clean up on error
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            
            error_msg = f"Unexpected error: {str(e)}"
            log_error(logger, error_msg, e, request_id=request_id, job_id=job_id)
            app.state.job_queue.fail_job(job_id, error_msg)
            
            error_response = create_error_response(
                "INTERNAL_ERROR",
                request_id=request_id,
                details={"error": str(e)}
            )
            raise HTTPException(
                status_code=500,
                detail=error_response
            )
    
    @app.get("/api/status/{job_id}", response_model=JobStatusResponse)
    async def get_job_status(job_id: str, request: Request):
        """
        Get status of analysis job
        
        Args:
            job_id: Job identifier
            request: Request object
            
        Returns:
            Job status and result if completed
        """
        request_id = str(uuid.uuid4())
        logger = app.state.logger
        
        log_with_context(
            logger, "debug",
            f"Status request for job: {job_id}",
            request_id=request_id,
            job_id=job_id
        )
        
        job_status = app.state.job_queue.get_job_status(job_id)
        
        if job_status is None:
            error_response = create_error_response(
                "JOB_NOT_FOUND",
                request_id=request_id,
                details={"job_id": job_id}
            )
            log_with_context(
                logger, "warning",
                f"Job not found: {job_id}",
                request_id=request_id
            )
            raise HTTPException(
                status_code=ERROR_CODES["JOB_NOT_FOUND"]["status_code"],
                detail=error_response
            )
        
        # Build response
        response = JobStatusResponse(
            job_id=job_status.job_id,
            status=job_status.status.value,
            progress=job_status.progress,
            error=job_status.error,
            created_at=job_status.created_at.isoformat(),
            completed_at=job_status.completed_at.isoformat() if job_status.completed_at else None
        )
        
        # Add result if completed
        if job_status.status.value == "completed" and job_status.result:
            try:
                response.result = SentenceAnalysisResult(**job_status.result)
                log_with_context(
                    logger, "debug",
                    f"Job {job_id} completed successfully",
                    request_id=request_id,
                    job_id=job_id
                )
            except ValidationError as e:
                log_error(
                    logger,
                    f"Failed to validate result for job {job_id}",
                    e,
                    request_id=request_id,
                    job_id=job_id
                )
                # Return response without result if validation fails
                response.result = None
        
        return response
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        logger = app.state.logger
        
        try:
            queue_size = app.state.job_queue.get_queue_size()
            health_status = {
                "status": "healthy",
                "queue_size": queue_size,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
            
            log_with_context(
                logger, "debug",
                "Health check requested",
                context={"queue_size": queue_size}
            )
            
            return health_status
        except Exception as e:
            log_error(logger, "Health check failed", e)
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
    
    return app

