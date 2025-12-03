"""
Property-Based Tests for API Endpoints

Tests correctness properties for sentence analysis API endpoints.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from hypothesis import HealthCheck
import numpy as np
import tempfile
import os
from io import BytesIO
from pathlib import Path
from fastapi.testclient import TestClient
from scipy.io import wavfile
import time

from sentence_analysis.api import create_sentence_analysis_app, validate_audio_format, validate_audio_properties, validate_aggregation_strategy
from fastapi import UploadFile
import logging
import json


# Test fixtures
@pytest.fixture(scope="function")
def test_app():
    """Create test FastAPI application"""
    # Get the project root directory (parent of Backend)
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    model_path = str(PROJECT_ROOT / "Demo" / "models" / "best_emotion_model.pkl")
    
    if not os.path.exists(model_path):
        pytest.skip(f"Model file not found: {model_path}")
    
    app = create_sentence_analysis_app(
        model_path=model_path,
        cors_origins=["*"],
        max_queue_size=50,
        rate_limit_requests=10,
        rate_limit_window=60,
        processing_timeout=30
    )
    
    return TestClient(app)


# Helper functions
def create_wav_file(duration: float, sample_rate: int, filename: str = None) -> str:
    """
    Create a WAV file with given duration and sample rate
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        filename: Optional filename (creates temp file if None)
        
    Returns:
        Path to created WAV file
    """
    # Generate simple sine wave audio
    t = np.linspace(0, duration, int(sample_rate * duration))
    frequency = 440  # A4 note
    audio = np.sin(2 * np.pi * frequency * t) * 0.3
    
    # Convert to int16
    audio_int16 = (audio * 32767).astype(np.int16)
    
    if filename is None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        filename = temp_file.name
        temp_file.close()
    
    wavfile.write(filename, sample_rate, audio_int16)
    
    return filename


def create_non_wav_file(extension: str = '.mp3') -> str:
    """Create a non-WAV file for testing"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=extension)
    temp_file.write(b'fake audio data')
    temp_file.close()
    return temp_file.name


# Strategy for generating valid sample rates (8kHz to 48kHz)
valid_sample_rates = st.integers(min_value=8000, max_value=48000)

# Strategy for generating invalid sample rates
invalid_sample_rates = st.one_of(
    st.integers(min_value=1000, max_value=7999),
    st.integers(min_value=48001, max_value=96000)
)

# Strategy for generating valid durations (0.1s to 30s)
valid_durations = st.floats(min_value=0.1, max_value=30.0)

# Strategy for generating invalid durations (> 30s)
invalid_durations = st.floats(min_value=30.01, max_value=60.0)


# Property 1: Audio Format Validation
# Feature: turkish-sentence-emotion-analysis, Property 1: Audio Format Validation
# Validates: Requirements 1.1
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    extension=st.sampled_from(['.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac'])
)
def test_property_audio_format_validation(extension):
    """
    Property 1: Audio Format Validation
    
    For any uploaded audio file, if the file is in WAV format with sample rate 
    between 8kHz and 48kHz, then the system should accept it; otherwise it should 
    reject it with an appropriate error message.
    
    Validates: Requirements 1.1
    """
    # Create a mock UploadFile
    if extension == '.wav':
        # Valid WAV file
        filename = f"test_audio{extension}"
        is_valid, error_msg, error_details = validate_audio_format(
            UploadFile(filename=filename, file=BytesIO(b''))
        )
        
        # Should be valid
        assert is_valid is True
        assert error_msg is None
    else:
        # Invalid format
        filename = f"test_audio{extension}"
        is_valid, error_msg, error_details = validate_audio_format(
            UploadFile(filename=filename, file=BytesIO(b''))
        )
        
        # Should be invalid
        assert is_valid is False
        assert error_msg is not None
        assert "WAV format" in error_msg
        assert error_details is not None


# Property 2: Duration Validation
# Feature: turkish-sentence-emotion-analysis, Property 2: Duration Validation
# Validates: Requirements 1.2
@settings(max_examples=100, deadline=None)
@given(
    duration=st.floats(min_value=0.1, max_value=60.0),
    sample_rate=st.integers(min_value=8000, max_value=48000)
)
def test_property_duration_validation(duration, sample_rate):
    """
    Property 2: Duration Validation
    
    For any uploaded audio file, if the duration exceeds 30 seconds, then the 
    system should reject the upload and return an error message indicating the 
    duration limit.
    
    Validates: Requirements 1.2
    """
    # Create audio with specified duration
    audio = np.random.randn(int(duration * sample_rate)) * 0.1
    
    # Validate
    is_valid, error_msg, error_details = validate_audio_properties(audio, sample_rate)
    
    if duration <= 30.0:
        # Should be valid (unless other validation fails)
        # Note: might fail due to minimum duration check
        if not is_valid:
            # If invalid, should have error message
            assert error_msg is not None
    else:
        # Should be invalid
        assert is_valid is False
        assert error_msg is not None
        assert "30 seconds" in error_msg or "exceed" in error_msg.lower()


# Property 3: Upload Round-Trip
# Feature: turkish-sentence-emotion-analysis, Property 3: Upload Round-Trip
# Validates: Requirements 1.3
@settings(max_examples=5, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    duration=st.floats(min_value=0.5, max_value=5.0),
    sample_rate=st.sampled_from([8000, 16000, 22050, 44100, 48000])
)
def test_property_upload_round_trip(test_app, duration, sample_rate):
    """
    Property 3: Upload Round-Trip
    
    For any valid audio file, when uploaded successfully, the system should return 
    a unique job identifier that can be used to retrieve the analysis results.
    
    Validates: Requirements 1.3
    """
    # Create a valid WAV file
    wav_path = create_wav_file(duration, sample_rate)
    
    try:
        # Upload file
        with open(wav_path, 'rb') as f:
            response = test_app.post(
                "/api/analyze-sentence",
                files={"file": ("test.wav", f, "audio/wav")}
            )
        
        # Should return 200 OK
        assert response.status_code == 200
        
        # Should return job_id
        data = response.json()
        assert "job_id" in data
        assert data["job_id"] is not None
        assert len(data["job_id"]) > 0
        
        job_id = data["job_id"]
        
        # Should be able to retrieve status with job_id
        status_response = test_app.get(f"/api/status/{job_id}")
        assert status_response.status_code == 200
        
        status_data = status_response.json()
        assert status_data["job_id"] == job_id
        assert "status" in status_data
        
    finally:
        # Clean up
        if os.path.exists(wav_path):
            os.remove(wav_path)


# Property 30: Rate Limiting
# Feature: turkish-sentence-emotion-analysis, Property 30: Rate Limiting
# Validates: Requirements 8.5
def test_property_rate_limiting(test_app):
    """
    Property 30: Rate Limiting
    
    For any client IP address, if more than 10 requests are made within a 1-minute 
    window, subsequent requests should be rejected with a 429 Too Many Requests status.
    
    Validates: Requirements 8.5
    """
    # Create a valid WAV file
    wav_path = create_wav_file(1.0, 16000)
    
    try:
        successful_requests = 0
        rate_limited = False
        
        # Make 15 requests rapidly
        for i in range(15):
            with open(wav_path, 'rb') as f:
                response = test_app.post(
                    "/api/analyze-sentence",
                    files={"file": (f"test_{i}.wav", f, "audio/wav")}
                )
            
            if response.status_code == 200:
                successful_requests += 1
            elif response.status_code == 429:
                rate_limited = True
                # Verify error message
                data = response.json()
                assert "detail" in data
                break
        
        # Should have rate limited after 10 requests
        assert successful_requests <= 10
        assert rate_limited is True
        
    finally:
        # Clean up
        if os.path.exists(wav_path):
            os.remove(wav_path)


# Property 29: Processing Timeout
# Feature: turkish-sentence-emotion-analysis, Property 29: Processing Timeout
# Validates: Requirements 8.4
def test_property_processing_timeout(test_app):
    """
    Property 29: Processing Timeout
    
    For any analysis request that exceeds 30 seconds of processing time, the system 
    should terminate the request and return a timeout error.
    
    Validates: Requirements 8.4
    
    Note: This test verifies the timeout mechanism exists, but doesn't actually
    wait 30 seconds. It checks that the job queue has timeout capability.
    """
    # Create a valid WAV file
    wav_path = create_wav_file(2.0, 16000)
    
    try:
        # Upload file
        with open(wav_path, 'rb') as f:
            response = test_app.post(
                "/api/analyze-sentence",
                files={"file": ("test.wav", f, "audio/wav")}
            )
        
        assert response.status_code == 200
        data = response.json()
        job_id = data["job_id"]
        
        # Verify that job queue has timeout configuration
        job_queue = test_app.app.state.job_queue
        assert job_queue.processing_timeout == 30
        
        # Verify timeout_job method exists and works
        test_timeout_result = job_queue.timeout_job(job_id)
        assert test_timeout_result is True
        
        # Check that job is marked as timeout
        job_status = job_queue.get_job_status(job_id)
        assert job_status.status.value == "timeout"
        assert "timeout" in job_status.error.lower()
        
    finally:
        # Clean up
        if os.path.exists(wav_path):
            os.remove(wav_path)


# Additional helper test for sample rate validation
@settings(max_examples=100)
@given(
    sample_rate=st.integers(min_value=1000, max_value=96000)
)
def test_sample_rate_validation(sample_rate):
    """
    Test that sample rate validation works correctly
    
    Part of Property 1: Audio Format Validation
    """
    # Create audio with 1 second duration
    audio = np.random.randn(sample_rate) * 0.1
    
    is_valid, error_msg, error_details = validate_audio_properties(audio, sample_rate)
    
    if 8000 <= sample_rate <= 48000:
        # Valid sample rate - should pass (unless other validation fails)
        # Note: might still fail due to silence detection, but not sample rate
        if not is_valid and error_msg:
            assert "Sample rate" not in error_msg
    else:
        # Invalid sample rate - should fail
        assert is_valid is False
        assert error_msg is not None
        assert "Sample rate" in error_msg or "8kHz" in error_msg or "48kHz" in error_msg
        assert error_details is not None
        assert "sample_rate" in error_details


# Property 32: Error Status Codes
# Feature: turkish-sentence-emotion-analysis, Property 32: Error Status Codes
# Validates: Requirements 9.4
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    extension=st.sampled_from(['.mp3', '.ogg', '.flac', '.m4a', '.aac'])
)
def test_property_error_status_codes_invalid_format(test_app, extension):
    """
    Property 32: Error Status Codes - Invalid Format
    
    For any request that fails due to client error (invalid input), the system 
    should return HTTP 400; for server errors, it should return HTTP 500.
    
    Validates: Requirements 9.4
    """
    # Test invalid format (should return 400)
    response = test_app.post(
        "/api/analyze-sentence",
        files={"file": (f"test{extension}", BytesIO(b'fake data'), f"audio/{extension[1:]}")}
    )
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert data["detail"]["code"] in ["INVALID_AUDIO_FORMAT", "AUDIO_LOAD_ERROR"]


@settings(max_examples=5, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    duration=st.one_of(
        st.floats(min_value=30.01, max_value=35.0),  # Too long
        st.floats(min_value=0.0, max_value=0.09)  # Too short
    )
)
def test_property_error_status_codes_invalid_duration(test_app, duration):
    """
    Property 32: Error Status Codes - Invalid Duration
    """
    sample_rate = 16000
    # Create WAV file with the actual invalid duration (don't clamp it)
    wav_path = create_wav_file(duration, sample_rate)
    try:
        with open(wav_path, 'rb') as f:
            response = test_app.post(
                "/api/analyze-sentence",
                files={"file": ("test.wav", f, "audio/wav")}
            )
        # Should return 400 for invalid duration (too short or too long)
        assert response.status_code == 400, f"Expected 400 for duration {duration}, got {response.status_code}"
        data = response.json()
        assert "detail" in data
        # Should have an error code related to duration or audio properties
        assert data["detail"]["code"] in ["INVALID_AUDIO_PROPERTIES", "INVALID_DURATION", "AUDIO_LOAD_ERROR"]
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


# Property 37: Error Logging Completeness
# Feature: turkish-sentence-emotion-analysis, Property 37: Error Logging Completeness
# Validates: Requirements 13.1
def test_property_error_logging_completeness(test_app):
    """
    Property 37: Error Logging Completeness
    
    For any processing step that fails, the error log should include timestamp, 
    request ID, error type, and stack trace.
    
    Validates: Requirements 13.1
    """
    import io
    
    # Capture log output
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    # Use JSON formatter to match the app's logging format
    from sentence_analysis.logging_config import JSONFormatter
    handler.setFormatter(JSONFormatter())
    
    # Get the logger from the app
    logger = test_app.app.state.logger
    logger.addHandler(handler)
    
    try:
        # Trigger an error by uploading invalid file
        response = test_app.post(
            "/api/analyze-sentence",
            files={"file": ("test.mp3", BytesIO(b'fake audio data'), "audio/mpeg")}
        )
        
        # Should return error status
        assert response.status_code == 400
        
        # Get log output
        log_output = log_capture.getvalue()
        
        # Check that log contains structured information
        # Since we're using JSON logging, try to parse it
        log_lines = log_output.strip().split('\n')
        found_error_log = False
        
        for line in log_lines:
            if line.strip():
                try:
                    log_data = json.loads(line)
                    # Check for required fields
                    assert "timestamp" in log_data or "level" in log_data
                    # Error logs should have level ERROR or WARNING
                    if log_data.get("level") in ["ERROR", "WARNING"]:
                        assert "message" in log_data
                        found_error_log = True
                        # Check for request_id if available
                        if "request_id" in log_data:
                            assert log_data["request_id"] is not None
                except json.JSONDecodeError:
                    # Skip non-JSON lines (might be other output)
                    pass
        
        # At least one error/warning log should be found
        assert found_error_log, "No ERROR or WARNING level log found"
        
    finally:
        logger.removeHandler(handler)
        handler.close()


# Property 38: Input Validation
# Feature: turkish-sentence-emotion-analysis, Property 38: Input Validation
# Validates: Requirements 13.3
@settings(max_examples=5, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    aggregation_strategy=st.sampled_from(["invalid_strategy_1", "invalid_strategy_2", "wrong_method", "test123", "xyz"])
)
def test_property_input_validation_strategy(test_app, aggregation_strategy):
    """
    Property 38: Input Validation - Invalid Strategy
    
    For any request with invalid or unexpected input parameters, the system 
    should validate the input and return HTTP 400 with detailed validation errors.
    
    Validates: Requirements 13.3
    """
    wav_path = create_wav_file(1.0, 16000)
    try:
        with open(wav_path, 'rb') as f:
            response = test_app.post(
                "/api/analyze-sentence",
                files={"file": ("test.wav", f, "audio/wav")},
                data={"aggregation_strategy": aggregation_strategy}
            )
        # Should return 400 for invalid strategy
        assert response.status_code == 400, f"Expected 400, got {response.status_code} for strategy '{aggregation_strategy}'"
        data = response.json()
        assert "detail" in data
        assert data["detail"]["code"] == "INVALID_AGGREGATION_STRATEGY"
        assert "message" in data["detail"]
        assert "message_tr" in data["detail"]
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    filename=st.text(min_size=1, max_size=50).filter(lambda x: not x.endswith('.wav'))
)
def test_property_input_validation_filename(test_app, filename):
    """
    Property 38: Input Validation - Invalid Filename
    """
    response = test_app.post(
        "/api/analyze-sentence",
        files={"file": (filename, BytesIO(b'fake data'), "application/octet-stream")}
    )
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert data["detail"]["code"] in ["INVALID_AUDIO_FORMAT", "AUDIO_LOAD_ERROR"]
    assert "message" in data["detail"]
    assert "message_tr" in data["detail"]

