# Turkish Sentence Emotion Analysis - Backend Infrastructure

This module provides the core infrastructure for analyzing Turkish sentence-level emotions by segmenting audio, extracting features, and aggregating word-level predictions.

## Architecture

The system consists of the following components:

### 1. Audio Segmentation (`audio_segmenter.py`)
- **AudioSegmenter**: Segments sentence audio into word-level chunks
- Uses Voice Activity Detection (VAD) with energy-based thresholding
- Fallback to fixed-duration windowing when VAD fails
- Configurable silence threshold (100ms default) and segment padding (50ms default)

### 2. Feature Extraction (`feature_extractor.py`)
- **FeatureExtractor**: Extracts 378-dimensional feature vectors
- Features include:
  - MFCC: 40 coefficients × 2 (mean, std) = 80 features
  - Chroma: 12 coefficients × 2 = 24 features
  - Mel Spectrogram: 128 bands × 2 = 256 features
  - Spectral Contrast: 7 bands × 2 = 14 features
  - Zero Crossing Rate: 2 features
  - RMS Energy: 2 features

### 3. Word-Level Prediction (`word_predictor.py`)
- **WordLevelPredictor**: Applies trained Gradient Boosting model
- Returns emotion predictions with confidence scores
- Flags uncertain predictions (confidence < 0.4)
- Provides fallback uniform distribution on errors

### 4. Aggregation Engine (`aggregation_engine.py`)
- **AggregationEngine**: Combines word-level predictions into sentence-level
- Supports multiple strategies:
  - **Weighted Average**: Uses confidence scores as weights
  - **Majority Voting**: Most frequent emotion wins
  - **Temporal Weighted**: Later words receive higher weights
  - **Confidence Threshold**: Only includes high-confidence predictions
- Detects mixed emotions when no single emotion dominates

### 5. Job Queue Manager (`job_queue.py`)
- **JobQueueManager**: Handles asynchronous processing
- FIFO (First-In-First-Out) queue ordering
- Rate limiting (10 requests/minute per IP)
- Processing timeout (30 seconds)
- Maximum queue size (50 jobs)

### 6. Logging Infrastructure (`logging_config.py`)
- Structured JSON logging
- Contextual logging with request IDs and job IDs
- Performance metrics tracking
- Error logging with stack traces

## Usage Example

```python
from sentence_analysis import (
    AudioSegmenter,
    FeatureExtractor,
    WordLevelPredictor,
    AggregationEngine,
    JobQueueManager
)
import librosa

# Load audio
audio, sr = librosa.load("sentence.wav", sr=16000)

# Segment audio
segmenter = AudioSegmenter()
segments = segmenter.segment_audio(audio, sr)

# Extract features
extractor = FeatureExtractor()
features_list = [
    extractor.extract_features(seg.audio_data, seg.sample_rate)
    for seg in segments
]

# Predict word-level emotions
predictor = WordLevelPredictor(model_path="models/best_emotion_model.pkl")
predictions = predictor.predict_batch(features_list)

# Aggregate to sentence-level
aggregator = AggregationEngine()
sentence_result = aggregator.aggregate(predictions, strategy="weighted_average")

print(f"Emotion: {sentence_result.primary_emotion}")
print(f"Confidence: {sentence_result.confidence:.2f}")
print(f"Probabilities: {sentence_result.probabilities}")
```

## Testing

Property-based tests are implemented using Hypothesis:

```bash
# Run all tests
pytest tests/test_job_queue_properties.py -v

# Run with coverage
pytest tests/ --cov=sentence_analysis --cov-report=html
```

### Property Tests

- **Property 28: FIFO Queue Ordering** - Validates that jobs are processed in the order they were received

## Requirements

- Python 3.10+
- librosa >= 0.10.0
- scikit-learn >= 1.3.0
- numpy >= 1.24.0
- hypothesis >= 6.92.0 (for testing)
- pytest >= 7.4.0 (for testing)

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

All components support configuration through constructor parameters:

```python
# Configure segmenter
segmenter = AudioSegmenter(
    silence_threshold=0.02,
    min_silence_duration=0.1,
    segment_padding=0.05
)

# Configure job queue
queue_manager = JobQueueManager(
    max_queue_size=50,
    rate_limit_requests=10,
    rate_limit_window=60
)
```

## Logging

Enable structured JSON logging:

```python
from sentence_analysis.logging_config import setup_logging

logger = setup_logging(level="INFO", json_format=True)
```

## Next Steps

1. Implement API endpoints (Task 6)
2. Add remaining property tests for other components
3. Integrate with frontend
4. Deploy and monitor
