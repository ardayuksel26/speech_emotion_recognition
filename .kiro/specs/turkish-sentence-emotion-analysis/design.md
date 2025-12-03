# Design Document

## Overview

The Turkish Sentence Emotion Analysis system extends the existing word-level emotion recognition to analyze complete Turkish sentences. The system leverages the proven Gradient Boosting Classifier (which achieved the best performance in testing) as the foundation for word-level predictions, then applies sophisticated aggregation strategies to derive sentence-level emotional content.

### Key Design Principles

1. **Modularity**: Separate concerns between audio processing, ML inference, aggregation, and presentation
2. **Extensibility**: Support multiple aggregation strategies and optional dedicated sentence-level models
3. **Performance**: Asynchronous processing with efficient resource management
4. **User Experience**: Professional, intuitive interface with rich visualizations
5. **Reliability**: Comprehensive error handling and graceful degradation

### System Goals

- Analyze Turkish sentence audio files (up to 30 seconds)
- Provide accurate emotion classification (angry, calm, happy, sad)
- Display word-by-word emotion progression
- Support multiple aggregation strategies
- Maintain existing features (voice recording, bilingual support)
- Achieve minimum 70% accuracy on sentence-level classification

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Audio Upload │  │ Voice Record │  │ Results View │      │
│  │   Component  │  │  Component   │  │  Component   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         │                  │                  │              │
│         └──────────────────┴──────────────────┘              │
│                            │                                 │
└────────────────────────────┼─────────────────────────────────┘
                             │ HTTP/REST
┌────────────────────────────┼─────────────────────────────────┐
│                     Backend (FastAPI)                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              API Layer                                │   │
│  │  /api/analyze-sentence  |  /api/status/{job_id}     │   │
│  └──────────────────────────────────────────────────────┘   │
│                            │                                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         Sentence Analysis Service                     │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐     │   │
│  │  │   Audio    │  │  Feature   │  │ Aggregation│     │   │
│  │  │ Segmenter  │  │ Extractor  │  │  Engine    │     │   │
│  │  └────────────┘  └────────────┘  └────────────┘     │   │
│  └──────────────────────────────────────────────────────┘   │
│                            │                                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         Word-Level Model (Gradient Boosting)         │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐     │   │
│  │  │   Model    │  │   Scaler   │  │   Label    │     │   │
│  │  │  Artifact  │  │  Artifact  │  │  Encoder   │     │   │
│  │  └────────────┘  └────────────┘  └────────────┘     │   │
│  └──────────────────────────────────────────────────────┘   │
│                            │                                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         Job Queue & Task Manager                      │   │
│  │  (Async processing, rate limiting, timeout)          │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

1. **User uploads audio** → Frontend sends to `/api/analyze-sentence`
2. **Backend validates** → Creates job, returns job_id
3. **Audio segmentation** → Detects word boundaries using VAD
4. **Feature extraction** → Extracts MFCC, chroma, spectral features per segment
5. **Word-level prediction** → Applies Gradient Boosting model to each segment
6. **Aggregation** → Combines predictions using multiple strategies
7. **Response generation** → Returns sentence-level + word-level results
8. **Frontend visualization** → Displays results with charts and timeline

## Components and Interfaces

### Backend Components

#### 1. Audio Segmentation Module

**Purpose**: Segment sentence audio into individual word segments

**Key Classes**:
```python
class AudioSegmenter:
    def segment_audio(self, audio: np.ndarray, sr: int) -> List[AudioSegment]
    def detect_word_boundaries(self, audio: np.ndarray, sr: int) -> List[Tuple[float, float]]
    def apply_vad(self, audio: np.ndarray, sr: int, threshold: float = 0.02) -> np.ndarray
    def extract_segment(self, audio: np.ndarray, start: float, end: float, padding: float = 0.05) -> np.ndarray
```

**Algorithms**:
- Voice Activity Detection (VAD) using energy-based thresholding
- Minimum silence duration: 100ms
- Segment padding: 50ms on each side
- Fallback: Fixed-duration windowing (500ms windows, 250ms overlap)

#### 2. Feature Extraction Module

**Purpose**: Extract audio features consistent with word-level model training

**Key Classes**:
```python
class FeatureExtractor:
    def extract_features(self, audio: np.ndarray, sr: int) -> np.ndarray
    def extract_mfcc(self, audio: np.ndarray, sr: int, n_mfcc: int = 40) -> np.ndarray
    def extract_chroma(self, audio: np.ndarray, sr: int) -> np.ndarray
    def extract_mel_spectrogram(self, audio: np.ndarray, sr: int) -> np.ndarray
    def extract_spectral_contrast(self, audio: np.ndarray, sr: int) -> np.ndarray
    def extract_temporal_features(self, audio: np.ndarray) -> Dict[str, float]
```

**Feature Vector Composition** (same as word-level model):
- MFCC: 40 coefficients × 2 (mean, std) = 80 features
- Chroma: 12 coefficients × 2 = 24 features
- Mel Spectrogram: 128 bands × 2 = 256 features
- Spectral Contrast: 7 bands × 2 = 14 features
- Zero Crossing Rate: 2 features (mean, std)
- RMS Energy: 2 features (mean, std)
- **Total: 378 features per segment**

#### 3. Word-Level Prediction Module

**Purpose**: Apply trained Gradient Boosting model to word segments

**Key Classes**:
```python
class WordLevelPredictor:
    def __init__(self, model_path: str)
    def load_model(self) -> Dict
    def predict(self, features: np.ndarray) -> PredictionResult
    def predict_batch(self, features_list: List[np.ndarray]) -> List[PredictionResult]
    
@dataclass
class PredictionResult:
    emotion: str
    confidence: float
    probabilities: Dict[str, float]
    is_uncertain: bool  # True if confidence < 0.4
```

#### 4. Aggregation Engine

**Purpose**: Combine word-level predictions into sentence-level prediction

**Key Classes**:
```python
class AggregationEngine:
    def aggregate(self, predictions: List[PredictionResult], strategy: str) -> SentencePrediction
    def weighted_average(self, predictions: List[PredictionResult]) -> SentencePrediction
    def majority_voting(self, predictions: List[PredictionResult]) -> SentencePrediction
    def temporal_weighted(self, predictions: List[PredictionResult]) -> SentencePrediction
    def confidence_threshold(self, predictions: List[PredictionResult], threshold: float = 0.5) -> SentencePrediction
    def detect_mixed_emotions(self, probabilities: Dict[str, float]) -> bool

@dataclass
class SentencePrediction:
    primary_emotion: str
    confidence: float
    probabilities: Dict[str, float]
    is_mixed: bool
    secondary_emotion: Optional[str]
    strategy_used: str
    metadata: Dict
```

**Aggregation Strategies**:

1. **Weighted Average** (Primary Strategy):
   - Weight each word prediction by its confidence score
   - Formula: `P(emotion) = Σ(confidence_i × prob_i) / Σ(confidence_i)`

2. **Majority Voting**:
   - Select the most frequently predicted emotion
   - Tie-breaking: Use highest average confidence

3. **Temporal Weighted**:
   - Apply linear decay weights from start to end
   - Later words receive higher weights (recency bias)
   - Formula: `weight_i = (i + 1) / n`

4. **Confidence Threshold**:
   - Only include predictions with confidence > 0.5
   - Fallback to all predictions if none meet threshold

#### 5. Job Queue Manager

**Purpose**: Handle asynchronous processing and resource management

**Key Classes**:
```python
class JobQueueManager:
    def __init__(self, max_queue_size: int = 50)
    def create_job(self, audio_file: UploadFile) -> str  # Returns job_id
    def get_job_status(self, job_id: str) -> JobStatus
    def process_job(self, job_id: str) -> None
    def apply_rate_limiting(self, client_ip: str) -> bool
    
@dataclass
class JobStatus:
    job_id: str
    status: str  # 'queued', 'processing', 'completed', 'failed'
    progress: float  # 0.0 to 1.0
    result: Optional[SentencePrediction]
    error: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]
```

### Frontend Components

#### 1. Audio Input Component

**Features**:
- File upload (drag-and-drop + file picker)
- Voice recording with real-time waveform visualization
- Audio playback controls
- Format validation (WAV, 8-48kHz, max 30s)

**React Component Structure**:
```jsx
<AudioInputComponent>
  <FileUploadZone />
  <VoiceRecorder 
    onRecordingComplete={handleRecording}
    maxDuration={30}
  />
  <AudioPlayer src={audioUrl} />
  <ValidationFeedback errors={validationErrors} />
</AudioInputComponent>
```

#### 2. Results Display Component

**Features**:
- Primary emotion badge with color coding
- Confidence meter (circular progress or bar)
- Probability distribution chart (horizontal bar chart)
- Word-by-word emotion timeline
- Expandable detailed view
- Export functionality

**React Component Structure**:
```jsx
<ResultsDisplay result={sentenceResult}>
  <EmotionBadge 
    emotion={result.primary_emotion}
    confidence={result.confidence}
  />
  <ConfidenceMeter value={result.confidence} />
  <ProbabilityChart data={result.probabilities} />
  <WordTimeline 
    words={result.word_predictions}
    onWordHover={showTooltip}
  />
  <DetailedView expanded={showDetails}>
    <MetadataPanel />
    <AggregationStrategyInfo />
  </DetailedView>
  <ExportButton formats={['json', 'csv']} />
</ResultsDisplay>
```

#### 3. Visualization Components

**Word Timeline Component**:
- Horizontal timeline showing word positions
- Color-coded emotion indicators per word
- Transition markers between different emotions
- Interactive tooltips with detailed info

**Probability Chart Component**:
- Horizontal bar chart for 4 emotions
- Color scheme: Angry (red), Calm (blue), Happy (yellow), Sad (purple)
- Percentage labels
- Animated transitions

**Confidence Meter Component**:
- Circular progress indicator
- Color gradient based on confidence level
- Percentage display in center

#### 4. Language Switcher Component

**Features**:
- Toggle between Turkish and English
- Persists selection to localStorage
- Updates all UI text dynamically
- Emotion label translations

## Data Models

### Backend Data Models

```python
@dataclass
class AudioSegment:
    """Represents a single word segment"""
    audio_data: np.ndarray
    start_time: float
    end_time: float
    duration: float
    sample_rate: int
    word_index: int

@dataclass
class WordPrediction:
    """Word-level emotion prediction"""
    word_index: int
    start_time: float
    end_time: float
    emotion: str
    confidence: float
    probabilities: Dict[str, float]
    is_uncertain: bool
    features: Optional[np.ndarray] = None

@dataclass
class SentenceAnalysisResult:
    """Complete sentence analysis result"""
    job_id: str
    sentence_prediction: SentencePrediction
    word_predictions: List[WordPrediction]
    metadata: AnalysisMetadata
    processing_time: float
    timestamp: datetime

@dataclass
class AnalysisMetadata:
    """Metadata about the analysis"""
    num_words: int
    audio_duration: float
    sample_rate: int
    aggregation_strategy: str
    model_name: str
    model_accuracy: float
    has_mixed_emotions: bool
    emotion_transitions: int
```

### Frontend Data Models

```typescript
interface SentenceResult {
  jobId: string;
  primaryEmotion: 'angry' | 'calm' | 'happy' | 'sad';
  confidence: number;
  probabilities: {
    angry: number;
    calm: number;
    happy: number;
    sad: number;
  };
  isMixed: boolean;
  secondaryEmotion?: string;
  wordPredictions: WordPrediction[];
  metadata: AnalysisMetadata;
  processingTime: number;
}

interface WordPrediction {
  wordIndex: number;
  startTime: number;
  endTime: number;
  emotion: string;
  confidence: number;
  probabilities: Record<string, number>;
  isUncertain: boolean;
}

interface AnalysisMetadata {
  numWords: number;
  audioDuration: number;
  aggregationStrategy: string;
  modelName: string;
  modelAccuracy: number;
  hasMixedEmotions: boolean;
  emotionTransitions: number;
}
```

### API Request/Response Models

```python
# Request
class AnalyzeSentenceRequest(BaseModel):
    audio_file: UploadFile
    aggregation_strategy: Optional[str] = "weighted_average"
    include_word_details: bool = True

# Response
class AnalyzeSentenceResponse(BaseModel):
    job_id: str
    status: str
    message: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    result: Optional[SentenceAnalysisResult]
    error: Optional[str]
```


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Audio Format Validation

*For any* uploaded audio file, if the file is in WAV format with sample rate between 8kHz and 48kHz, then the system should accept it; otherwise it should reject it with an appropriate error message.

**Validates: Requirements 1.1**

### Property 2: Duration Validation

*For any* uploaded audio file, if the duration exceeds 30 seconds, then the system should reject the upload and return an error message indicating the duration limit.

**Validates: Requirements 1.2**

### Property 3: Upload Round-Trip

*For any* valid audio file, when uploaded successfully, the system should return a unique job identifier that can be used to retrieve the analysis results.

**Validates: Requirements 1.3**

### Property 4: Silence Detection

*For any* audio file where silence or non-speech content exceeds 50% of the duration, the system should detect this condition and include a warning in the response.

**Validates: Requirements 1.4**

### Property 5: Request Isolation

*For any* sequence of multiple audio uploads, each upload should receive a unique job ID and processing one should not affect the results of another.

**Validates: Requirements 1.5**

### Property 6: Word Boundary Detection

*For any* sentence audio file, the VAD-based segmentation should detect word boundaries where silence exceeds 100ms, and each detected boundary should result in a separate audio segment.

**Validates: Requirements 2.1**

### Property 7: Segment Padding Consistency

*For any* detected word boundary, the extracted audio segment should include exactly 50ms of padding on both the start and end sides.

**Validates: Requirements 2.2**

### Property 8: Segmentation Fallback

*For any* audio where VAD-based segmentation fails to detect word boundaries, the system should apply fixed-duration windowing with 500ms windows and 250ms overlap.

**Validates: Requirements 2.5**

### Property 9: Feature Vector Completeness

*For any* audio segment, the extracted feature vector should contain exactly 378 features (80 MFCC + 24 chroma + 256 mel + 14 spectral contrast + 2 ZCR + 2 RMS).

**Validates: Requirements 3.1, 3.2**

### Property 10: Scaler Consistency

*For any* feature vector, the StandardScaler transformation applied should use the same parameters (mean, std) as the scaler used during word-level model training.

**Validates: Requirements 3.3**

### Property 11: Model Output Format

*For any* valid feature vector input to the Gradient Boosting Classifier, the output should be a probability distribution across all four emotion classes (angry, calm, happy, sad) that sums to 1.0.

**Validates: Requirements 4.1, 4.2**

### Property 12: Uncertainty Flagging

*For any* word-level prediction, if the highest probability is below 0.4, then the prediction should be flagged as uncertain.

**Validates: Requirements 4.3**

### Property 13: Temporal Order Preservation

*For any* sequence of word segments processed, the output predictions should maintain the same temporal order as the input segments.

**Validates: Requirements 4.4**

### Property 14: Prediction Fallback

*For any* word segment where model prediction fails, the system should assign a uniform probability distribution (0.25 for each emotion) as a fallback.

**Validates: Requirements 4.5**

### Property 15: Weighted Average Aggregation

*For any* list of word-level predictions with confidence scores, the weighted average aggregation should compute sentence-level probabilities as Σ(confidence_i × prob_i) / Σ(confidence_i).

**Validates: Requirements 5.1**

### Property 16: Majority Voting Aggregation

*For any* list of word-level predictions, the majority voting aggregation should select the emotion that appears most frequently, with ties broken by highest average confidence.

**Validates: Requirements 5.2**

### Property 17: Temporal Weighting Aggregation

*For any* list of n word-level predictions, the temporal-weighted aggregation should apply weights where weight_i = (i + 1) / n, giving later words progressively higher influence.

**Validates: Requirements 5.3**

### Property 18: Confidence Threshold Aggregation

*For any* list of word-level predictions, the confidence-threshold aggregation should only include predictions with confidence > 0.5 in the final calculation.

**Validates: Requirements 5.4**

### Property 19: Multi-Strategy Execution

*For any* sentence analysis request, when multiple aggregation strategies are applied, the response should contain results from all requested strategies.

**Validates: Requirements 5.5**

### Property 20: Sentence Result Completeness

*For any* completed sentence-level analysis, the result should include primary emotion label, confidence score, and probability distributions for all four emotions.

**Validates: Requirements 6.1, 6.2, 6.3**

### Property 21: Mixed Emotion Detection

*For any* sentence-level prediction, if no single emotion has probability > 0.5, then the result should be flagged as "mixed" and include the top two emotions.

**Validates: Requirements 6.4**

### Property 22: Metadata Inclusion

*For any* sentence analysis result, the metadata should include the number of words analyzed and the aggregation strategy used.

**Validates: Requirements 6.5**

### Property 23: Word-Level Data Structure

*For any* word prediction in the results, it should include timestamp, word index, predicted emotion, and confidence score.

**Validates: Requirements 7.2**

### Property 24: Emotion Transition Detection

*For any* pair of consecutive word predictions with different emotions, the system should mark this as an emotion transition in the visualization data.

**Validates: Requirements 7.3**

### Property 25: Export Format Support

*For any* completed analysis, the system should be able to export the results in both JSON and CSV formats with all required data fields.

**Validates: Requirements 7.5, 15.1, 15.2, 15.3**

### Property 26: Asynchronous Processing

*For any* sentence analysis request, the API should return a job identifier immediately (within 100ms) before processing completes.

**Validates: Requirements 8.1**

### Property 27: Processing Performance

*For any* sentence with 10 words, the complete analysis (segmentation + feature extraction + prediction + aggregation) should complete within 5 seconds.

**Validates: Requirements 8.2**

### Property 28: FIFO Queue Ordering

*For any* sequence of queued requests, they should be processed in the order they were received (first-in, first-out).

**Validates: Requirements 8.3**

### Property 29: Processing Timeout

*For any* analysis request that exceeds 30 seconds of processing time, the system should terminate the request and return a timeout error.

**Validates: Requirements 8.4**

### Property 30: Rate Limiting

*For any* client IP address, if more than 10 requests are made within a 1-minute window, subsequent requests should be rejected with a 429 Too Many Requests status.

**Validates: Requirements 8.5**

### Property 31: API Response Structure

*For any* completed analysis, the JSON response should contain sentence-level prediction, confidence, word-level details, and visualization data fields.

**Validates: Requirements 9.2**

### Property 32: Error Status Codes

*For any* request that fails due to client error (invalid input), the system should return HTTP 400; for server errors, it should return HTTP 500.

**Validates: Requirements 9.4**

### Property 33: CORS Headers

*For any* API response, the headers should include appropriate CORS headers allowing requests from the configured frontend origin.

**Validates: Requirements 9.5**

### Property 34: Minimum Accuracy Threshold

*For any* test set of labeled Turkish sentence audio samples, the system should achieve at least 70% accuracy on sentence-level emotion classification.

**Validates: Requirements 11.2**

### Property 35: Aggregation Strategy Logging

*For any* analysis that compares multiple aggregation strategies, the system should log performance metrics (accuracy, F1-score) for each strategy.

**Validates: Requirements 11.3**

### Property 36: Graceful Error Handling

*For any* edge case input (very short sentences, very long sentences, noisy audio), the system should handle it without crashing and return either results or a descriptive error.

**Validates: Requirements 11.5**

### Property 37: Error Logging Completeness

*For any* processing step that fails, the error log should include timestamp, request ID, error type, and stack trace.

**Validates: Requirements 13.1**

### Property 38: Input Validation

*For any* request with invalid or unexpected input parameters, the system should validate the input and return HTTP 400 with detailed validation errors.

**Validates: Requirements 13.3**

### Property 39: Resource Exhaustion Handling

*For any* situation where system resources are exhausted, the system should return HTTP 503 Service Unavailable.

**Validates: Requirements 13.4**

### Property 40: Success Logging

*For any* successfully completed analysis, the system should log performance metrics including processing time, number of words, and confidence scores.

**Validates: Requirements 13.5**

### Property 41: Export Filename Format

*For any* exported file, the filename should include a timestamp and unique identifier in the format: `emotion_analysis_{job_id}_{timestamp}.{format}`.

**Validates: Requirements 15.5**


## Error Handling

### Error Categories

#### 1. Client Errors (4xx)

**400 Bad Request**:
- Invalid audio format (not WAV)
- Invalid sample rate (outside 8-48kHz range)
- Audio duration exceeds 30 seconds
- Missing required parameters
- Malformed request body

**429 Too Many Requests**:
- Rate limit exceeded (>10 requests/minute per IP)
- Queue is full (>50 pending jobs)

**404 Not Found**:
- Job ID does not exist
- Requested resource not found

#### 2. Server Errors (5xx)

**500 Internal Server Error**:
- Model loading failure
- Feature extraction failure
- Unexpected exception during processing

**503 Service Unavailable**:
- System resources exhausted
- Model not loaded
- Service temporarily unavailable

### Error Response Format

```json
{
  "error": {
    "code": "INVALID_AUDIO_FORMAT",
    "message": "The uploaded file must be in WAV format",
    "message_tr": "Yüklenen dosya WAV formatında olmalıdır",
    "details": {
      "received_format": "mp3",
      "expected_format": "wav"
    },
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "req_abc123"
  }
}
```

### Error Handling Strategy

1. **Validation Errors**: Catch early at API layer, return immediately with 400
2. **Processing Errors**: Log detailed error, return 500 with safe error message
3. **Timeout Errors**: Cancel processing, return partial results if available
4. **Resource Errors**: Implement circuit breaker pattern, return 503
5. **Graceful Degradation**: If word segmentation fails, use fallback windowing

### Logging Strategy

```python
# Error logging format
{
    "timestamp": "2024-01-15T10:30:00Z",
    "level": "ERROR",
    "request_id": "req_abc123",
    "error_type": "FeatureExtractionError",
    "error_message": "Failed to extract MFCC features",
    "stack_trace": "...",
    "context": {
        "job_id": "job_xyz789",
        "segment_index": 3,
        "audio_duration": 0.45
    }
}

# Success logging format
{
    "timestamp": "2024-01-15T10:30:05Z",
    "level": "INFO",
    "request_id": "req_abc123",
    "event": "analysis_completed",
    "metrics": {
        "processing_time": 2.34,
        "num_words": 8,
        "primary_emotion": "happy",
        "confidence": 0.87,
        "aggregation_strategy": "weighted_average"
    }
}
```

## Testing Strategy

### Unit Testing

**Backend Unit Tests**:
- Audio segmentation logic (VAD, windowing)
- Feature extraction (MFCC, chroma, spectral)
- Aggregation strategies (weighted average, majority voting, etc.)
- Input validation
- Error handling

**Frontend Unit Tests**:
- Component rendering
- State management
- API integration
- Language switching
- Export functionality

**Test Framework**: pytest for backend, Jest + React Testing Library for frontend

### Property-Based Testing

**Property Testing Library**: Hypothesis (Python) for backend

**Key Properties to Test**:
1. Feature vector always has 378 dimensions
2. Probability distributions always sum to 1.0
3. Temporal order is preserved through pipeline
4. Aggregation strategies produce valid outputs
5. Error handling never crashes the system

**Configuration**: Each property test should run minimum 100 iterations

### Integration Testing

**API Integration Tests**:
- End-to-end audio upload → analysis → results flow
- Job status polling
- Error scenarios (invalid input, timeouts)
- Rate limiting
- CORS headers

**Test Data**:
- 50+ labeled Turkish sentence audio samples
- Covering all 4 emotions
- Various lengths (short, medium, long)
- Edge cases (noisy audio, silence, continuous speech)

### Performance Testing

**Benchmarks**:
- 10-word sentence: < 5 seconds
- Single request latency: < 100ms to return job ID
- Concurrent requests: Support 10 simultaneous analyses
- Memory usage: < 500MB per analysis

**Tools**: locust for load testing, pytest-benchmark for micro-benchmarks

### Test Coverage Goals

- Backend code coverage: > 80%
- Frontend code coverage: > 70%
- Critical paths (audio processing, model inference): 100%

## Implementation Notes

### Technology Stack

**Backend**:
- FastAPI 0.104+
- Python 3.10+
- librosa 0.10+ (audio processing)
- scikit-learn 1.3+ (ML models)
- numpy, pandas (data processing)
- pydantic (data validation)
- uvicorn (ASGI server)

**Frontend**:
- React 19
- Vite 7
- TypeScript 5+
- Tailwind CSS 3+
- Chart.js or Recharts (visualizations)
- i18next (internationalization)
- Axios (HTTP client)

**Development Tools**:
- pytest (backend testing)
- Jest + React Testing Library (frontend testing)
- Hypothesis (property-based testing)
- Black + isort (Python formatting)
- ESLint + Prettier (JavaScript formatting)

### Deployment Considerations

**Backend Deployment**:
- Docker container with Python 3.10
- Pre-load model artifacts on startup
- Environment variables for configuration
- Health check endpoint: `/health`
- Metrics endpoint: `/metrics`

**Frontend Deployment**:
- Static build served via CDN or nginx
- Environment-specific API URLs
- Gzip compression enabled
- Cache headers for static assets

**Model Artifacts**:
- Store trained model in `models/` directory
- Include scaler, label encoder, and model in pickle file
- Version model artifacts (e.g., `model_v1.0.pkl`)
- Document model training parameters

### Security Considerations

1. **Input Validation**: Strict validation of file size, format, duration
2. **Rate Limiting**: Prevent abuse with per-IP rate limits
3. **File Cleanup**: Delete temporary audio files after processing
4. **CORS**: Configure allowed origins (not wildcard in production)
5. **Error Messages**: Don't expose internal system details in errors
6. **Logging**: Sanitize logs to avoid logging sensitive data

### Scalability Considerations

**Current Design** (Single Server):
- Handles 10 concurrent requests
- Queue size: 50 requests
- Suitable for graduation project / demo

**Future Scalability** (if needed):
- Horizontal scaling: Multiple backend instances behind load balancer
- Distributed queue: Redis or RabbitMQ for job queue
- Separate workers: Dedicated worker processes for analysis
- Caching: Cache model artifacts in shared memory
- Database: Store job results in PostgreSQL or MongoDB

### Performance Optimization

1. **Model Loading**: Load model once at startup, reuse for all requests
2. **Feature Extraction**: Vectorize operations using numpy
3. **Batch Processing**: Process multiple segments in parallel if possible
4. **Caching**: Cache scaler parameters, avoid recomputing
5. **Async I/O**: Use async file operations for audio loading
6. **Memory Management**: Clear audio data after processing

### Monitoring and Observability

**Metrics to Track**:
- Request rate (requests/minute)
- Processing time (p50, p95, p99)
- Error rate (errors/total requests)
- Queue depth (pending jobs)
- Model inference time
- Memory usage

**Logging**:
- Structured JSON logs
- Log levels: DEBUG, INFO, WARNING, ERROR
- Include request IDs for tracing
- Rotate logs daily

**Health Checks**:
- `/health`: Basic liveness check
- `/health/ready`: Readiness check (model loaded, queue not full)
- `/metrics`: Prometheus-compatible metrics

## User Experience Enhancements

### Professional UI Design

**Color Scheme**:
- Angry: #EF4444 (red-500)
- Calm: #3B82F6 (blue-500)
- Happy: #F59E0B (amber-500)
- Sad: #8B5CF6 (purple-500)
- Background: #F9FAFB (gray-50)
- Text: #111827 (gray-900)

**Typography**:
- Headings: Inter or Poppins (bold, 600-700 weight)
- Body: Inter or system font (regular, 400 weight)
- Monospace: JetBrains Mono (for technical data)

**Layout**:
- Responsive design (mobile, tablet, desktop)
- Maximum content width: 1200px
- Generous whitespace
- Card-based components with subtle shadows

### Interactive Features

1. **Real-time Waveform**: Show audio waveform during recording
2. **Animated Transitions**: Smooth transitions between states
3. **Loading States**: Skeleton loaders and progress indicators
4. **Tooltips**: Contextual help on hover
5. **Keyboard Shortcuts**: Space to play/pause, R to record
6. **Drag-and-Drop**: Visual feedback during file drag

### Accessibility

- ARIA labels for screen readers
- Keyboard navigation support
- High contrast mode support
- Focus indicators
- Alt text for icons
- Semantic HTML

### User Feedback

**Success States**:
- Checkmark animation on successful upload
- Confetti effect for high-confidence predictions
- Success toast notifications

**Error States**:
- Clear error messages in user's language
- Suggestions for fixing errors
- Retry buttons
- Error toast notifications

**Loading States**:
- Progress bar showing analysis stages
- Estimated time remaining
- Cancel button for long operations

### Data Visualization Best Practices

**Emotion Timeline**:
- Horizontal scrollable timeline
- Color-coded segments
- Smooth transitions between emotions
- Zoom controls for long sentences

**Probability Chart**:
- Horizontal bars sorted by probability
- Percentage labels
- Animated bar growth
- Highlight primary emotion

**Confidence Meter**:
- Circular progress with gradient
- Color changes based on confidence level
- Animated fill
- Percentage in center

### Bilingual Support

**Language Toggle**:
- Prominent language switcher in header
- Flag icons for visual recognition
- Persists selection to localStorage
- Instant UI update (no page reload)

**Translations**:
- All UI text translated
- Error messages in both languages
- Emotion labels translated
- Help text and tooltips translated

**RTL Support** (future):
- Prepare for right-to-left languages
- Flexible layout system

## Future Enhancements

### Phase 2 Features (Post-Graduation)

1. **Real-time Streaming**: Analyze audio as it's being recorded
2. **Batch Processing**: Upload multiple files at once
3. **Historical Analysis**: Save and compare past analyses
4. **User Accounts**: Personal dashboards and history
5. **Advanced Visualizations**: 3D emotion space, heatmaps
6. **Mobile App**: Native iOS/Android applications
7. **API Keys**: Public API with authentication
8. **Webhooks**: Notify external systems of results
9. **Custom Models**: Allow users to train on their own data
10. **More Emotions**: Expand beyond 4 emotions (surprise, fear, disgust)

### Research Opportunities

1. **Deep Learning Models**: LSTM, Transformer-based models
2. **Transfer Learning**: Fine-tune pre-trained models
3. **Multi-modal Analysis**: Combine audio with text transcription
4. **Speaker Identification**: Detect multiple speakers
5. **Emotion Intensity**: Not just type, but intensity level
6. **Cultural Adaptation**: Turkish-specific emotion patterns
7. **Contextual Analysis**: Consider linguistic context
8. **Temporal Dynamics**: Model emotion changes over time

## Conclusion

This design provides a comprehensive, professional, and scalable solution for Turkish sentence emotion analysis. The system builds on proven word-level models, implements multiple aggregation strategies, and delivers a user-friendly interface with rich visualizations. The modular architecture allows for future enhancements while maintaining reliability and performance suitable for a graduation project.

Key strengths:
- ✅ Leverages existing Gradient Boosting model (best performer)
- ✅ Multiple aggregation strategies for robustness
- ✅ Professional, engaging UI with visualizations
- ✅ Comprehensive error handling and logging
- ✅ Bilingual support (Turkish/English)
- ✅ Testable with property-based testing
- ✅ Scalable architecture
- ✅ Well-documented and maintainable

The implementation will follow this design to create a graduation-quality project that demonstrates both technical excellence and practical usability.
