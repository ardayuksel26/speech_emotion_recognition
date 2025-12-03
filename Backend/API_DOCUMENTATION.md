# Turkish Sentence Emotion Analysis API Documentation

## Overview

This API provides sentence-level emotion analysis for Turkish audio. It segments audio into words, analyzes each word using a trained Gradient Boosting model, and aggregates the results using multiple strategies.

## Base URL

```
http://localhost:8000
```

## Endpoints

### 1. Analyze Sentence

Analyze a Turkish sentence audio file for emotion.

**Endpoint:** `POST /api/analyze-sentence`

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body Parameters:
  - `file` (required): Audio file in WAV format
  - `aggregation_strategy` (optional): Strategy to use for aggregation
    - Options: `weighted_average` (default), `majority_voting`, `temporal_weighted`, `confidence_threshold`

**Audio Requirements:**
- Format: WAV
- Sample Rate: 8kHz to 48kHz
- Duration: Maximum 30 seconds
- Language: Turkish

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "message": "Analysis job created successfully"
}
```

**Status Codes:**
- `200 OK`: Job created successfully
- `400 Bad Request`: Invalid audio format or properties
- `429 Too Many Requests`: Rate limit exceeded (max 10 requests/minute)
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Queue is full

**Example (curl):**
```bash
curl -X POST "http://localhost:8000/api/analyze-sentence" \
  -F "file=@sentence.wav" \
  -F "aggregation_strategy=weighted_average"
```

**Example (Python):**
```python
import requests

url = "http://localhost:8000/api/analyze-sentence"
files = {"file": open("sentence.wav", "rb")}
data = {"aggregation_strategy": "weighted_average"}

response = requests.post(url, files=files, data=data)
job_data = response.json()
job_id = job_data["job_id"]
```

---

### 2. Get Job Status

Get the status and results of an analysis job.

**Endpoint:** `GET /api/status/{job_id}`

**Request:**
- Method: GET
- Path Parameters:
  - `job_id` (required): Job identifier returned from analyze-sentence

**Response (Processing):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "progress": 0.5,
  "error": null,
  "result": null,
  "created_at": "2024-01-15T10:30:00",
  "completed_at": null
}
```

**Response (Completed):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "progress": 1.0,
  "error": null,
  "result": {
    "primary_emotion": "happy",
    "confidence": 0.87,
    "probabilities": {
      "angry": 0.05,
      "calm": 0.08,
      "happy": 0.87,
      "sad": 0.00
    },
    "is_mixed": false,
    "secondary_emotion": null,
    "word_predictions": [
      {
        "word_index": 0,
        "start_time": 0.0,
        "end_time": 0.5,
        "emotion": "happy",
        "confidence": 0.85,
        "probabilities": {
          "angry": 0.05,
          "calm": 0.10,
          "happy": 0.85,
          "sad": 0.00
        },
        "is_uncertain": false
      },
      {
        "word_index": 1,
        "start_time": 0.5,
        "end_time": 1.0,
        "emotion": "happy",
        "confidence": 0.89,
        "probabilities": {
          "angry": 0.05,
          "calm": 0.06,
          "happy": 0.89,
          "sad": 0.00
        },
        "is_uncertain": false
      }
    ],
    "metadata": {
      "num_words": 2,
      "audio_duration": 1.0,
      "sample_rate": 16000,
      "aggregation_strategy": "weighted_average",
      "has_mixed_emotions": false
    },
    "processing_time": 2.34
  },
  "created_at": "2024-01-15T10:30:00",
  "completed_at": "2024-01-15T10:30:02"
}
```

**Status Codes:**
- `200 OK`: Job status retrieved successfully
- `404 Not Found`: Job ID not found

**Job Status Values:**
- `queued`: Job is waiting in queue
- `processing`: Job is being processed
- `completed`: Job completed successfully
- `failed`: Job failed with error
- `timeout`: Job exceeded processing timeout

**Example (curl):**
```bash
curl "http://localhost:8000/api/status/550e8400-e29b-41d4-a716-446655440000"
```

**Example (Python):**
```python
import requests
import time

url = f"http://localhost:8000/api/status/{job_id}"

# Poll until complete
while True:
    response = requests.get(url)
    data = response.json()
    
    if data["status"] in ["completed", "failed", "timeout"]:
        break
    
    time.sleep(1)

if data["status"] == "completed":
    result = data["result"]
    print(f"Emotion: {result['primary_emotion']}")
    print(f"Confidence: {result['confidence']:.2f}")
```

---

### 3. Health Check

Check API health and queue status.

**Endpoint:** `GET /health`

**Request:**
- Method: GET

**Response:**
```json
{
  "status": "healthy",
  "queue_size": 3,
  "timestamp": "2024-01-15T10:30:00"
}
```

**Status Codes:**
- `200 OK`: Service is healthy

---

## Aggregation Strategies

### 1. Weighted Average (Default)
Uses confidence scores as weights to compute sentence-level probabilities.

**Formula:** `P(emotion) = Σ(confidence_i × prob_i) / Σ(confidence_i)`

**Best for:** General-purpose sentence analysis with varying word confidence

### 2. Majority Voting
Selects the most frequently predicted emotion across words.

**Tie-breaking:** Uses highest average confidence

**Best for:** Sentences with clear dominant emotion

### 3. Temporal Weighted
Applies linear decay weights, giving later words higher influence.

**Formula:** `weight_i = (i + 1) / n`

**Best for:** Sentences where final words carry more emotional weight

### 4. Confidence Threshold
Only includes predictions with confidence > 0.5 in aggregation.

**Fallback:** Uses all predictions if none meet threshold

**Best for:** Filtering out uncertain word predictions

---

## Rate Limiting

- **Limit:** 10 requests per minute per IP address
- **Response:** HTTP 429 Too Many Requests
- **Reset:** Sliding window of 60 seconds

---

## Error Handling

### Error Response Format

```json
{
  "detail": {
    "code": "ERROR_CODE",
    "message": "Error message in English",
    "message_tr": "Türkçe hata mesajı"
  }
}
```

### Common Error Codes

| Code | Status | Description |
|------|--------|-------------|
| `INVALID_AUDIO_FORMAT` | 400 | File is not in WAV format |
| `INVALID_AUDIO_PROPERTIES` | 400 | Sample rate or duration invalid |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `QUEUE_FULL` | 503 | Job queue is full |
| `JOB_NOT_FOUND` | 404 | Job ID does not exist |
| `INTERNAL_ERROR` | 500 | Server error |

---

## Complete Example

```python
import requests
import time

# 1. Upload audio file
url = "http://localhost:8000/api/analyze-sentence"
files = {"file": open("turkish_sentence.wav", "rb")}
data = {"aggregation_strategy": "weighted_average"}

response = requests.post(url, files=files, data=data)
if response.status_code != 200:
    print(f"Error: {response.json()}")
    exit(1)

job_id = response.json()["job_id"]
print(f"Job created: {job_id}")

# 2. Poll for results
status_url = f"http://localhost:8000/api/status/{job_id}"

while True:
    response = requests.get(status_url)
    data = response.json()
    
    print(f"Status: {data['status']} ({data['progress']*100:.0f}%)")
    
    if data["status"] == "completed":
        result = data["result"]
        print(f"\nPrimary Emotion: {result['primary_emotion']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"\nProbabilities:")
        for emotion, prob in result['probabilities'].items():
            print(f"  {emotion}: {prob:.2%}")
        
        print(f"\nWord-by-word analysis:")
        for word in result['word_predictions']:
            print(f"  Word {word['word_index']}: {word['emotion']} "
                  f"({word['confidence']:.2%})")
        break
    
    elif data["status"] in ["failed", "timeout"]:
        print(f"Error: {data['error']}")
        break
    
    time.sleep(1)
```

---

## Running the Server

### Development

```bash
cd Backend
python main_sentence_analysis.py
```

### Production

```bash
cd Backend
uvicorn main_sentence_analysis:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## Testing

Run the property-based tests:

```bash
cd Backend
pytest tests/test_api_properties.py -v
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Application                       │
└─────────────────────────────────────────────────────────────┘
                             │
                             │ HTTP/REST
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Endpoints                        │
│  POST /api/analyze-sentence  |  GET /api/status/{job_id}   │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                     Job Queue Manager                        │
│  (Async processing, rate limiting, timeout handling)        │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                  Sentence Analysis Pipeline                  │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐           │
│  │   Audio    │→ │  Feature   │→ │   Word     │           │
│  │ Segmenter  │  │ Extractor  │  │ Predictor  │           │
│  └────────────┘  └────────────┘  └────────────┘           │
│                             │                                │
│                             ▼                                │
│                  ┌────────────────────┐                     │
│                  │ Aggregation Engine │                     │
│                  └────────────────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Performance

- **Processing Time:** ~2-5 seconds for 10-word sentence
- **Concurrent Requests:** Up to 50 queued jobs
- **Rate Limit:** 10 requests/minute per IP
- **Timeout:** 30 seconds per job

---

## Support

For issues or questions, please refer to the project documentation or contact the development team.
