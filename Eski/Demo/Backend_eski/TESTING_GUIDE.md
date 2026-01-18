# Backend Test Rehberi

Bu rehber, backend'in doğru çalışıp çalışmadığını test etmek için farklı yöntemleri açıklar.

## 📋 İçindekiler

1. [Hızlı Test](#hızlı-test)
2. [Unit ve Property Testler](#unit-ve-property-testler)
3. [Integration Test](#integration-test)
4. [Manuel API Testi](#manuel-api-testi)
5. [Test Senaryoları](#test-senaryoları)

---

## 🚀 Hızlı Test

### 1. Server'ı Başlat

```bash
cd Backend
python main_sentence_analysis.py
```

Server başladığında şu mesajı görmelisiniz:
```
============================================================
Turkish Sentence Emotion Analysis API
============================================================
Model: Demo/models/best_emotion_model.pkl
Max Queue Size: 50
Rate Limit: 10 requests per 60s
Processing Timeout: 30s
============================================================

API Endpoints:
  POST /api/analyze-sentence - Analyze Turkish sentence audio
  GET  /api/status/{job_id}  - Get analysis job status
  GET  /health               - Health check
============================================================

Starting server on http://0.0.0.0:8000
```

### 2. Health Check

Başka bir terminal açın ve:

```bash
# Windows PowerShell
curl http://localhost:8000/health

# veya Python ile
python -c "import requests; print(requests.get('http://localhost:8000/health').json())"
```

Beklenen çıktı:
```json
{
  "status": "healthy",
  "queue_size": 0,
  "timestamp": "2024-01-15T10:30:00"
}
```

---

## 🧪 Unit ve Property Testler

### Tüm Testleri Çalıştır

```bash
cd Backend
pytest tests/ -v
```

### Belirli Test Dosyalarını Çalıştır

```bash
# API property testleri
pytest tests/test_api_properties.py -v

# Job queue testleri
pytest tests/test_job_queue_properties.py -v

# Aggregation engine testleri
pytest tests/test_aggregation_engine_properties.py -v

# Audio segmenter testleri
pytest tests/test_audio_segmenter_properties.py -v

# Feature extractor testleri
pytest tests/test_feature_extractor_properties.py -v

# Word predictor testleri
pytest tests/test_word_predictor_properties.py -v
```

### Belirli Bir Testi Çalıştır

```bash
# Örnek: Error status codes testi
pytest tests/test_api_properties.py::test_property_error_status_codes -v

# Örnek: Rate limiting testi
pytest tests/test_api_properties.py::test_property_rate_limiting -v
```

### Test Coverage ile Çalıştır

```bash
# Coverage raporu oluştur
pytest tests/ --cov=sentence_analysis --cov-report=html

# HTML raporunu aç (Windows)
start htmlcov/index.html
```

---

## 🔗 Integration Test

Integration test, tüm API workflow'unu test eder (upload → processing → results).

### Integration Test Çalıştır

**Önce server'ı başlatın** (başka bir terminalde):
```bash
cd Backend
python main_sentence_analysis.py
```

**Sonra testi çalıştırın**:
```bash
cd Backend
python test_api_integration.py
```

### Beklenen Çıktı

```
============================================================
Turkish Sentence Emotion Analysis API - Integration Test
============================================================
✓ Server is healthy
  Queue size: 0

1. Creating test audio file...
✓ Created: C:\Users\...\tmpXXXXXX.wav

2. Uploading audio for analysis...
✓ Job created: 550e8400-e29b-41d4-a716-446655440000

3. Waiting for analysis to complete...
✓ Analysis completed!

============================================================
RESULTS
============================================================
Primary Emotion: CALM
Confidence: 45.23%
Mixed Emotions: No

Probability Distribution:
  calm    : ████████████████████████████████████████ 45.23%
  happy   : ████████████████████████ 28.15%
  angry   : ████████████ 15.42%
  sad     : ████████ 11.20%

Word-by-Word Analysis:
  Word 0: calm     (42.15%) [0.00s - 0.50s]
  Word 1: happy    (35.20%) [0.50s - 1.00s]
  ...

Metadata:
  Number of words: 4
  Audio duration: 2.00s
  Sample rate: 16000 Hz
  Aggregation strategy: weighted_average
  Processing time: 2.34s
============================================================
✓ Integration test PASSED
```

---

## 🖥️ Manuel API Testi

### 1. curl ile Test

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Audio Upload
```bash
curl -X POST "http://localhost:8000/api/analyze-sentence" \
  -F "file=@path/to/your/audio.wav" \
  -F "aggregation_strategy=weighted_average"
```

**Yanıt:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "message": "Analysis job created successfully"
}
```

#### Status Kontrolü
```bash
curl "http://localhost:8000/api/status/550e8400-e29b-41d4-a716-446655440000"
```

### 2. Python ile Test

```python
import requests
import time

# 1. Health check
response = requests.get("http://localhost:8000/health")
print("Health:", response.json())

# 2. Upload audio
with open("test_audio.wav", "rb") as f:
    files = {"file": ("test.wav", f, "audio/wav")}
    data = {"aggregation_strategy": "weighted_average"}
    response = requests.post(
        "http://localhost:8000/api/analyze-sentence",
        files=files,
        data=data
    )

job_data = response.json()
job_id = job_data["job_id"]
print(f"Job ID: {job_id}")

# 3. Poll for results
while True:
    response = requests.get(f"http://localhost:8000/api/status/{job_id}")
    status_data = response.json()
    
    print(f"Status: {status_data['status']} ({status_data['progress']*100:.0f}%)")
    
    if status_data["status"] == "completed":
        result = status_data["result"]
        print(f"Primary Emotion: {result['primary_emotion']}")
        print(f"Confidence: {result['confidence']:.2%}")
        break
    elif status_data["status"] in ["failed", "timeout"]:
        print(f"Error: {status_data.get('error')}")
        break
    
    time.sleep(1)
```

### 3. Postman/Insomnia ile Test

1. **POST** `http://localhost:8000/api/analyze-sentence`
   - Body: `form-data`
   - Key: `file` (type: File), value: audio.wav
   - Key: `aggregation_strategy` (type: Text), value: `weighted_average`

2. **GET** `http://localhost:8000/api/status/{job_id}`
   - `job_id`'yi önceki yanıttan alın

---

## 📝 Test Senaryoları

### ✅ Başarılı Senaryolar

#### 1. Geçerli WAV Dosyası
- Format: `.wav`
- Sample rate: 8kHz - 48kHz
- Süre: 0.1s - 30s
- **Beklenen:** 200 OK, job_id döner

#### 2. Farklı Aggregation Stratejileri
```bash
# Weighted average (default)
curl -X POST "http://localhost:8000/api/analyze-sentence" \
  -F "file=@audio.wav" -F "aggregation_strategy=weighted_average"

# Majority voting
curl -X POST "http://localhost:8000/api/analyze-sentence" \
  -F "file=@audio.wav" -F "aggregation_strategy=majority_voting"

# Temporal weighted
curl -X POST "http://localhost:8000/api/analyze-sentence" \
  -F "file=@audio.wav" -F "aggregation_strategy=temporal_weighted"

# Confidence threshold
curl -X POST "http://localhost:8000/api/analyze-sentence" \
  -F "file=@audio.wav" -F "aggregation_strategy=confidence_threshold"
```

### ❌ Hata Senaryoları

#### 1. Geçersiz Format
```bash
curl -X POST "http://localhost:8000/api/analyze-sentence" \
  -F "file=@audio.mp3"
```
**Beklenen:** 400 Bad Request
```json
{
  "detail": {
    "code": "INVALID_AUDIO_FORMAT",
    "message": "The uploaded file must be in WAV format",
    "message_tr": "Yüklenen dosya WAV formatında olmalıdır",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

#### 2. Çok Uzun Süre (>30s)
```bash
# 35 saniyelik bir dosya yükleyin
curl -X POST "http://localhost:8000/api/analyze-sentence" \
  -F "file=@long_audio.wav"
```
**Beklenen:** 400 Bad Request
```json
{
  "detail": {
    "code": "INVALID_DURATION",
    "message": "Audio duration must not exceed 30 seconds",
    "message_tr": "Ses dosyası süresi 30 saniyeyi geçmemelidir"
  }
}
```

#### 3. Geçersiz Sample Rate
```bash
# 5kHz sample rate ile dosya
curl -X POST "http://localhost:8000/api/analyze-sentence" \
  -F "file=@low_sr_audio.wav"
```
**Beklenen:** 400 Bad Request
```json
{
  "detail": {
    "code": "INVALID_SAMPLE_RATE",
    "message": "Sample rate must be between 8kHz and 48kHz",
    "message_tr": "Örnekleme hızı 8kHz ile 48kHz arasında olmalıdır"
  }
}
```

#### 4. Rate Limit Aşımı
```bash
# 15 istek hızlıca gönderin
for i in {1..15}; do
  curl -X POST "http://localhost:8000/api/analyze-sentence" \
    -F "file=@audio.wav"
done
```
**Beklenen:** İlk 10 istek 200 OK, sonraki 429 Too Many Requests

#### 5. Geçersiz Job ID
```bash
curl "http://localhost:8000/api/status/invalid-job-id"
```
**Beklenen:** 404 Not Found
```json
{
  "detail": {
    "code": "JOB_NOT_FOUND",
    "message": "Job invalid-job-id not found",
    "message_tr": "İş bulunamadı"
  }
}
```

#### 6. Geçersiz Aggregation Strategy
```bash
curl -X POST "http://localhost:8000/api/analyze-sentence" \
  -F "file=@audio.wav" -F "aggregation_strategy=invalid_strategy"
```
**Beklenen:** 400 Bad Request
```json
{
  "detail": {
    "code": "INVALID_AGGREGATION_STRATEGY",
    "message": "Invalid aggregation strategy. Must be one of: ...",
    "message_tr": "Geçersiz birleştirme stratejisi..."
  }
}
```

---

## 🔍 Debugging

### Logları İncele

Server çalışırken JSON formatında loglar göreceksiniz:
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "logger": "sentence_analysis",
  "message": "Job created: 550e8400-...",
  "request_id": "req_abc123",
  "job_id": "550e8400-...",
  "context": {...}
}
```

### Hata Durumlarını Kontrol Et

1. **Job Status Kontrolü:**
```bash
curl "http://localhost:8000/api/status/{job_id}"
```

2. **Log Çıktısını İncele:**
   - Server terminalinde JSON loglar görünecek
   - Hata durumlarında `"level": "ERROR"` ve `"exception"` alanları olacak

### Yaygın Sorunlar

#### Model Dosyası Bulunamadı
```
FileNotFoundError: Model file not found: Demo/models/best_emotion_model.pkl
```
**Çözüm:** Model dosyasının doğru yolda olduğundan emin olun.

#### Port Zaten Kullanılıyor
```
ERROR:    [Errno 10048] Only one usage of each socket address...
```
**Çözüm:** Başka bir port kullanın veya mevcut process'i sonlandırın.

#### Audio Yüklenemiyor
```
AUDIO_LOAD_ERROR
```
**Çözüm:** Dosyanın geçerli bir WAV dosyası olduğundan emin olun.

---

## 📊 Test Sonuçları

### Başarılı Test Çıktısı

Tüm testler geçtiğinde:
```
============================= test session starts =============================
collected 25 items

tests/test_api_properties.py::test_property_audio_format_validation PASSED
tests/test_api_properties.py::test_property_duration_validation PASSED
tests/test_api_properties.py::test_property_upload_round_trip PASSED
tests/test_api_properties.py::test_property_rate_limiting PASSED
tests/test_api_properties.py::test_property_processing_timeout PASSED
tests/test_api_properties.py::test_property_error_status_codes PASSED
tests/test_api_properties.py::test_property_error_logging_completeness PASSED
tests/test_api_properties.py::test_property_input_validation PASSED
...

============================= 25 passed in 45.23s =============================
```

---

## 🎯 Önerilen Test Sırası

1. ✅ **Health Check** - Server çalışıyor mu?
2. ✅ **Unit Testler** - Tüm modüller doğru çalışıyor mu?
3. ✅ **Integration Test** - End-to-end workflow çalışıyor mu?
4. ✅ **Manuel Test** - Gerçek audio dosyası ile test
5. ✅ **Error Senaryoları** - Hata durumları doğru handle ediliyor mu?

---

## 📚 Ek Kaynaklar

- [API Dokümantasyonu](API_DOCUMENTATION.md)
- [Property Testler](tests/test_api_properties.py)
- [Integration Test](test_api_integration.py)

