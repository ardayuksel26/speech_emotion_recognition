"""
Speech-to-Text Service Module
Vosk ve WhisperX motorlarını kullanarak kelime bazlı zaman damgaları döndürür.
"""

import os
import wave
import json
import logging
import tempfile
import subprocess

logger = logging.getLogger('SER_API')

# ============================================================================
# VOSK MODEL PATH (proje kökündeki Speech-to-Text klasöründen yüklenir)
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOSK_MODEL_PATH = os.path.join(BASE_DIR, '..', 'Speech-to-Text', 'vosk-model-small-tr-0.3')

# Lazy-loaded model instances
_vosk_model = None
_whisperx_model = None


def _ensure_wav_format(audio_path: str) -> str:
    """
    Ses dosyasının 16kHz Mono PCM WAV formatında olduğundan emin olur.
    Gerekirse ffmpeg ile dönüştürür.
    """
    try:
        with wave.open(audio_path, 'rb') as wf:
            if wf.getnchannels() == 1 and wf.getsampwidth() == 2 and wf.getframerate() == 16000:
                return audio_path
    except Exception:
        pass

    # ffmpeg ile dönüştür
    converted_path = audio_path.replace('.wav', '_16k.wav')
    if converted_path == audio_path:
        converted_path = audio_path + '_16k.wav'

    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', audio_path,
            '-ar', '16000', '-ac', '1', '-sample_fmt', 's16',
            converted_path
        ], capture_output=True, check=True, timeout=30)
        return converted_path
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning(f"ffmpeg dönüştürme başarısız: {e}. Orijinal dosya kullanılacak.")
        return audio_path


def transcribe_with_vosk(audio_path: str) -> list:
    """
    Vosk ile ses dosyasını kelime kelime ayrıştırır.
    
    Returns:
        [{"word": "merhaba", "start": 0.5, "end": 1.2}, ...]
    """
    global _vosk_model

    try:
        from vosk import Model, KaldiRecognizer, SetLogLevel
        SetLogLevel(-1) # Vosk'un C++ loglarını tamamen gizle
    except ImportError:
        raise RuntimeError("Vosk kütüphanesi yüklü değil. 'pip install vosk' komutunu çalıştırın.")

    # Model'i lazy-load et
    if _vosk_model is None:
        if not os.path.exists(VOSK_MODEL_PATH):
            raise RuntimeError(f"Vosk model dosyası bulunamadı: {VOSK_MODEL_PATH}")
        logger.info("Vosk modeli yükleniyor...")
        _vosk_model = Model(VOSK_MODEL_PATH)
        logger.info("Vosk modeli başarıyla yüklendi.")

    # Ses formatını kontrol et ve gerekirse dönüştür
    wav_path = _ensure_wav_format(audio_path)
    cleanup_converted = wav_path != audio_path

    try:
        with wave.open(wav_path, 'rb') as wf:
            rec = KaldiRecognizer(_vosk_model, wf.getframerate())
            rec.SetWords(True)

            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                rec.AcceptWaveform(data)

            result = json.loads(rec.FinalResult())

        words = []
        if 'result' in result:
            for item in result['result']:
                words.append({
                    'word': item['word'],
                    'start': round(float(item['start']), 3),
                    'end': round(float(item['end']), 3)
                })

        return words

    finally:
        if cleanup_converted and os.path.exists(wav_path):
            os.remove(wav_path)


def transcribe_with_whisperx(audio_path: str, device: str = "cpu") -> list:
    """
    WhisperX ile ses dosyasını kelime kelime ayrıştırır.
    
    Args:
        audio_path: Ses dosyası yolu
        device: "cpu" veya "cuda"
    
    Returns:
        [{"word": "merhaba", "start": 0.5, "end": 1.2}, ...]
    """
    global _whisperx_model

    try:
        import whisperx
        import logging
        logging.getLogger("whisperx").setLevel(logging.ERROR)
        logging.getLogger("whisperx.vads.pyannote").setLevel(logging.ERROR)
        logging.getLogger("whisperx.asr").setLevel(logging.ERROR)
        logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    except ImportError:
        raise RuntimeError("WhisperX kütüphanesi yüklü değil. 'pip install whisperx' komutunu çalıştırın.")

    # Model'i lazy-load et
    if _whisperx_model is None:
        logger.info("WhisperX modeli yükleniyor...")
        # CPU için int8 zorunludur çünkü float16 desteklenmez. CUDA varsa float16 veya float32 kullanılabilir.
        compute_type = "int8" if device == "cpu" else "float16" 
        _whisperx_model = whisperx.load_model("base", device, compute_type=compute_type)
        logger.info("WhisperX modeli başarıyla yüklendi.")

    # 1. Sesi yükle — ffmpeg yerine librosa kullan (Windows uyumluluğu)
    try:
        audio = whisperx.load_audio(audio_path)
    except (FileNotFoundError, OSError) as e:
        # ffmpeg yoksa sessizce librosa fallback yap
        import librosa
        import numpy as np
        audio_np, sr = librosa.load(audio_path, sr=16000, mono=True)
        audio = audio_np.astype(np.float32)

    # 2. Deşifre et
    result = _whisperx_model.transcribe(audio, language="tr")

    # 3. Hizalama (Alignment) modelini yükle ve kelime sürelerini bul
    model_a, metadata = whisperx.load_align_model(language_code="tr", device=device)
    result = whisperx.align(
        result["segments"], model_a, metadata, audio, device,
        return_char_alignments=False
    )

    # 4. Kelime bazlı sonuçları çıkar
    words = []
    for segment in result.get("segments", []):
        for word_info in segment.get("words", []):
            start = word_info.get('start')
            end = word_info.get('end')
            text = word_info.get('word', '')

            if start is not None and end is not None and text:
                words.append({
                    'word': text.strip(),
                    'start': round(float(start), 3),
                    'end': round(float(end), 3)
                })

    return words


def transcribe(audio_path: str, engine: str = "vosk") -> list:
    """
    Ana giriş noktası. Motor seçimine göre uygun fonksiyonu çağırır.
    
    Args:
        audio_path: Ses dosyası yolu
        engine: "vosk" veya "whisperx"
    
    Returns:
        [{"word": "...", "start": float, "end": float}, ...]
    """
    if engine == "vosk":
        return transcribe_with_vosk(audio_path)
    elif engine == "whisperx":
        return transcribe_with_whisperx(audio_path)
    else:
        raise ValueError(f"Bilinmeyen STT motoru: {engine}. 'vosk' veya 'whisperx' kullanın.")
