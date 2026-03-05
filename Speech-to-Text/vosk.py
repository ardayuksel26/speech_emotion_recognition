import wave
import json
from vosk import Model, KaldiRecognizer

model_path = "vosk-model-small-tr-0.3" # İndirip klasöre çıkardığın modelin adı
audio_file = "test_sesi.wav" # Önemli: Ses dosyası 16kHz, Mono WAV olmalı!

print("Model yükleniyor...")
model = Model(model_path)

with wave.open(audio_file, "rb") as wf:
    # Ses formatı kontrolü
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        print("HATA: Ses dosyası Mono ve PCM WAV formatında olmalıdır.")
        exit()
    
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True) # Kelime bazlı zaman damgalarını aktif eder

    print("Ses işleniyor...")
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        rec.AcceptWaveform(data)

    # İşlem bitince sonuçları al
    result = json.loads(rec.FinalResult())
    
    print("\n--- SONUÇLAR ---")
    if "result" in result:
        for word in result["result"]:
            print(f"Kelime: '{word['word']}' | Başlangıç: {word['start']}s | Bitiş: {word['end']}s")