import whisperx

# Ayarlar
device = "cpu" # Eğer NVIDIA ekran kartın varsa "cuda" yaparsan çok daha hızlı çalışır
audio_file = "test_sesi.wav" # Test edeceğin ses dosyasının yolu

print("Ses yükleniyor ve metne çevriliyor...")
# 1. Temel modeli yükle (Türkçe algılaması için)
model = whisperx.load_model("base", device)

# 2. Sesi yükle ve deşifre et
audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, language="tr")

print("Kelimeler zaman damgaları ile eşleştiriliyor...")
# 3. Hizalama (Alignment) modelini yükle ve kelime sürelerini bul
model_a, metadata = whisperx.load_align_model(language_code="tr", device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

# Sonuçları ekrana yazdır
print("\n--- SONUÇLAR ---")
for segment in result["segments"]:
    for word in segment.get("words", []):
        start = word.get('start', 'Bulunamadı')
        end = word.get('end', 'Bulunamadı')
        text = word.get('word', '')
        print(f"Kelime: '{text}' | Başlangıç: {start}s | Bitiş: {end}s")