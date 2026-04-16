import os
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

# --- AYARLAR ---
SOURCE_PATH = "TurEV-DB/Sound Source"
OUTPUT_PATH = "Data_with_noise"
SAMPLE_RATE = 22050

EMOTIONS = ["Angry", "Calm", "Happy", "Sad"]

# Her ses için uygulanacak gürültü türleri ve parametreleri
# Her tür, orijinal sese ek olarak ayrı bir dosya üretir.
NOISE_CONFIGS = [
    {
        "name": "white_low",        # Düşük seviyeli beyaz gürültü
        "type": "white",
        "snr_db": 30,               # Sinyal/Gürültü oranı (dB) — yüksek = az gürültü
    },
    {
        "name": "white_medium",     # Orta seviyeli beyaz gürültü
        "type": "white",
        "snr_db": 15,
    },
    {
        "name": "white_high",       # Yüksek seviyeli beyaz gürültü
        "type": "white",
        "snr_db": 5,
    },
    {
        "name": "pink_medium",      # Orta seviyeli pembe gürültü (daha doğal)
        "type": "pink",
        "snr_db": 15,
    },
    {
        "name": "background_cafe",  # Kafeterya ortamı simülasyonu (bant sınırlı gürültü)
        "type": "bandpass",
        "snr_db": 12,
        "low_hz": 200,
        "high_hz": 3400,
    },
]


def generate_white_noise(length: int) -> np.ndarray:
    """Standart beyaz (Gaussian) gürültü üretir."""
    return np.random.randn(length).astype(np.float32)


def generate_pink_noise(length: int) -> np.ndarray:
    """
    Pembe (1/f) gürültü üretir.
    Beyaza kıyasla alçak frekansları daha güçlüdür — insan kulağına daha doğal gelir.
    Voss-McCartney algoritması (FFT tabanlı).
    """
    white = np.fft.rfft(np.random.randn(length))
    freqs = np.fft.rfftfreq(length)
    freqs[0] = 1e-6                    # 0'a bölmeden kaçın
    pink_filter = 1.0 / np.sqrt(freqs)
    pink = np.fft.irfft(white * pink_filter, n=length)
    pink = pink.astype(np.float32)
    return pink / (np.max(np.abs(pink)) + 1e-9)


def generate_bandpass_noise(length: int, sr: int, low_hz: float, high_hz: float) -> np.ndarray:
    """
    Belirli frekans bandında gürültü üretir.
    Örn: 200–3400 Hz → telefon/kafeterya ortamı simülasyonu.
    """
    white = np.random.randn(length).astype(np.float32)
    # Butterworth bant geçiren filtre
    from scipy.signal import butter, sosfilt
    sos = butter(4, [low_hz, high_hz], btype='bandpass', fs=sr, output='sos')
    filtered = sosfilt(sos, white)
    filtered = filtered.astype(np.float32)
    return filtered / (np.max(np.abs(filtered)) + 1e-9)


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Verilen SNR değerine göre gürültüyü sinyal üzerine ekler.
    SNR (dB) = 10 * log10(P_signal / P_noise)
    """
    # Gürültüyü sinyal uzunluğuna eşitle
    if len(noise) < len(signal):
        repeats = int(np.ceil(len(signal) / len(noise)))
        noise = np.tile(noise, repeats)
    noise = noise[: len(signal)]

    signal_power = np.mean(signal ** 2) + 1e-9
    noise_power  = np.mean(noise ** 2)  + 1e-9

    # İstenen SNR'a göre gürültü ölçeği
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / noise_power)

    noisy = signal + scale * noise
    # Kırpma (clipping) önlemek için normalize et
    peak = np.max(np.abs(noisy))
    if peak > 1.0:
        noisy /= peak
    return noisy.astype(np.float32)


def add_noise(signal: np.ndarray, config: dict, sr: int) -> np.ndarray:
    """Konfigürasyona göre uygun gürültüyü üretip sinyale karıştırır."""
    noise_type = config["type"]
    snr_db     = config["snr_db"]

    if noise_type == "white":
        noise = generate_white_noise(len(signal))
    elif noise_type == "pink":
        noise = generate_pink_noise(len(signal))
    elif noise_type == "bandpass":
        noise = generate_bandpass_noise(
            len(signal), sr,
            low_hz=config["low_hz"],
            high_hz=config["high_hz"]
        )
    else:
        raise ValueError(f"Bilinmeyen gürültü türü: {noise_type}")

    return mix_at_snr(signal, noise, snr_db)


def process_all():
    total_created = 0
    total_skipped = 0

    print("=" * 55)
    print("  Yapay Gürültü Ekleme — TurEV-DB Veri Artırma Aracı")
    print("=" * 55)
    print(f"Kaynak  : {SOURCE_PATH}")
    print(f"Çıktı   : {OUTPUT_PATH}")
    print(f"Gürültü türleri: {[c['name'] for c in NOISE_CONFIGS]}")
    print()

    for emotion in EMOTIONS:
        src_dir = os.path.join(SOURCE_PATH, emotion)
        if not os.path.isdir(src_dir):
            print(f"[UYARI] Klasör bulunamadı, atlanıyor: {src_dir}")
            continue

        wav_files = [f for f in os.listdir(src_dir) if f.lower().endswith(".wav")]
        print(f"[{emotion}] {len(wav_files)} dosya bulundu, {len(NOISE_CONFIGS)} gürültü türü uygulanacak...")

        for cfg in NOISE_CONFIGS:
            out_dir = os.path.join(OUTPUT_PATH, emotion, cfg["name"])
            os.makedirs(out_dir, exist_ok=True)

        for wav_file in tqdm(wav_files, desc=f"  {emotion}", unit="ses"):
            src_path = os.path.join(src_dir, wav_file)

            try:
                signal, sr = librosa.load(src_path, sr=SAMPLE_RATE, mono=True)
            except Exception as e:
                print(f"\n  [HATA] {wav_file} yüklenemedi: {e}")
                continue

            base_name = os.path.splitext(wav_file)[0]

            for cfg in NOISE_CONFIGS:
                out_filename = f"{base_name}_{cfg['name']}.wav"
                out_path = os.path.join(OUTPUT_PATH, emotion, cfg["name"], out_filename)

                if os.path.exists(out_path):
                    total_skipped += 1
                    continue

                try:
                    noisy_signal = add_noise(signal, cfg, sr)
                    sf.write(out_path, noisy_signal, sr, subtype="PCM_16")
                    total_created += 1
                except Exception as e:
                    print(f"\n  [HATA] {wav_file} — {cfg['name']}: {e}")

    print()
    print("=" * 55)
    print(f"  TAMAMLANDI")
    print(f"  Oluşturulan : {total_created} dosya")
    print(f"  Atlanan     : {total_skipped} dosya (zaten mevcut)")
    print("=" * 55)

    # Özet tablo
    print("\nDosya sayıları (duygu × gürültü türü):\n")
    header = f"{'Duygu':<10}" + "".join(f"{c['name']:<18}" for c in NOISE_CONFIGS)
    print(header)
    print("-" * len(header))
    for emotion in EMOTIONS:
        row = f"{emotion:<10}"
        for cfg in NOISE_CONFIGS:
            d = os.path.join(OUTPUT_PATH, emotion, cfg["name"])
            count = len([f for f in os.listdir(d) if f.endswith(".wav")]) if os.path.isdir(d) else 0
            row += f"{count:<18}"
        print(row)


if __name__ == "__main__":
    process_all()
