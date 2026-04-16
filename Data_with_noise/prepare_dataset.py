"""
CNN-LSTM için Mel-Spectrogram verisi hazırlama.

Eğitim kaynakları:
  - Orijinal sesler : TurEV-DB/Sound Source
  - Gürültülü sesler: Data_with_noise/<Duygu>/white_low, white_medium, pink_medium, background_cafe
  (white_high kasıtlı olarak EKLENMEDİ)

Çıktı:
  Data_with_noise/features_spectrograms.npy  — (N, 128, 130, 1)
  Data_with_noise/labels_onehot.npy          — (N, 4)
  Data_with_noise/metadata.json
"""

import os
import numpy as np
import librosa
import soundfile as sf
import json
from tqdm import tqdm

# --- AYARLAR ---
ORIGINAL_SOURCE = "TurEV-DB/Sound Source"
NOISY_SOURCE    = "Data_with_noise"
OUTPUT_DIR      = "Data_with_noise"

SAMPLE_RATE      = 22050
DURATION         = 3
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

EMOTIONS = {
    "Angry": 0,
    "Calm":  1,
    "Happy": 2,
    "Sad":   3
}

NOISE_TYPES = ["white_low", "white_medium", "pink_medium", "background_cafe"]


def compute_melspec(y: np.ndarray, sr: int) -> np.ndarray:
    """VAD + padding + Mel-Spectrogram."""
    y, _ = librosa.effects.trim(y, top_db=40)
    if len(y) >= SAMPLES_PER_TRACK:
        y = y[:SAMPLES_PER_TRACK]
    else:
        y = np.pad(y, (0, SAMPLES_PER_TRACK - len(y)), mode='constant')
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    return librosa.power_to_db(mel, ref=np.max)


def build_file_list() -> list[tuple[str, int]]:
    files = []
    for emotion, label in EMOTIONS.items():
        # Orijinal
        orig_dir = os.path.join(ORIGINAL_SOURCE, emotion)
        if os.path.isdir(orig_dir):
            for f in os.listdir(orig_dir):
                if f.lower().endswith(".wav"):
                    files.append((os.path.join(orig_dir, f), label))
        # Gürültülü
        for noise_type in NOISE_TYPES:
            noisy_dir = os.path.join(NOISY_SOURCE, emotion, noise_type)
            if os.path.isdir(noisy_dir):
                for f in os.listdir(noisy_dir):
                    if f.lower().endswith(".wav"):
                        files.append((os.path.join(noisy_dir, f), label))
    return files


def prepare():
    print("=" * 55)
    print("  CNN-LSTM Veri Hazırlama — Spectrogram (.npy)")
    print("=" * 55)

    file_list = build_file_list()
    print(f"Toplam {len(file_list)} ses dosyası işlenecek...\n")

    X, y = [], []
    for file_path, label in tqdm(file_list, unit="ses"):
        try:
            sig, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
            mel = compute_melspec(sig, sr)
            X.append(mel)
            y.append(label)
        except Exception as e:
            print(f"\n  [HATA] {file_path}: {e}")

    X = np.array(X)[..., np.newaxis]           # (N, 128, T, 1)
    y_int = np.array(y)
    y_ohe = np.eye(len(EMOTIONS))[y_int]        # (N, 4)

    np.save(os.path.join(OUTPUT_DIR, "features_spectrograms.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "labels_onehot.npy"), y_ohe)

    meta = {k.lower(): v for k, v in EMOTIONS.items()}
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f)

    print()
    print("=" * 55)
    print(f"  TAMAMLANDI")
    print(f"  X boyutu: {X.shape}")
    print(f"  y boyutu: {y_ohe.shape}")
    print(f"  Çıktı   : {OUTPUT_DIR}")
    print("=" * 55)


if __name__ == "__main__":
    prepare()
