"""
All_Sounds/Train_Sounds/ klasöründen CNN-LSTM için Mel-Spectrogram hazırlama.
Çıktı: All_Sounds/features_spectrograms.npy — (N, 128, 130, 1)
        All_Sounds/labels_onehot.npy          — (N, 4)
"""

import os
import numpy as np
import librosa
import json
from tqdm import tqdm

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(BASE_DIR, ".."))
TRAIN_SOUNDS = os.path.join(PROJECT_ROOT, "All_Sounds", "Train_Sounds")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "All_Sounds")

SAMPLE_RATE       = 22050
DURATION          = 3
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

EMOTIONS = {"Angry": 0, "Calm": 1, "Happy": 2, "Sad": 3}


def compute_melspec(y: np.ndarray, sr: int) -> np.ndarray:
    y, _ = librosa.effects.trim(y, top_db=40)
    if len(y) >= SAMPLES_PER_TRACK:
        y = y[:SAMPLES_PER_TRACK]
    else:
        y = np.pad(y, (0, SAMPLES_PER_TRACK - len(y)), mode='constant')
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    return librosa.power_to_db(mel, ref=np.max)


def build_file_list() -> list[tuple[str, int]]:
    files = []
    for fname in sorted(os.listdir(TRAIN_SOUNDS)):
        if not fname.lower().endswith(".wav"):
            continue
        emotion = fname.split("_")[0]
        if emotion in EMOTIONS:
            files.append((os.path.join(TRAIN_SOUNDS, fname), EMOTIONS[emotion]))
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

    X = np.array(X)[..., np.newaxis]
    y_int = np.array(y)
    y_ohe = np.eye(len(EMOTIONS))[y_int]

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
