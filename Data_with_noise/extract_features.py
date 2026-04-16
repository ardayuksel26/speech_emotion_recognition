"""
Gürültülü ses dosyalarından OpenSMILE IS10 özellik çıkarma.
Train1 ile aynı özellik seti (1582 özellik) kullanılır.

Eğitim kaynakları:
  - Orijinal sesler : TurEV-DB/Sound Source
  - Gürültülü sesler: Data_with_noise/<Duygu>/white_low
                                              white_medium
                                              pink_medium
                                              background_cafe
  (white_high kasıtlı olarak EKLENMEDİ)

Çıktı: Data_with_noise/Extracted CSV/angry.csv, calm.csv, happy.csv, sad.csv
"""

import os
import pandas as pd
import opensmile
from tqdm import tqdm

# --- AYARLAR ---
ORIGINAL_SOURCE = "TurEV-DB/Sound Source"
NOISY_SOURCE    = "Data_with_noise"
OUTPUT_DIR      = "Data_with_noise/Extracted CSV"

EMOTIONS    = ["Angry", "Calm", "Happy", "Sad"]
NOISE_TYPES = ["white_low", "white_medium", "pink_medium", "background_cafe"]

# OpenSMILE — Train1 ile aynı feature set
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.IS10,
    feature_level=opensmile.FeatureLevel.Functionals,
)


def build_file_list(emotion: str) -> list[str]:
    """Orijinal + gürültülü ses dosyalarının tam yollarını döndürür."""
    files = []

    # Orijinal
    orig_dir = os.path.join(ORIGINAL_SOURCE, emotion)
    if os.path.isdir(orig_dir):
        for f in sorted(os.listdir(orig_dir)):
            if f.lower().endswith(".wav"):
                files.append(os.path.join(orig_dir, f))

    # Gürültülü
    for noise_type in NOISE_TYPES:
        noisy_dir = os.path.join(NOISY_SOURCE, emotion, noise_type)
        if os.path.isdir(noisy_dir):
            for f in sorted(os.listdir(noisy_dir)):
                if f.lower().endswith(".wav"):
                    files.append(os.path.join(noisy_dir, f))

    return files


def extract_all():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  OpenSMILE IS10 Özellik Çıkarma — Orijinal + Gürültülü")
    print("=" * 60)
    print(f"Özellik sayısı : {len(smile.feature_names)}")
    print(f"Gürültü türleri: {NOISE_TYPES}")
    print()

    total_ok  = 0
    total_err = 0

    for emotion in EMOTIONS:
        emotion_lower = emotion.lower()
        file_list     = build_file_list(emotion)

        print(f"[{emotion}] {len(file_list)} dosya işlenecek...")

        rows = []
        for file_path in tqdm(file_list, desc=f"  {emotion}", unit="ses"):
            try:
                feat_df = smile.process_file(file_path)
                # process_file MultiIndex döndürür; satırı düzleştir
                row = feat_df.iloc[0].to_dict()
                rows.append(row)
                total_ok += 1
            except Exception as e:
                print(f"\n  [HATA] {file_path}: {e}")
                total_err += 1

        if rows:
            df       = pd.DataFrame(rows)
            out_path = os.path.join(OUTPUT_DIR, f"{emotion_lower}.csv")
            df.to_csv(out_path, index=False)
            print(f"  -> {len(rows)} örnek, {len(df.columns)} özellik → {out_path}\n")
        else:
            print(f"  [UYARI] {emotion} için hiç örnek üretilemedi!\n")

    print("=" * 60)
    print(f"  TAMAMLANDI")
    print(f"  Başarılı : {total_ok} dosya")
    print(f"  Hatalı   : {total_err} dosya")
    print(f"  Çıktı    : {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    extract_all()
