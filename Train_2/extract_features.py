"""
All_Sounds/Train_Sounds/ klasöründen OpenSMILE IS10 özellik çıkarma.
Dosya adı formatı: Emotion_speakerID_gender_word_noisetype.wav
Çıktı: All_Sounds/Extracted_CSV/{angry,calm,happy,sad}.csv
"""

import os
import pandas as pd
import opensmile
from tqdm import tqdm

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(BASE_DIR, ".."))
TRAIN_SOUNDS = os.path.join(PROJECT_ROOT, "All_Sounds", "Train_Sounds")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "All_Sounds", "Extracted_CSV")

EMOTIONS = ["Angry", "Calm", "Happy", "Sad"]

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.IS10,
    feature_level=opensmile.FeatureLevel.Functionals,
)


def build_file_list(emotion: str) -> list[str]:
    files = []
    for fname in sorted(os.listdir(TRAIN_SOUNDS)):
        if fname.lower().endswith(".wav") and fname.startswith(emotion + "_"):
            files.append(os.path.join(TRAIN_SOUNDS, fname))
    return files


def extract_all():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  OpenSMILE IS10 Özellik Çıkarma — All_Sounds/Train_Sounds")
    print("=" * 60)
    print(f"Özellik sayısı : {len(smile.feature_names)}")
    print()

    total_ok = total_err = 0

    for emotion in EMOTIONS:
        file_list = build_file_list(emotion)
        print(f"[{emotion}] {len(file_list)} dosya işlenecek...")

        rows = []
        for file_path in tqdm(file_list, desc=f"  {emotion}", unit="ses"):
            try:
                feat_df = smile.process_file(file_path)
                row = feat_df.iloc[0].to_dict()
                rows.append(row)
                total_ok += 1
            except Exception as e:
                print(f"\n  [HATA] {file_path}: {e}")
                total_err += 1

        if rows:
            df = pd.DataFrame(rows)
            out_path = os.path.join(OUTPUT_DIR, f"{emotion.lower()}.csv")
            df.to_csv(out_path, index=False)
            print(f"  -> {len(rows)} örnek, {len(df.columns)} özellik → {out_path}\n")

    print("=" * 60)
    print(f"  TAMAMLANDI — Başarılı: {total_ok}, Hatalı: {total_err}")
    print("=" * 60)


if __name__ == "__main__":
    extract_all()
