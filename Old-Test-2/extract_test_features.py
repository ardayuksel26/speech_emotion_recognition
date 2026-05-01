"""
All_Sounds/Test_Sounds/ klasöründen OpenSMILE IS10 özellik çıkarma.
Çıktı: All_Sounds/Test_CSV/{angry,calm,happy,sad}.csv
"""

import os
import pandas as pd
import opensmile
from tqdm import tqdm

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(BASE_DIR, ".."))
TEST_SOUNDS  = os.path.join(PROJECT_ROOT, "All_Sounds", "Test_Sounds")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "All_Sounds", "Test_CSV")

EMOTIONS = ["Angry", "Calm", "Happy", "Sad"]

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.IS10,
    feature_level=opensmile.FeatureLevel.Functionals,
)


def extract_all():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  OpenSMILE IS10 — All_Sounds/Test_Sounds")
    print("=" * 60)
    print(f"Özellik sayısı : {len(smile.feature_names)}\n")

    total_ok = total_err = 0

    for emotion in EMOTIONS:
        files = sorted([
            os.path.join(TEST_SOUNDS, f)
            for f in os.listdir(TEST_SOUNDS)
            if f.lower().endswith(".wav") and f.startswith(emotion + "_")
        ])
        print(f"[{emotion}] {len(files)} dosya işlenecek...")

        rows = []
        for fp in tqdm(files, desc=f"  {emotion}", unit="ses"):
            try:
                row = smile.process_file(fp).iloc[0].to_dict()
                rows.append(row)
                total_ok += 1
            except Exception as e:
                print(f"\n  [HATA] {fp}: {e}")
                total_err += 1

        if rows:
            out = os.path.join(OUTPUT_DIR, f"{emotion.lower()}.csv")
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"  -> {len(rows)} örnek → {out}\n")

    print("=" * 60)
    print(f"  TAMAMLANDI — Başarılı: {total_ok}, Hatalı: {total_err}")
    print("=" * 60)


if __name__ == "__main__":
    extract_all()
