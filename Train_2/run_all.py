"""
Train_2/run_all.py
Adımları sırayla çalıştırır:
  1. Özellik çıkarma (All_Sounds/Train_Sounds → Extracted_CSV)
  2. Random Forest, LightGBM, XGBoost, CatBoost, Gradient Boosting eğitimi → Models_2
"""

import subprocess
import sys
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ("Özellik Çıkarma (IS10)",      "extract_features.py"),
    ("Random Forest",                "train_random_forest.py"),
    ("LightGBM",                     "train_lightgbm.py"),
    ("XGBoost",                      "train_xgboost.py"),
    ("CatBoost",                     "train_catboost.py"),
    ("Gradient Boosting",            "train_gradient_boosting.py"),
]

results = []

for label, script in STEPS:
    script_path = os.path.join(BASE_DIR, script)
    print(f"\n{'='*60}")
    print(f"  BAŞLIYOR: {label}")
    print(f"{'='*60}\n")

    start = time.time()
    ret = subprocess.run([sys.executable, script_path])
    elapsed = time.time() - start

    status = "✓ BAŞARILI" if ret.returncode == 0 else "✗ BAŞARISIZ"
    results.append((label, status, elapsed))
    print(f"\n  {status} — {elapsed:.1f}s")

print(f"\n{'='*60}")
print("  ÖZET")
print(f"{'='*60}")
for label, status, elapsed in results:
    print(f"  {status}  {label:<30} {elapsed:>7.1f}s")
print(f"{'='*60}\n")
