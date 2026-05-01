"""
Test2/run_test.py
1. Test_Sounds'tan IS10 özellik çıkar (Test_CSV yoksa)
2. 5 modeli test et, sonuçları Test2/Results'a kaydet
"""

import subprocess
import sys
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(BASE_DIR, ".."))

TEST_CSV_DIR = os.path.join(PROJECT_ROOT, "All_Sounds", "Test_CSV")
STEPS = []

# Özellik çıkarma — CSV zaten varsa atla
csvs_exist = all(
    os.path.exists(os.path.join(TEST_CSV_DIR, f))
    for f in ["angry.csv", "calm.csv", "happy.csv", "sad.csv"]
)
if not csvs_exist:
    STEPS.append(("Özellik Çıkarma (Test IS10)", "extract_test_features.py"))
else:
    print("Test CSV'leri zaten mevcut, özellik çıkarma adımı atlandı.\n")

STEPS.append(("Model Testi", "test_models.py"))

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
    print(f"  {status}  {label:<35} {elapsed:>7.1f}s")
print(f"{'='*60}\n")
