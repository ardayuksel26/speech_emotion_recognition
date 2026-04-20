"""
Test3/run_test.py
"""
import subprocess, sys, os, time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("="*60)
print("  BAŞLIYOR: Blend Oranı Testi")
print("="*60 + "\n")

start = time.time()
ret = subprocess.run([sys.executable, os.path.join(BASE_DIR, "test_blend_ratios.py")])
elapsed = time.time() - start

status = "✓ BAŞARILI" if ret.returncode == 0 else "✗ BAŞARISIZ"
print(f"\n  {status} — {elapsed:.1f}s")
