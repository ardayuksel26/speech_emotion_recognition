import os
import glob
import subprocess

scripts = sorted(glob.glob("train_*.py"))
for script in scripts:
    print(f"\n{'='*50}\nÇALIŞTIRILIYOR: {script}\n{'='*50}")
    subprocess.run(["python", script])
