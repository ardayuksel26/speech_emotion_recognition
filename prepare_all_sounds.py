import os
import shutil
import random

SEED = 42
TRAIN_RATIO = 0.8

BASE = "C:/Users/arday/OneDrive/Desktop/Speech_Emotion_Recognition_Project"
CLEAN_SRC = os.path.join(BASE, "TurEV-DB/Sound Source")
NOISY_SRC = os.path.join(BASE, "Data_with_noise")
TRAIN_DST = os.path.join(BASE, "All_Sounds/Train_Sounds")
TEST_DST  = os.path.join(BASE, "All_Sounds/Test_Sounds")

EMOTIONS   = ["Angry", "Calm", "Happy", "Sad"]
NOISE_TYPES = ["background_cafe", "pink_medium", "white_high", "white_low", "white_medium"]

os.makedirs(TRAIN_DST, exist_ok=True)
os.makedirs(TEST_DST,  exist_ok=True)

random.seed(SEED)

total_train = 0
total_test  = 0
missing     = []

for emotion in EMOTIONS:
    clean_folder = os.path.join(CLEAN_SRC, emotion)
    files = sorted([f for f in os.listdir(clean_folder) if f.endswith(".wav")])

    random.shuffle(files)
    split = int(len(files) * TRAIN_RATIO)
    train_files = files[:split]
    test_files  = files[split:]

    for dst_folder, file_list, label in [
        (TRAIN_DST, train_files, "TRAIN"),
        (TEST_DST,  test_files,  "TEST"),
    ]:
        for fname in file_list:
            stem = fname.replace(".wav", "")  # e.g. 1157_kz_acik

            # --- clean ---
            src = os.path.join(clean_folder, fname)
            dst = os.path.join(dst_folder, f"{emotion}_{stem}_clean.wav")
            shutil.copy2(src, dst)
            if label == "TRAIN":
                total_train += 1
            else:
                total_test += 1

            # --- noisy ---
            for noise in NOISE_TYPES:
                noisy_fname = f"{stem}_{noise}.wav"
                src = os.path.join(NOISY_SRC, emotion, noise, noisy_fname)
                if not os.path.exists(src):
                    missing.append(src)
                    continue
                dst = os.path.join(dst_folder, f"{emotion}_{stem}_{noise}.wav")
                shutil.copy2(src, dst)
                if label == "TRAIN":
                    total_train += 1
                else:
                    total_test += 1

    print(f"{emotion}: {len(train_files)} train utterance, {len(test_files)} test utterance")

print(f"\nToplam Train dosyası : {total_train}")
print(f"Toplam Test dosyası  : {total_test}")
if missing:
    print(f"\nEksik bulunan {len(missing)} dosya:")
    for m in missing[:10]:
        print(" ", m)
else:
    print("\nTüm kirli sesler bulundu.")
