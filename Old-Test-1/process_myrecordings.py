import os
import glob
import random
import numpy as np
import soundfile as sf
import librosa
import opensmile
import pandas as pd
from tqdm import tqdm

# --- YOLLAR ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MY_REC_DIR = os.path.abspath(os.path.join(BASE_DIR, "MyRecordings"))
TARGET_DIR = os.path.join(MY_REC_DIR, "Sentence")

EMOTIONS = ["angry", "calm", "happy", "sad"]
SAMPLE_RATE = 16000
MIN_LEN_SEC = 3.0
MAX_LEN_SEC = 10.0

print(f"🎧 OpenSMILE (IS10) başlatılıyor...")
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.IS10,
    feature_level=opensmile.FeatureLevel.Functionals,
)
print("✅ OpenSMILE başarıyla yüklendi!")

def create_synthetic_sentence(files, sr=SAMPLE_RATE):
    # Kayıtlardan rastgele 6-12 kelime seç
    num_words = random.randint(6, 12)
    
    # 20 kelime varsa 6-12 tane alabiliriz
    chosen = random.sample(files, min(num_words, len(files)))
    
    sentence_audio = [np.zeros(int(random.uniform(0.05, 0.2) * sr))]
    
    for idx, f in enumerate(chosen):
        try:
            y, _ = librosa.load(f, sr=sr)
            y_trimmed, _ = librosa.effects.trim(y, top_db=30) 
            sentence_audio.append(y_trimmed)
        except Exception:
            continue
            
        if idx != len(chosen) - 1:
            # Rastgele kısa/uzun boşluklar
            gap_len = random.uniform(0.05, 0.15) if random.random() > 0.1 else random.uniform(0.3, 0.5)
            sentence_audio.append(np.zeros(int(gap_len * sr)))
        
    if not sentence_audio: return None
        
    final_audio = np.concatenate(sentence_audio)
    duration = len(final_audio) / sr
    
    # Çok kısaysa sessizlik ekle
    if duration < MIN_LEN_SEC:
        padding = np.zeros(int((MIN_LEN_SEC - duration) * sr) + int(0.2*sr))
        final_audio = np.concatenate((final_audio, padding))
        
    # Çok uzunsa kes
    if (len(final_audio) / sr) > MAX_LEN_SEC:
        final_audio = final_audio[:int(MAX_LEN_SEC * sr)]
        
    # Normalizasyon
    max_abs = np.max(np.abs(final_audio))
    if max_abs > 0:
        final_audio = final_audio / max_abs * random.uniform(0.8, 0.95)
        
    return final_audio

def process_my_recordings(num_sentences=20):
    print(f"\nSistem: Kullanıcı Kayıtları İşleniyor -> {TARGET_DIR}")
    
    csv_out_dir = os.path.join(TARGET_DIR, "Extracted CSV")
    os.makedirs(csv_out_dir, exist_ok=True)

    for emo in EMOTIONS:
        source_folder = os.path.join(MY_REC_DIR, emo)
        wav_files = glob.glob(os.path.join(source_folder, "*.wav"))
        
        wav_out_dir = os.path.join(TARGET_DIR, emo)
        os.makedirs(wav_out_dir, exist_ok=True)
            
        if not wav_files:
            print(f"UYARI: {source_folder} klasöründe .wav dosyası bulunamadı, atlanıyor.")
            continue
            
        print(f"⏳ {emo} için {num_sentences} cümle üretiliyor...")
        rows = []
        
        for i in tqdm(range(num_sentences), desc=f"{emo}"):
            audio_array = create_synthetic_sentence(wav_files, sr=SAMPLE_RATE)
            if audio_array is None: continue
            
            # Kaydet
            file_name = f"my_sentence_{emo}_{i+1:02d}.wav"
            file_path = os.path.join(wav_out_dir, file_name)
            sf.write(file_path, audio_array, SAMPLE_RATE)
            
            # Özellik Çıkarımı
            try:
                feat_df = smile.process_signal(audio_array, SAMPLE_RATE)
                row = feat_df.iloc[0].to_dict()
                row['speaker_id'] = 'user' # Kullanıcı etiketlemesi
                rows.append(row)
            except Exception as e:
                print(f"Hata ({file_name}): {e}")
                
        if rows:
            df_csv = pd.DataFrame(rows)
            csv_path = os.path.join(csv_out_dir, f"{emo}.csv")
            df_csv.to_csv(csv_path, index=False)
            print(f"✅ {emo} CSV kaydedildi: {csv_path}")

if __name__ == "__main__":
    process_my_recordings(num_sentences=20)
