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
SOUND_DIR = os.path.abspath(os.path.join(BASE_DIR, "../TurEV-DB/Sound Source"))

EMOTIONS = ["Angry", "Calm", "Happy", "Sad"]
SAMPLE_RATE = 16000
MIN_LEN_SEC = 3.0
MAX_LEN_SEC = 10.0

print(f"🎧 OpenSMILE başlatılıyor...")
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.IS10,
    feature_level=opensmile.FeatureLevel.Functionals,
)
print("✅ OpenSMILE başarıyla yüklendi!")

def create_synthetic_sentence(files, sr=SAMPLE_RATE):
    num_words = random.randint(6, 15)
    chosen = random.sample(files, min(num_words, len(files)))
    
    sentence_audio = [np.zeros(int(random.uniform(0.05, 0.2) * sr))]
    
    for idx, f in enumerate(chosen):
        try:
            y, _ = librosa.load(f, sr=sr)
            y_trimmed, _ = librosa.effects.trim(y, top_db=35) 
            sentence_audio.append(y_trimmed)
        except Exception:
            continue
            
        if idx != len(chosen) - 1:
            if random.random() < 0.10:
                gap_len = random.uniform(0.4, 0.6)
            else:
                gap_len = random.uniform(0.05, 0.2)
            sentence_audio.append(np.zeros(int(gap_len * sr)))
        
    if not sentence_audio: return None
        
    final_audio = np.concatenate(sentence_audio)
    duration = len(final_audio) / sr
    
    if duration < MIN_LEN_SEC:
        padding = np.zeros(int((MIN_LEN_SEC - duration) * sr) + int(0.2*sr))
        final_audio = np.concatenate((final_audio, padding))
        
    if (len(final_audio) / sr) > MAX_LEN_SEC:
        final_audio = final_audio[:int(MAX_LEN_SEC * sr)]
        fade_len = int(0.05 * sr)
        fade_out = np.linspace(1.0, 0.0, fade_len)
        final_audio[-fade_len:] = final_audio[-fade_len:] * fade_out
        
    final_audio = final_audio * random.uniform(0.8, 1.1) 
    
    if random.random() < 0.30: 
        final_audio += np.random.normal(0, random.uniform(0.001, 0.003), len(final_audio))
        
    if random.random() < 0.15:
        noise_dur = int(sr * random.uniform(0.05, 0.1))
        noise_start = random.randint(0, max(0, len(final_audio) - noise_dur))
        pop = np.random.normal(0, 0.02, noise_dur) * np.linspace(1, 0, noise_dur)
        final_audio[noise_start:noise_start+noise_dur] += pop

    max_abs = np.max(np.abs(final_audio))
    if max_abs > 1.0: 
        final_audio = np.clip(final_audio, -1.0, 1.0)
    elif max_abs > 0: 
        final_audio = final_audio / np.max(np.abs(final_audio)) * random.uniform(0.7, 0.95)
        
    return final_audio

def process_dataset(target_dir, pools, num_sentences):
    """
    pools dict formatı: {'Angry': [wav_files...], 'Calm': [wav_files...], ...}
    """
    print(f"\nSistem: Hedef Klasör: {target_dir}")
    
    csv_out_dir = os.path.join(target_dir, "Extracted CSV")
    if not os.path.exists(csv_out_dir):
        os.makedirs(csv_out_dir)

    for emo in EMOTIONS:
        wav_files = pools.get(emo, [])
        wav_out_dir = os.path.join(target_dir, emo.lower())
        
        if not os.path.exists(wav_out_dir):
            os.makedirs(wav_out_dir)
            
        if not wav_files:
            print(f"HATA: {emo} için kelime havuzu boş, atlanıyor.")
            continue
            
        print(f"⏳ {emo} kategorisinden {num_sentences} cümle üretiliyor... (Havuz boyutu: {len(wav_files)} ses)")
        count = 0
        attempts = 0
        
        rows = []
        
        with tqdm(total=num_sentences, desc=f"{emo}") as pbar:
            while count < num_sentences and attempts < num_sentences * 5:
                attempts += 1
                audio_array = create_synthetic_sentence(wav_files, sr=SAMPLE_RATE)
                if audio_array is None:
                    continue
                
                # WAV Olarak Kaydet
                file_name = f"sentence_{emo.lower()}_{count+1:02d}.wav"
                file_path = os.path.join(wav_out_dir, file_name)
                sf.write(file_path, audio_array, SAMPLE_RATE)
                
                # Özellik Çıkarımı
                try:
                    feat_df = smile.process_signal(audio_array, SAMPLE_RATE)
                    row = feat_df.iloc[0].to_dict()
                    rows.append(row)
                    count += 1
                    pbar.update(1)
                except Exception as e:
                    print(f"Özellik çıkarma hatası ({file_name}): {e}")
                
        # Duygu için CSV Kaydet
        if len(rows) > 0:
            df_csv = pd.DataFrame(rows)
            csv_path = os.path.join(csv_out_dir, f"{emo.lower()}.csv")
            df_csv.to_csv(csv_path, index=False)
            print(f"✅ {emo} için {len(df_csv)} satır başarıyla oluşturuldu ve -> {csv_path} alanına kaydedildi.")
        
    print(f"\n🎉 İşlem Tamamlandı: '{target_dir}' \n")

def process_dataset_by_speaker(target_dir, pools, num_sentences_per_speaker):
    """
    pools format: {ActorID: {Emotion: [files]}}
    """
    print(f"\nSistem: Hedef Klasör (Speaker-Aware): {target_dir}")
    csv_out_dir = os.path.join(target_dir, "Extracted CSV")
    os.makedirs(csv_out_dir, exist_ok=True)
    
    emotion_rows = {emo.lower(): [] for emo in EMOTIONS}

    for spk_id, emo_dict in pools.items():
        print(f"👤 Konuşmacı: {spk_id} işleniyor...")
        for emo_name, wav_files in emo_dict.items():
            if not wav_files: continue
            
            wav_out_dir = os.path.join(target_dir, emo_name.lower())
            os.makedirs(wav_out_dir, exist_ok=True)

            count = 0
            for i in range(num_sentences_per_speaker):
                audio_array = create_synthetic_sentence(wav_files, sr=SAMPLE_RATE)
                if audio_array is None: continue
                
                # Dosya ismi
                file_name = f"sentence_{spk_id}_{emo_name.lower()}_{i+1:02d}.wav"
                file_path = os.path.join(wav_out_dir, file_name)
                sf.write(file_path, audio_array, SAMPLE_RATE)
                
                # Özellik Çıkarımı
                try:
                    feat_df = smile.process_signal(audio_array, SAMPLE_RATE)
                    row = feat_df.iloc[0].to_dict()
                    row['speaker_id'] = spk_id # KRİTİK: Konuşmacı ID ekle
                    emotion_rows[emo_name.lower()].append(row)
                    count += 1
                except Exception as e:
                    print(f"Hata: {e}")
            
    # Her duygu için ayrı CSV kaydet
    for emo_name, rows in emotion_rows.items():
        if rows:
            df = pd.DataFrame(rows)
            csv_path = os.path.join(csv_out_dir, f"{emo_name}.csv")
            df.to_csv(csv_path, index=False)
            print(f"✅ {emo_name.upper()} CSV Kaydedildi: {len(df)} satır.")

def split_data_pools_by_speaker():
    """Kayıtları önce aktörlere, sonra duygulara göre ayırır."""
    test_speakers = ['6783', '7895']
    train_pools = {} # {spk_id: {emo: [files]}}
    test_pools = {}
    
    for emo in EMOTIONS:
        source_folder = os.path.join(SOUND_DIR, emo)
        wav_files = glob.glob(os.path.join(source_folder, "*.wav"))
        
        for f in wav_files:
            spk_id = os.path.basename(f).split('_')[0]
            pool = test_pools if spk_id in test_speakers else train_pools
            
            if spk_id not in pool: pool[spk_id] = {e: [] for e in EMOTIONS}
            pool[spk_id][emo].append(f)
            
    return train_pools, test_pools

if __name__ == "__main__":
    train_out = os.path.join(BASE_DIR, "sentencevoice_train")
    test_out = os.path.join(BASE_DIR, "sentencevoice_test")
    
    print("Veriler Konuşmacı Bazlı Havuzlara (Speaker-Centric) ayrılıyor...")
    train_p, test_p = split_data_pools_by_speaker()
    
    # 4 Train aktörü var. Her biri için 25er cümle üretsek 100 eder (toplam yine 400).
    print("\n=================== 1. AŞAMA: TRAIN ===================")
    process_dataset_by_speaker(train_out, train_p, num_sentences_per_speaker=25)
    
    # 2 Test aktörü var. Her biri için 20şer cümle üretsek 40 eder (toplam 160).
    print("\n=================== 2. AŞAMA: TEST ===================")
    process_dataset_by_speaker(test_out, test_p, num_sentences_per_speaker=20)
