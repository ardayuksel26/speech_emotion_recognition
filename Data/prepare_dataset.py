import os
import librosa
import numpy as np
import soundfile as sf
import json
from tqdm import tqdm

# --- AYARLAR (CONFIGURATION) ---
SOURCE_PATH = "TurEV-DB/Sound Source"   # Ham seslerin olduğu yer
DATA_ROOT = "Data"                      # Çıktıların kaydedileceği yer

SAMPLE_RATE = 22050                     # Standart Hz
DURATION = 3                            # Sabitleme süresi (En uzun ses 2.47sn olduğu için 3sn güvenli)
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

# Hedef Duygular ve Etiketleri
# Klasör/Dosya isimlerinde bu kelimeler aranacak
EMOTIONS = {
    "angry": 0,
    "calm": 1,
    "happy": 2,
    "sad": 3
}

def preprocess_and_save(file_path, emotion_name, filename):
    """
    1. Sesi yükler.
    2. Sessizlikleri kırpar (VAD).
    3. Kontrol için .wav olarak kaydeder.
    4. Model için Mel-Spectrogram'a çevirir.
    """
    # 1. Sesi Yükle
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    
    # 2. VAD (Sessizlik Temizleme)
    # top_db=40 referans değerdir. -40dB altını sessizlik kabul eder.
    yt, _ = librosa.effects.trim(y, top_db=40)
    
    # --- KONTROL KAYDI (WAV) ---
    # Temizlenmiş sesi dinleyebilmen için Data/angry/dosya.wav gibi kaydeder.
    target_folder = os.path.join(DATA_ROOT, emotion_name)
    os.makedirs(target_folder, exist_ok=True)
    
    save_path = os.path.join(target_folder, filename)
    sf.write(save_path, yt, sr)
    # ---------------------------
    
    # 3. Boyut Eşitleme (Padding)
    # Ses 3 saniyeden kısaysa arkasını sessizlikle doldurur.
    if len(yt) > SAMPLES_PER_TRACK:
        yt = yt[:SAMPLES_PER_TRACK] # Çok uzunsa kes (Bizim veride bu olmayacak)
    else:
        padding = int(SAMPLES_PER_TRACK - len(yt))
        yt = np.pad(yt, (0, padding), mode='constant')
        
    # 4. Mel-Spectrogram Çıkarma (Resimleştirme)
    mel_spec = librosa.feature.melspectrogram(y=yt, sr=sr, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db

def prepare_dataset():
    X = [] # Özellikler (Resimler)
    y = [] # Etiketler (0, 1, 2, 3)
    
    # Ana veri klasörünü oluştur
    if not os.path.exists(DATA_ROOT):
        os.makedirs(DATA_ROOT)
        
    print(f"🚀 Veri hazırlama işlemi başlıyor...")
    print(f"📂 Kaynak: {SOURCE_PATH}")
    print(f"🎯 Hedef : {DATA_ROOT}")
    
    # Tüm dosyaları listele
    all_files = []
    for root, _, filenames in os.walk(SOURCE_PATH):
        for filename in filenames:
            if filename.endswith('.wav'):
                all_files.append((os.path.join(root, filename), filename))
                
    print(f"Toplam {len(all_files)} ses dosyası bulundu. İşleniyor...")
    
    # Dosyaları tek tek işle
    for file_path, filename in tqdm(all_files):
        # Dosya yolunda duygu ismini ara (büyük/küçük harf duyarsız)
        path_lower = file_path.lower()
        found_emotion = None
        
        for emotion_key in EMOTIONS:
            if emotion_key in path_lower:
                found_emotion = emotion_key
                break
        
        if found_emotion is not None:
            # İşle ve listeye ekle
            spec = preprocess_and_save(file_path, found_emotion, filename)
            X.append(spec)
            y.append(EMOTIONS[found_emotion])
            
    # --- VERİ DÖNÜŞÜMLERİ ---
    
    # 1. X verisini Numpy dizisine çevir ve Kanal Ekle (CNN formatı: Genişlik, Yükseklik, 1)
    X = np.array(X)
    X = X[..., np.newaxis]
    
    # 2. Y verisini One-Hot Encoding'e çevir
    # Örn: 0 -> [1, 0, 0, 0]
    y_integers = np.array(y)
    num_classes = len(EMOTIONS)
    y_one_hot = np.eye(num_classes)[y_integers]
    
    print("-" * 30)
    print(f"✅ İŞLEM TAMAMLANDI!")
    print(f"📊 Girdi Verisi (X): {X.shape}  (Örnek Sayısı, Mel, Zaman, Kanal)")
    print(f"🏷️  Etiket Verisi (y): {y_one_hot.shape} (Örnek Sayısı, Sınıf Sayısı)")
    
    # --- KAYIT ---
    
    # Model Girdileri
    np.save(os.path.join(DATA_ROOT, "features_spectrograms.npy"), X)
    np.save(os.path.join(DATA_ROOT, "labels_onehot.npy"), y_one_hot)
    
    # Rehber Dosya (Metadata)
    with open(os.path.join(DATA_ROOT, "metadata.json"), "w") as f:
        json.dump(EMOTIONS, f)
        
    print("-" * 30)
    print(f"Dosyalar '{DATA_ROOT}' klasörüne kaydedildi:")
    print("1. 🎵 [Klasörler]: angry, calm... (Temizlenmiş sesleri buradan dinleyebilirsin)")
    print("2. 🧠 features_spectrograms.npy (Model Girdisi)")
    print("3. 🔑 labels_onehot.npy (Cevap Anahtarı)")
    print("4. 📝 metadata.json (Duygu-Sayı Eşleşmesi)")

if __name__ == "__main__":
    prepare_dataset()