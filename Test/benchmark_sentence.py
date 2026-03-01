import os
import glob
import random
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import opensmile
import noisereduce as nr
import warnings
warnings.filterwarnings('ignore')

# Modeller
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Test Araçları
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

# --- AYARLAR ---
DB_DIR = "../TurEV-DB/Sound Source"
EMOTIONS = ["Angry", "Calm", "Happy", "Sad"]
NUM_SENTENCES_PER_EMOTION = 100 # Toplam 400 cümle olacak
SAMPLE_RATE = 16000
MIN_LEN_SEC = 3.0
MAX_LEN_SEC = 5.0

print(f"\n🎧 OpenSMILE motoru başlatılıyor...")
try:
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.IS10,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    print("✅ OpenSMILE başarıyla yüklendi!")
except Exception as e:
    print(f"❌ OpenSMILE yüklenemedi: {e}")
    exit(1)

def create_synthetic_sentence(files, sr=SAMPLE_RATE, min_len=MIN_LEN_SEC, max_len=MAX_LEN_SEC):
    """
    Kelime bazlı ses dosyalarını birleştirerek yapay bir cümle audiosu oluşturur.
    Random boşluklar eklentisi ve dış ses/gürültü ilavesi yapar.
    """
    # 2 ile 5 arası kelime seç (cümle uzunluğuna yetecek kadar)
    num_words = random.randint(2, 5)
    chosen = random.sample(files, min(num_words, len(files)))
    
    sentence_audio = []
    
    # Başlangıca ufak bir boşluk (0.1s - 0.4s)
    sentence_audio.append(np.zeros(int(random.uniform(0.1, 0.4) * sr)))
    
    for f in chosen:
        try:
            y, _ = librosa.load(f, sr=sr)
            # Sessizlik kısımlarını kırp ki kelimeler netleşsin
            y_trimmed, _ = librosa.effects.trim(y, top_db=25)
            sentence_audio.append(y_trimmed)
        except Exception:
            continue
            
        # Kelimeler arası rastgele boşluk (0.1s ile 0.6s arası)
        gap_len = random.uniform(0.1, 0.6)
        sentence_audio.append(np.zeros(int(gap_len * sr)))
        
    if not sentence_audio:
        return None
        
    final_audio = np.concatenate(sentence_audio)
    duration = len(final_audio) / sr
    
    # Uzunluk kontrolü (3-5 saniye kuralı)
    if duration < min_len:
        # Eksik kısmı sessizlik ve arkaplan gürültüsü ile doldur
        pad_len = int((min_len - duration) * sr)
        pad_audio = np.zeros(pad_len)
        final_audio = np.concatenate((final_audio, pad_audio))
        
    if (len(final_audio) / sr) > max_len:
        # En fazla max_len saniye olsun
        final_audio = final_audio[:int(max_len * sr)]
        
    # --- DIŞ SES (ROBUSTNESS) EKLENTİSİ ---
    # Gerçekçi dış ses için genliği randomize ediyoruz (SNR yaklaşık 15-20dB)
    noise_amp = random.uniform(0.005, 0.02)
    noise = np.random.normal(0, noise_amp, len(final_audio))
    final_audio = final_audio + noise
    
    # Normalizasyon (Clipping önlemek için)
    max_val = np.max(np.abs(final_audio))
    if max_val > 0:
        final_audio = final_audio / max_val
        
    return final_audio

def extract_features(audio_data, sr):
    """
    Ses verisinden Backend pipeline'ındaki gibi gürültü filtreleme uygulayıp
    OpenSMILE özellikleri çıkarır (1584 boyutlu vektör).
    """
    try:
        # 1. Gürültü Azaltma (Denoising - Backend app.py benzeri)
        reduced_audio = nr.reduce_noise(y=audio_data, sr=sr, prop_decrease=0.7)
        
        # 2. OpenSMILE Signal Processing
        # (signal shape expected to be (channels, samples) or (samples,) array)
        if reduced_audio.ndim == 1:
            reduced_audio = np.expand_dims(reduced_audio, axis=0) # (1, samples)
            
        df = smile.process_signal(reduced_audio, sr)
        
        if df is None or df.empty: return None
        
        features = df.to_numpy().flatten()
        if len(features) != 1582: return None
        
        # 1584 boyuta padding yapıyoruz
        features_padded = np.zeros(1584, dtype=np.float32)
        features_padded[1:1583] = features
        return features_padded
        
    except Exception as e:
        return None

def build_sentence_dataset():
    print("\n" + "="*70)
    print("🎬 ADIM 1: CÜMLE BAZLI VE GÜRÜLTÜLÜ VERİSETİ OLUŞTURULUYOR")
    print("="*70)
    
    X_list, y_list = [], []
    
    for emotion in EMOTIONS:
        folder = os.path.join(DB_DIR, emotion)
        if not os.path.exists(folder):
            print(f"HATA: {folder} bulunamadı!")
            continue
            
        wav_files = glob.glob(os.path.join(folder, "*.wav"))
        if len(wav_files) < 2:
            print(f"Uyarı: {emotion} için yeterli ses dosyası yok ({len(wav_files)}).")
            continue
            
        print(f"⏳ {emotion} klasöründen {NUM_SENTENCES_PER_EMOTION} sentetik cümle üretiliyor...")
        count = 0
        attempts = 0
        while count < NUM_SENTENCES_PER_EMOTION and attempts < NUM_SENTENCES_PER_EMOTION * 3:
            attempts += 1
            # Cümleyi sentezle
            audio_array = create_synthetic_sentence(wav_files, sr=SAMPLE_RATE)
            if audio_array is None: continue
            
            # Özellik çıkart
            feats = extract_features(audio_data=audio_array, sr=SAMPLE_RATE)
            
            if feats is not None:
                X_list.append(feats)
                y_list.append(emotion.lower())
                count += 1
                if count % 25 == 0:
                    print(f"  -> {count}/{NUM_SENTENCES_PER_EMOTION} tamamlandı.")
                    
    X = np.array(X_list)
    y = np.array(y_list)
    print(f"\n✅ Veriseti oluşturma tamamlandı. Toplam Cümle: {len(X)}")
    return X, y

def run_benchmark(X, y):
    print("\n" + "="*70)
    print("🚀 ADIM 2: OLUŞTURULAN CÜMLE VERİLERİ İLE MODELLER YARIŞIYOR")
    print("="*70)
    
    # Encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # --- MODELLER ---
    models = [
        ('CatBoost', CatBoostClassifier(iterations=300, learning_rate=0.08, depth=6, verbose=0, random_seed=42)),
        ('XGBoost', xgb.XGBClassifier(n_estimators=200, learning_rate=0.08, max_depth=6, use_label_encoder=False, eval_metric='mlogloss', random_state=42)),
        ('LightGBM', lgb.LGBMClassifier(n_estimators=200, learning_rate=0.08, max_depth=6, verbosity=-1, random_state=42)),
        ('Random Forest', RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)),
        ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ('SVM', SVC(kernel='rbf', probability=True, random_state=42)),
        ('MLP Neural Net', MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)),
        ('k-NN', KNeighborsClassifier(n_neighbors=5))
    ]

    results = []
    names = []

    print(f"{'MODEL':<20} | {'DURUM':<40}")
    print("-" * 70)

    # HER MODEL İÇİN DÖNGÜ (5-Fold Cross Validation)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models:
        print(f"\n   📍 {name} K-Fold Başlıyor:")
        fold_scores = []
        start_total = time.time()
        
        for i, (train_index, test_index) in enumerate(kfold.split(X, y_encoded), 1):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y_encoded[train_index], y_encoded[test_index]
            
            # Scaler - Her fold'da train datası ile fit et
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            
            start_fold = time.time()
            
            # Train and Predict
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            
            # Flatten fix for some models
            if hasattr(y_pred, 'flatten') and not isinstance(model, MLPClassifier):
                y_pred = y_pred.flatten()
                
            acc = accuracy_score(y_test, y_pred)
            fold_scores.append(acc)
            
            elapsed_fold = time.time() - start_fold
            print(f"      🔹 Fold {i}/5 | Validasyon: %{acc*100:.2f} | Süre: {elapsed_fold:.2f}sn")

        mean_score = np.mean(fold_scores)
        std_dev = np.std(fold_scores)
        elapsed_total = time.time() - start_total
        
        results.append(fold_scores)
        names.append(name)
        
        print(f"   ✅ {name} BİTTİ! -> Ort: %{mean_score*100:.2f} (+/- {std_dev:.4f}) | Toplam: {elapsed_total:.2f}sn")
        print("-" * 70)

    # --- GRAFİK ÇİZ ---
    print("\n📊 Cümle Benchmark Grafiği oluşturuluyor...")
    plt.figure(figsize=(12, 6))
    plt.boxplot(results, labels=names, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red', linewidth=2))
    plt.title('Modellerin Sentetik (Gürültülü) Cümle Üzerindeki Karşılaştırması (K-Fold)')
    plt.ylabel('Başarı Oranı (Accuracy)')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    chart_name = 'benchmark_sentence_results.png'
    plt.savefig(chart_name, dpi=300)
    print(f"📈 Grafik: {chart_name} olarak Test klasörüne kaydedildi.")
    
    # Text olarak kaydet
    txt_name = 'benchmark_sentence_results.txt'
    with open(txt_name, 'w', encoding='utf-8') as f:
        f.write("--- SENTETİK CÜMLE (ROBUST) BENCHMARK SONUÇLARI ---\n\n")
        f.write(f"Parametreler: Ortalama 3-5 saniye, {NUM_SENTENCES_PER_EMOTION} Cümle/Duygu\n")
        f.write("\n======================================================\n")
        for n, r in zip(names, results):
            mean_acc = np.mean(r) * 100
            f.write(f"- {n:<18} : %{mean_acc:.2f} Accuracy (+/- {np.std(r):.4f})\n")
    print(f"📝 Rapor: {txt_name} olarak kaydedildi.")
    print("🏁 İşlem Tamamlandı.")

if __name__ == "__main__":
    X, y = build_sentence_dataset()
    if X is not None and len(X) > 0:
        run_benchmark(X, y)
    else:
        print("HATA: Veriseti oluşturulamadı, benchmark başlatılamıyor.")
