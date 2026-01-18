from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
import librosa
import os
import shutil
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import warnings
import glob

warnings.filterwarnings('ignore')

# --- GLOBAL DEĞİŞKENLER ---
models = {}

# --- YARDIMCI FONKSİYONLAR ---
# Bu fonksiyon hem EĞİTİM hem de TAHMİN aşamasında ortak kullanılacak.
# Böylece boyut uyuşmazlığı hatası almayız.
def extract_features(audio, sr):
    features = []
    
    # 1. MFCC (Mel-frequency cepstral coefficients)
    # Sesin tınısını temsil eder
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    features.extend(np.mean(mfccs, axis=1))
    features.extend(np.std(mfccs, axis=1))
    
    # 2. Chroma
    # Sesin perdesini ve tonunu temsil eder
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))
    
    # 3. Mel Spectrogram
    # İnsan kulağının duyduğu frekansları temsil eder
    mel = librosa.feature.melspectrogram(y=audio, sr=sr)
    features.extend(np.mean(mel, axis=1))
    features.extend(np.std(mel, axis=1))
    
    # 4. Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    features.extend(np.mean(contrast, axis=1))
    features.extend(np.std(contrast, axis=1))
    
    # 5. Zero Crossing Rate (Gürültü/sessizlik oranı)
    zcr = librosa.feature.zero_crossing_rate(audio)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))
    
    # 6. RMS Energy (Ses şiddeti)
    rms = librosa.feature.rms(y=audio)
    features.append(np.mean(rms))
    features.append(np.std(rms))
    
    return np.array(features)

def train_model_from_raw_audio():
    """
    CSV yerine doğrudan Sound_Source klasöründeki ses dosyalarını okuyarak eğitir.
    Bu işlem biraz zaman alabilir ama en garantili yöntemdir.
    """
    base_path = "Sound_Source"
    
    # Klasör kontrolü
    if not os.path.exists(base_path):
        print(f"HATA: '{base_path}' klasörü bulunamadı!")
        print("Lütfen 'Sound_Source' klasörünü 'Backend' klasörünün içine taşıyın.")
        return None

    X = []
    y = []
    
    # Klasör yapısı: Sound_Source/Angry/audio1.wav gibi olmalı
    # Duygu klasörlerini bul (Büyük/küçük harf duyarlılığı için mapleyelim)
    target_emotions = {
        'angry': 'angry', 'Angry': 'angry',
        'calm': 'calm', 'Calm': 'calm',
        'happy': 'happy', 'Happy': 'happy',
        'sad': 'sad', 'Sad': 'sad'
    }
    
    print("Ses dosyaları taranıyor ve özellikler çıkarılıyor (Bu işlem 30-60sn sürebilir)...")
    
    file_count = 0
    
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        
        if os.path.isdir(folder_path) and folder_name in target_emotions:
            emotion_label = target_emotions[folder_name]
            
            # Klasördeki WAV dosyalarını al
            for file_name in os.listdir(folder_path):
                if file_name.lower().endswith('.wav'):
                    file_path = os.path.join(folder_path, file_name)
                    
                    try:
                        # Librosa ile yükle (Hız için süre kısıtlaması koyabiliriz ama kalite için full okuyoruz)
                        # sr=16000 konuşma için idealdir
                        audio, sr = librosa.load(file_path, sr=16000, duration=3.0) # İlk 3 saniyeyi al (hız için)
                        
                        # Özellik çıkar
                        features = extract_features(audio, sr)
                        
                        X.append(features)
                        y.append(emotion_label)
                        file_count += 1
                        
                        if file_count % 50 == 0:
                            print(f"{file_count} dosya işlendi...")
                            
                    except Exception as e:
                        print(f"Dosya okunamadı: {file_name} - {e}")
    
    if file_count == 0:
        print("HATA: Hiçbir ses dosyası bulunamadı!")
        return None
        
    print(f"Toplam {file_count} dosya ile eğitim başlıyor...")
    
    # Numpy array'e çevir
    X = np.array(X)
    y = np.array(y)
    
    # Eksik veri varsa doldur (NaN check)
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    
    # Label Encoding
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Model Eğitimi
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y_encoded)
    
    print(f"✅ Model başarıyla eğitildi! (Doğruluk: Eğitim verisi üzerinde test edilmedi)")
    
    return {
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'imputer': imputer
    }

# --- LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Sunucu başlatılıyor...")
    # Modeli raw audio'dan eğit
    artifacts = train_model_from_raw_audio()
    
    if artifacts:
        models["emotion_model"] = artifacts
    else:
        print("UYARI: Model yüklenemedi. Sound_Source klasörünü kontrol edin.")
        
    yield
    print("Sunucu kapatıldı.")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if "emotion_model" not in models:
        raise HTTPException(status_code=500, detail="Model yüklü değil. Sound_Source klasörü eksik olabilir.")
    
    temp_filename = f"temp_{file.filename}"
    
    try:
        # Dosyayı kaydet
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Sesi yükle
        audio, sr = librosa.load(temp_filename, sr=16000) # Duration yok, tüm sesi analiz et
        
        artifacts = models["emotion_model"]
        
        # Özellik çıkar (Eğitimdeki aynı fonksiyon!)
        features = extract_features(audio, sr)
        
        # Şekil düzeltme (1 örnek için)
        features = features.reshape(1, -1)
        
        # NaN kontrolü ve temizleme
        features = artifacts['imputer'].transform(features)
        
        # Scaling
        x_scaled = artifacts['scaler'].transform(features)

        # Tahmin
        probs = artifacts['model'].predict_proba(x_scaled)[0]
        pred_idx = np.argmax(probs)
        pred_label = artifacts['label_encoder'].classes_[pred_idx]
        confidence = np.max(probs)
        
        all_probs = {
            cls: float(prob) 
            for cls, prob in zip(artifacts['label_encoder'].classes_, probs)
        }

        return {
            "prediction": pred_label,
            "confidence": float(confidence),
            "all_probabilities": all_probs
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)