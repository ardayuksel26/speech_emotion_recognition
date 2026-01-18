import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- AYARLAR ---
# Artık tek bir dosya değil, klasör yolunu veriyoruz
DATA_DIR = "TurEV-DB/Extracted CSV"  
MODEL_DIR = "Models"
MODEL_NAME = "random_forest_model.pkl"
SCALER_NAME = "scaler_rf.pkl"
LABEL_ENCODER_NAME = "label_encoder_rf.pkl"

# Duygu dosya isimleri (Dosya adı -> Etiket)
# Klasördeki dosya isimlerinle birebir aynı olmalı (büyük/küçük harf duyarlı)
EMOTION_FILES = {
    "angry.csv": "angry",
    "calm.csv": "calm",
    "happy.csv": "happy",
    "sad.csv": "sad"
}

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def load_data_from_multiple_files():
    """
    Ayrı ayrı duran CSV dosyalarını okur, etiketlerini ekler ve birleştirir.
    """
    print(f"Veriler okunuyor: {DATA_DIR} klasöründen...")
    
    all_dataframes = []
    
    for filename, emotion_label in EMOTION_FILES.items():
        file_path = os.path.join(DATA_DIR, filename)
        
        if os.path.exists(file_path):
            # CSV'yi oku
            df_temp = pd.read_csv(file_path)
            
            # Etiket sütunu ekle (angry.csv'den gelenlere 'angry' etiketi bas)
            # 'label' adında yeni bir sütun açıyoruz
            df_temp['label'] = emotion_label
            
            all_dataframes.append(df_temp)
            print(f"  -> {filename} yüklendi. ({len(df_temp)} veri, Etiket: {emotion_label})")
        else:
            print(f"UYARI: {filename} bulunamadı! ({file_path})")
            
    if not all_dataframes:
        print("HATA: Hiçbir veri dosyası yüklenemedi!")
        return None

    # Hepsini alt alta birleştir
    full_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"Toplam Birleştirilmiş Veri: {full_df.shape}")
    return full_df

def train_rf():
    # 1. Verileri Yükle ve Birleştir
    df = load_data_from_multiple_files()
    
    if df is None:
        return

    # 2. Ön İşleme (Preprocessing)
    print("Veri ön işleme yapılıyor...")
    
    target_col = 'label'
    
    # Özellik olmayan sütunları temizle
    # CSV'nin içinde 'filename' veya 'name' gibi metin sütunları varsa atalım
    drop_cols = [target_col]
    for col in ['filename', 'name', 'path', 'dosya_adi']: 
        if col in df.columns:
            drop_cols.append(col)
    
    X = df.drop(columns=drop_cols, errors='ignore')
    y = df[target_col]
    
    # Etiketleri Sayıya Çevir (angry -> 0, happy -> 1 vb.)
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # 3. Veriyi Ölçekle
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 4. Train / Test Ayrımı (%20 Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Modeli Eğit
    print("Random Forest eğitiliyor...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # 6. Test Et
    print("Test ediliyor...")
    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("-" * 30)
    print(f"✅ Random Forest Başarısı: %{acc * 100:.2f}")
    print("-" * 30)
    
    class_names = [str(cls) for cls in le.classes_]
    print(classification_report(y_test, y_pred, target_names=class_names))

    # 7. Kaydet
    print("Dosyalar kaydediliyor...")
    joblib.dump(rf_model, os.path.join(MODEL_DIR, MODEL_NAME))
    joblib.dump(scaler, os.path.join(MODEL_DIR, SCALER_NAME))
    joblib.dump(le, os.path.join(MODEL_DIR, LABEL_ENCODER_NAME))
    
    print(f"Model kaydedildi: {os.path.join(MODEL_DIR, MODEL_NAME)}")
    
    # Görselleştir
    plot_confusion_matrix(y_test, y_pred, class_names)

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
    plt.title('Random Forest - Confusion Matrix')
    plt.xlabel('Tahmin')
    plt.ylabel('Gerçek')
    plt.show()

if __name__ == "__main__":
    train_rf()