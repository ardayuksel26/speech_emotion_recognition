import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb  # XGBoost Kütüphanesi

# --- AYARLAR ---
DATA_DIR = "TurEV-DB/Extracted CSV"    # Verilerin olduğu yer
MODEL_DIR = "Models/XGBoost"           # <-- Hedef Klasör
MODEL_NAME = "xgboost_model.pkl"
SCALER_NAME = "scaler_xgb.pkl"
LABEL_ENCODER_NAME = "label_encoder_xgb.pkl"

# Klasör Kontrolü: Yoksa oluştur
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    print(f"Bilgi: '{MODEL_DIR}' klasörü oluşturuldu.")

# Dosya adı -> Etiket eşleşmesi
EMOTION_FILES = {
    "angry.csv": "angry",
    "calm.csv": "calm",
    "happy.csv": "happy",
    "sad.csv": "sad"
}

def load_data():
    """CSV dosyalarını okur ve birleştirir."""
    print(f"Veriler okunuyor: {DATA_DIR}...")
    all_dfs = []
    
    for filename, label in EMOTION_FILES.items():
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['label'] = label # Etiket sütunu ekle
            all_dfs.append(df)
            print(f"  -> {filename} yüklendi. ({len(df)} satır)")
        else:
            print(f"UYARI: {filename} bulunamadı!")
    
    if not all_dfs:
        return None
        
    return pd.concat(all_dfs, ignore_index=True)

def train_xgboost():
    # 1. Veriyi Yükle
    df = load_data()
    if df is None: return

    print("Veri ön işleme yapılıyor...")
    
    # 2. Ön İşleme
    target_col = 'label'
    
    # Gereksiz sütunları temizle
    drop_cols = [target_col]
    for col in ['filename', 'name', 'path', 'dosya_adi']:
        if col in df.columns: drop_cols.append(col)
            
    X = df.drop(columns=drop_cols, errors='ignore')
    y = df[target_col]

    # Etiketleri Sayıya Çevir (XGBoost için 0,1,2,3 olması şarttır)
    le = LabelEncoder()
    y = le.fit_transform(y)

    # 3. Ölçekleme (Standardizasyon)
    # XGBoost ölçeklemeden çok etkilenmese de, veriyi standardize etmek 
    # kararlılığı artırır ve diğer modellerle (SVM) kıyaslamayı adil yapar.
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 4. Train / Test Ayrımı
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Modeli Eğit
    print("XGBoost Modeli eğitiliyor...")
    
    # Parametreler:
    # n_estimators=100: 100 ağaç kur
    # learning_rate=0.1: Öğrenme hızı (Standart başlangıç)
    # eval_metric='mlogloss': Çoklu sınıflandırma (Multi-class) hatasını ölç
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42
    )
    
    xgb_model.fit(X_train, y_train)

    # 6. Test Et
    y_pred = xgb_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("-" * 30)
    print(f"✅ XGBoost Başarısı: %{acc * 100:.2f}")
    print("-" * 30)

    # Detaylı Rapor
    class_names = [str(c) for c in le.classes_]
    print(classification_report(y_test, y_pred, target_names=class_names))

    # 7. Kaydet
    print(f"Dosyalar '{MODEL_DIR}' klasörüne kaydediliyor...")
    
    # Tam dosya yollarını oluştur
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    scaler_path = os.path.join(MODEL_DIR, SCALER_NAME)
    le_path = os.path.join(MODEL_DIR, LABEL_ENCODER_NAME)
    
    joblib.dump(xgb_model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(le, le_path)
    
    print(f"Model kaydedildi: {model_path}")
    
    # Görselleştir
    plot_confusion_matrix(y_test, y_pred, class_names)

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    # XGBoost için turuncu tonları kullanalım (farklılık olsun)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=class_names, yticklabels=class_names)
    plt.title('XGBoost - Confusion Matrix')
    plt.xlabel('Tahmin')
    plt.ylabel('Gerçek')
    plt.show()

if __name__ == "__main__":
    train_xgboost()