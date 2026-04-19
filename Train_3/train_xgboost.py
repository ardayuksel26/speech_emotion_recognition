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
DATA_DIR = "../Test/sentencevoice_train/Extracted CSV"    # Verilerin olduğu yer
MODEL_DIR = "../Models/Sentence_Models/XGBoost"           # <-- Hedef Klasör
MODEL_NAME = "xgboost_model.pkl"
SCALER_NAME = "scaler_xgb.pkl"
LABEL_ENCODER_NAME = "label_encoder_xgb.pkl"
os.makedirs(MODEL_DIR, exist_ok=True)

# Klasör Kontrolü: Yoksa oluştur



# Dosya adı -> Etiket eşleşmesi
EMOTION_FILES = {
    "angry.csv": "angry",
    "calm.csv": "calm",
    "happy.csv": "happy",
    "sad.csv": "sad"
}


def apply_speaker_normalization(df):
    """Her konuşmacıyı kendi içinde normalize ederek bireysel farkları ortadan kaldırır."""
    target_col = 'label'
    speaker_col = 'speaker_id'
    
    # Sayısal sütunları bul (label ve speaker_id hariç)
    skip_cols = [target_col, speaker_col, 'filename', 'name', 'path', 'dosya_adi']
    feat_cols = [c for c in df.columns if c not in skip_cols and df[c].dtype in ['float64', 'int64']]
    
    # Grupla ve normalize et
    df[feat_cols] = df.groupby(speaker_col)[feat_cols].transform(lambda x: (x - x.mean()) / (x.std() + 1e-6))
    return df

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
    
    print("Konuşmacı bazlı normalizasyon uygulanıyor...")
    df = apply_speaker_normalization(df)


    print("Veri ön işleme yapılıyor...")
    
    # 2. Ön İşleme
    target_col = 'label'
    
    # Gereksiz sütunları temizle
    drop_cols = [target_col, 'speaker_id']
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

    # 4. Use 100% data
    X_train, y_train = X, y

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

    # 6. Test Et (On Training Data)
    y_pred = xgb_model.predict(X_train)
    acc = accuracy_score(y_train, y_pred)
    
    print("-" * 30)
    print(f"✅ XGBoost Başarısı: %{acc * 100:.2f}")
    print("-" * 30)

    # Detaylı Rapor
    class_names = [str(c) for c in le.classes_]
    print(classification_report(y_train, y_pred, target_names=class_names))

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
    plot_confusion_matrix(y_train, y_pred, class_names)

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    # XGBoost için turuncu tonları kullanalım (farklılık olsun)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=class_names, yticklabels=class_names)
    plt.title('XGBoost - Confusion Matrix')
    plt.xlabel('Tahmin')
    plt.ylabel('Gerçek')
    plt.close() # plt.show()

if __name__ == "__main__":
    train_xgboost()