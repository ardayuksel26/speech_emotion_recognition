import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier # Sklearn'ün kendi GB modeli
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- AYARLAR ---
DATA_DIR = "../Test/sentencevoice_train/Extracted CSV"
MODEL_DIR = "../Models/Sentence_Models/GradientBoosting" # Yeni klasör
MODEL_NAME = "gradboost_model.pkl"
SCALER_NAME = "scaler_gb.pkl"
LABEL_ENCODER_NAME = "label_encoder_gb.pkl"
os.makedirs(MODEL_DIR, exist_ok=True)




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
    all_dfs = []
    for fname, label in EMOTION_FILES.items():
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['label'] = label
            all_dfs.append(df)
    if not all_dfs: return None
    return pd.concat(all_dfs, ignore_index=True)

def train_gb():
    print("--- Standart Gradient Boosting Eğitimi Başlıyor ---")

    df = load_data()
    if df is None: return
    
    print("Konuşmacı bazlı normalizasyon uygulanıyor...")
    df = apply_speaker_normalization(df)


    # Ön İşleme
    target_col = 'label'
    drop_cols = [target_col, 'speaker_id']
    for col in ['filename', 'name', 'path', 'dosya_adi']:
        if col in df.columns: drop_cols.append(col)
            
    X = df.drop(columns=drop_cols, errors='ignore')
    y = df[target_col]

    le = LabelEncoder()
    y = le.fit_transform(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, y_train = X, y

    # --- MODEL ---
    print("Gradient Boosting eğitiliyor (Bu işlem XGBoost'tan yavaş olabilir)...")
    
    # Standart Gradient Boosting Parametreleri
    gb_model = GradientBoostingClassifier(
        n_estimators=100,       # Ağaç sayısı
        learning_rate=0.1,      # Öğrenme hızı
        max_depth=3,            # Ağaç derinliği (Standart genelde 3'tür)
        random_state=42
    )
    gb_model.fit(X_train, y_train)

    y_pred = gb_model.predict(X_train)
    acc = accuracy_score(y_train, y_pred)
    
    print("-" * 30)
    print(f"✅ Gradient Boosting Başarısı: %{acc * 100:.2f}")
    print("-" * 30)

    # Kaydet
    joblib.dump(gb_model, os.path.join(MODEL_DIR, MODEL_NAME))
    joblib.dump(scaler, os.path.join(MODEL_DIR, SCALER_NAME))
    joblib.dump(le, os.path.join(MODEL_DIR, LABEL_ENCODER_NAME))
    
    class_names = [str(c) for c in le.classes_]
    print(classification_report(y_train, y_pred, target_names=class_names))

    # Görselleştir
    cm = confusion_matrix(y_train, y_pred)
    plt.figure(figsize=(8, 6))
    # Gri tonları kullanalım (Klasik yöntem olduğu için)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greys', xticklabels=class_names, yticklabels=class_names)
    plt.title('Standard Gradient Boosting Confusion Matrix')
    plt.close() # plt.show()

if __name__ == "__main__":
    train_gb()