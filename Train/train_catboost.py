import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier  # CatBoost Kütüphanesi

# --- AYARLAR ---
DATA_DIR = "TurEV-DB/Extracted CSV"
MODEL_DIR = "Models2/CatBoost"
MODEL_NAME = "catboost_model.pkl"
SCALER_NAME = "scaler_cb.pkl"
LABEL_ENCODER_NAME = "label_encoder_cb.pkl"

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

EMOTION_FILES = {
    "angry.csv": "angry",
    "calm.csv": "calm",
    "happy.csv": "happy",
    "sad.csv": "sad"
}

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

def train_catboost():
    print("--- CatBoost Eğitimi Başlıyor ---")
    df = load_data()
    if df is None: return

    # Ön İşleme
    target_col = 'label'
    drop_cols = [target_col]
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
    print("CatBoost eğitiliyor...")
    # verbose=0: Eğitim sırasında ekrana sürekli yazı basmasını engeller
    cb_model = CatBoostClassifier(
        iterations=500, 
        learning_rate=0.1, 
        depth=6, 
        random_seed=42,
        verbose=0  
    )
    cb_model.fit(X_train, y_train)

    y_pred = cb_model.predict(X_train)
    # CatBoost bazen sonucu [[1], [0]] gibi döndürür, düzeltelim:
    y_pred = y_pred.flatten()
    
    acc = accuracy_score(y_train, y_pred)
    
    print("-" * 30)
    print(f"✅ CatBoost Başarısı: %{acc * 100:.2f}")
    print("-" * 30)

    # Kaydet
    joblib.dump(cb_model, os.path.join(MODEL_DIR, MODEL_NAME))
    joblib.dump(scaler, os.path.join(MODEL_DIR, SCALER_NAME))
    joblib.dump(le, os.path.join(MODEL_DIR, LABEL_ENCODER_NAME))
    
    class_names = [str(c) for c in le.classes_]
    print(classification_report(y_train, y_pred, target_names=class_names))

    # Görselleştir
    cm = confusion_matrix(y_train, y_pred)
    plt.figure(figsize=(8, 6))
    # Mavi-Yeşil tonları (Yandex renkleri gibi olsun)
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=class_names, yticklabels=class_names)
    plt.title('CatBoost Confusion Matrix')
    plt.close() # plt.show()

if __name__ == "__main__":
    train_catboost()