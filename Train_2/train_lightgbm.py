import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb

# --- AYARLAR ---
DATA_DIR   = "Data_with_noise/Extracted CSV"
MODEL_DIR  = "Models2/LightGBM"
MODEL_NAME = "lightgbm_model.pkl"
SCALER_NAME = "scaler_lgb.pkl"
LABEL_ENCODER_NAME = "label_encoder_lgb.pkl"

os.makedirs(MODEL_DIR, exist_ok=True)

EMOTION_FILES = {
    "angry.csv": "angry",
    "calm.csv":  "calm",
    "happy.csv": "happy",
    "sad.csv":   "sad"
}


def load_data():
    print(f"Veriler okunuyor: {DATA_DIR}...")
    all_dfs = []
    for filename, label in EMOTION_FILES.items():
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['label'] = label
            all_dfs.append(df)
            print(f"  -> {filename} yüklendi. ({len(df)} satır)")
        else:
            print(f"UYARI: {filename} bulunamadı!")
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else None


def train_lightgbm():
    print("--- LightGBM Eğitimi Başlıyor ---")
    df = load_data()
    if df is None:
        return

    target_col = 'label'
    drop_cols  = [target_col] + [c for c in ['filename', 'name', 'path', 'dosya_adi'] if c in df.columns]

    X = df.drop(columns=drop_cols, errors='ignore')
    y = df[target_col]

    le = LabelEncoder()
    y  = le.fit_transform(y)

    scaler = StandardScaler()
    X      = scaler.fit_transform(X)

    X_train, y_train = X, y

    print("LightGBM eğitiliyor...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        verbosity=-1
    )
    lgb_model.fit(X_train, y_train)

    y_pred = lgb_model.predict(X_train)
    acc    = accuracy_score(y_train, y_pred)

    print("-" * 30)
    print(f"✅ LightGBM Başarısı: %{acc * 100:.2f}")
    print("-" * 30)

    class_names = [str(c) for c in le.classes_]
    print(classification_report(y_train, y_pred, target_names=class_names))

    joblib.dump(lgb_model, os.path.join(MODEL_DIR, MODEL_NAME))
    joblib.dump(scaler,    os.path.join(MODEL_DIR, SCALER_NAME))
    joblib.dump(le,        os.path.join(MODEL_DIR, LABEL_ENCODER_NAME))
    print(f"Model kaydedildi: {MODEL_DIR}")

    cm = confusion_matrix(y_train, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('LightGBM Confusion Matrix')
    plt.close()


if __name__ == "__main__":
    train_lightgbm()
