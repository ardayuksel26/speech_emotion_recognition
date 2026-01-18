import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier # k-NN kütüphanesi
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- AYARLAR ---
DATA_DIR = "TurEV-DB/Extracted CSV"
MODEL_DIR = "Models/KNN"
MODEL_NAME = "knn_model.pkl"
SCALER_NAME = "scaler_knn.pkl"
LABEL_ENCODER_NAME = "label_encoder_knn.pkl"

# Klasör Kontrolü
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

def train_knn():
    print("--- k-NN Eğitimi Başlıyor ---")
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

    # k-NN mesafe ölçtüğü için SCALER ŞARTTIR
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- MODEL ---
    print("k-NN eğitiliyor...")
    # n_neighbors=5: En yakın 5 komşuya bakarak karar ver
    # metric='minkowski': Mesafe ölçüm yöntemi (Euclidean gibi)
    knn_model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    knn_model.fit(X_train, y_train)

    y_pred = knn_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("-" * 30)
    print(f"✅ k-NN Başarısı: %{acc * 100:.2f}")
    print("-" * 30)

    # Kaydet
    joblib.dump(knn_model, os.path.join(MODEL_DIR, MODEL_NAME))
    joblib.dump(scaler, os.path.join(MODEL_DIR, SCALER_NAME))
    joblib.dump(le, os.path.join(MODEL_DIR, LABEL_ENCODER_NAME))
    
    class_names = [str(c) for c in le.classes_]
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Görselleştir
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    # Mor renk kullanalım
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=class_names, yticklabels=class_names)
    plt.title('k-NN Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    train_knn()