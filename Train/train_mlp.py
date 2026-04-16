import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier # Yapay Sinir Ağı
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- AYARLAR ---
DATA_DIR = "TurEV-DB/Extracted CSV"
MODEL_DIR = "Models2/MLP"
MODEL_NAME = "mlp_model.pkl"
SCALER_NAME = "scaler_mlp.pkl"
LABEL_ENCODER_NAME = "label_encoder_mlp.pkl"

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

def train_mlp():
    print("--- MLP (Sinir Ağı) Eğitimi Başlıyor ---")
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

    # Sinir ağları için SCALER ÇOK KRİTİKTİR
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, y_train = X, y

    # --- MODEL ---
    print("MLP eğitiliyor (Biraz sürebilir)...")
    # hidden_layer_sizes=(256, 128, 64): 3 katmanlı, gittikçe küçülen yapı
    # max_iter=500: Maksimum döngü sayısı
    # early_stopping=False: 100% veri ile eğitim yaparken False olması daha uygundur
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64), 
        activation='relu', 
        solver='adam', 
        alpha=0.0001, 
        batch_size='auto',
        learning_rate='adaptive',
        max_iter=500,
        early_stopping=False,
        random_state=42
    )
    mlp_model.fit(X_train, y_train)

    y_pred = mlp_model.predict(X_train)
    acc = accuracy_score(y_train, y_pred)
    
    print("-" * 30)
    print(f"✅ MLP Başarısı: %{acc * 100:.2f}")
    print("-" * 30)

    # Kaydet
    joblib.dump(mlp_model, os.path.join(MODEL_DIR, MODEL_NAME))
    joblib.dump(scaler, os.path.join(MODEL_DIR, SCALER_NAME))
    joblib.dump(le, os.path.join(MODEL_DIR, LABEL_ENCODER_NAME))
    
    class_names = [str(c) for c in le.classes_]
    print(classification_report(y_train, y_pred, target_names=class_names))

    # Görselleştir
    cm = confusion_matrix(y_train, y_pred)
    plt.figure(figsize=(8, 6))
    # Kırmızı tonları
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=class_names, yticklabels=class_names)
    plt.title('MLP Confusion Matrix')
    plt.close() # plt.show()

if __name__ == "__main__":
    train_mlp()