import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(BASE_DIR, ".."))
DATA_DIR     = os.path.join(PROJECT_ROOT, "All_Sounds", "Extracted_CSV")
MODEL_DIR    = os.path.join(PROJECT_ROOT, "Models3", "KNN")
os.makedirs(MODEL_DIR, exist_ok=True)

EMOTION_FILES = {"angry.csv": "angry", "calm.csv": "calm", "happy.csv": "happy", "sad.csv": "sad"}


def load_data():
    all_dfs = []
    for fname, label in EMOTION_FILES.items():
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['label'] = label
            all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)


def train_knn():
    print("--- KNN Eğitimi Başlıyor ---")
    df = load_data()

    drop_cols = ['label'] + [c for c in ['filename', 'name', 'path', 'dosya_adi'] if c in df.columns]
    X = df.drop(columns=drop_cols, errors='ignore')
    y = df['label']

    le = LabelEncoder()
    y = le.fit_transform(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model = KNeighborsClassifier(n_neighbors=5, metric='euclidean', n_jobs=-1)
    model.fit(X, y)

    y_pred = model.predict(X)
    print(f"KNN Train Accuracy: %{accuracy_score(y, y_pred)*100:.2f}")
    print(classification_report(y, y_pred, target_names=[str(c) for c in le.classes_]))

    joblib.dump(model,  os.path.join(MODEL_DIR, "knn_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_knn.pkl"))
    joblib.dump(le,     os.path.join(MODEL_DIR, "label_encoder_knn.pkl"))
    print(f"Model kaydedildi: {MODEL_DIR}")

    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('KNN - Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"))
    plt.close()


if __name__ == "__main__":
    train_knn()
