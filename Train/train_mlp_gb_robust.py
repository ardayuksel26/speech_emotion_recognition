import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../TurEV-DB/Extracted CSV")
MODELS_DIR = os.path.join(BASE_DIR, "../Models")

EMOTION_FILES = {"angry.csv": "angry", "calm.csv": "calm", "happy.csv": "happy", "sad.csv": "sad"}


def load_data():
    all_dfs = []
    for fname, label in EMOTION_FILES.items():
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["label"] = label
            all_dfs.append(df)
    if not all_dfs:
        raise FileNotFoundError(f"CSV bulunamadı: {DATA_DIR}")
    return pd.concat(all_dfs, ignore_index=True)


def prepare(df):
    drop = ["label"] + [c for c in ["filename", "name", "path", "dosya_adi"] if c in df.columns]
    X = df.drop(columns=drop, errors="ignore").to_numpy()
    y = df["label"].to_numpy()

    le = LabelEncoder()
    y = le.fit_transform(y)

    # Gaussian noise augmentation (simulates noisy/outdoor recording)
    X_light = X + np.random.normal(0, 0.03, X.shape)
    X_heavy = X + np.random.normal(0, 0.08, X.shape)
    X_aug = np.vstack([X, X_light, X_heavy])
    y_aug = np.concatenate([y, y, y])

    scaler = StandardScaler()
    X_aug = scaler.fit_transform(X_aug)

    logger.info(f"Orijinal: {len(X)} örnek → Artırılmış: {len(X_aug)} örnek")
    return X_aug, y_aug, scaler, le


def save(model, scaler, le, model_dir, model_filename):
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model,  os.path.join(model_dir, model_filename))
    joblib.dump(scaler, os.path.join(model_dir, "scaler_robust.pkl"))
    joblib.dump(le,     os.path.join(model_dir, "label_encoder_robust.pkl"))
    logger.info(f"Kaydedildi: {model_dir}")


def train_mlp_robust(X, y, scaler, le):
    logger.info("MLP Robust eğitiliyor...")
    model = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, random_state=42)
    model.fit(X, y)
    acc = accuracy_score(y, model.predict(X))
    logger.info(f"MLP Robust train accuracy: %{acc*100:.2f}")
    save(model, scaler, le, os.path.join(MODELS_DIR, "MLP_Robust"), "mlp_robust.pkl")


def train_gb_robust(X, y, scaler, le):
    logger.info("Gradient Boosting Robust eğitiliyor (yavaş olabilir)...")
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=3, random_state=42)
    model.fit(X, y)
    acc = accuracy_score(y, model.predict(X))
    logger.info(f"GradientBoosting Robust train accuracy: %{acc*100:.2f}")
    save(model, scaler, le, os.path.join(MODELS_DIR, "GradientBoosting_Robust"), "gradboost_robust.pkl")


if __name__ == "__main__":
    df = load_data()
    X, y, scaler, le = prepare(df)
    train_mlp_robust(X, y, scaler, le)
    train_gb_robust(X, y, scaler, le)
    logger.info("Tamamlandı. Backend'i yeniden başlatmayı unutma.")
