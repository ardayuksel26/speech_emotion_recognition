import pandas as pd
import numpy as np
import os
import joblib
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.utils import to_categorical
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Robust_Trainer")

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(BASE_DIR, ".."))
DATA_DIR     = os.path.join(PROJECT_ROOT, "All_Sounds", "Extracted_CSV")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "Models3")

EMOTION_FILES = {"angry.csv": "angry", "calm.csv": "calm", "happy.csv": "happy", "sad.csv": "sad"}


def load_data():
    all_dfs = []
    for fname, label in EMOTION_FILES.items():
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['label'] = label
            all_dfs.append(df)
    if not all_dfs:
        logger.error("Veri bulunamadı.")
        return None
    return pd.concat(all_dfs, ignore_index=True)


def preprocess(df):
    drop_cols = ['label'] + [c for c in ['filename', 'name', 'path', 'dosya_adi'] if c in df.columns]
    X = df.drop(columns=drop_cols, errors='ignore')
    y = df['label']
    le = LabelEncoder()
    y = le.fit_transform(y)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, le, scaler


def save_model(model, scaler, le, subdir, model_fname, scaler_fname, le_fname):
    d = os.path.join(MODELS_DIR, subdir)
    os.makedirs(d, exist_ok=True)
    joblib.dump(model,  os.path.join(d, model_fname))
    joblib.dump(scaler, os.path.join(d, scaler_fname))
    joblib.dump(le,     os.path.join(d, le_fname))
    logger.info(f"Kaydedildi: {d}")


def train_all():
    df = load_data()
    if df is None:
        return

    X, y, le, scaler = preprocess(df)

    models = [
        ("SVM",              SVC(kernel='rbf', probability=True, random_state=42),
         "SVM", "svm_model.pkl", "scaler_svm.pkl", "label_encoder_svm.pkl"),
        ("Random Forest",    RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
         "RandomForest", "random_forest_model.pkl", "scaler_rf.pkl", "label_encoder_rf.pkl"),
        ("XGBoost",          XGBClassifier(n_estimators=300, learning_rate=0.05, random_state=42,
                                           eval_metric='mlogloss', use_label_encoder=False),
         "XGBoost", "xgboost_model.pkl", "scaler_xgb.pkl", "label_encoder_xgb.pkl"),
        ("LightGBM",         LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1),
         "LightGBM", "lightgbm_model.pkl", "scaler_lgbm.pkl", "label_encoder_lgbm.pkl"),
        ("CatBoost",         CatBoostClassifier(iterations=300, learning_rate=0.05, random_seed=42, verbose=0),
         "CatBoost", "catboost_model.pkl", "scaler_catboost.pkl", "label_encoder_catboost.pkl"),
        ("KNN",              KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
         "KNN", "knn_model.pkl", "scaler_knn.pkl", "label_encoder_knn.pkl"),
        ("MLP",              MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=500,
                                           random_state=42, early_stopping=True),
         "MLP", "mlp_model.pkl", "scaler_mlp.pkl", "label_encoder_mlp.pkl"),
        ("Gradient Boosting", GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                                          max_depth=5, random_state=42),
         "GradientBoosting", "gradient_boosting_model.pkl", "scaler_gb.pkl", "label_encoder_gb.pkl"),
    ]

    results = {}
    for name, model, subdir, mf, sf, lef in models:
        logger.info(f"\n{'='*40}\n  {name} eğitiliyor...\n{'='*40}")
        try:
            model.fit(X, y)
            acc = accuracy_score(y, model.predict(X))
            results[name] = acc
            logger.info(f"{name} Train Accuracy: %{acc*100:.2f}")
            save_model(model, scaler, le, subdir, mf, sf, lef)
        except Exception as e:
            logger.error(f"{name} HATA: {e}")

    if TF_AVAILABLE:
        logger.info(f"\n{'='*40}\n  CNN1D eğitiliyor...\n{'='*40}")
        try:
            from tensorflow.keras.utils import to_categorical
            y_cat = to_categorical(y)
            X_cnn = np.expand_dims(X, axis=2)
            cnn = Sequential([
                Conv1D(64,  3, activation='relu', input_shape=(X.shape[1], 1)),
                BatchNormalization(), MaxPooling1D(2),
                Conv1D(128, 3, activation='relu'),
                BatchNormalization(), MaxPooling1D(2),
                Flatten(), Dense(128, activation='relu'), Dropout(0.5),
                Dense(4, activation='softmax'),
            ])
            cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            cnn.fit(X_cnn, y_cat, epochs=100, batch_size=32,
                    callbacks=[EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)],
                    verbose=0)
            d = os.path.join(MODELS_DIR, "CNN1D")
            os.makedirs(d, exist_ok=True)
            cnn.save(os.path.join(d, "cnn1d_model.h5"))
            joblib.dump(scaler, os.path.join(d, "scaler_cnn1d.pkl"))
            joblib.dump(le,     os.path.join(d, "label_encoder_cnn1d.pkl"))
            logger.info(f"CNN1D kaydedildi: {d}")
        except Exception as e:
            logger.error(f"CNN1D HATA: {e}")

    print("\n" + "="*40)
    print("  SONUÇLAR")
    print("="*40)
    for name, acc in results.items():
        print(f"  {name:<22}: %{acc*100:.2f}")


if __name__ == "__main__":
    train_all()
