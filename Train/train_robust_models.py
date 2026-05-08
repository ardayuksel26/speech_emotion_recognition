import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import logging

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

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../TurEV-DB/Extracted CSV")
BASE_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../Models")

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
    if not all_dfs: 
        logger.error("Veri bulunamadı. CSV yollarını kontrol ediniz.")
        return None
    return pd.concat(all_dfs, ignore_index=True)

def build_dnn(input_shape, num_classes):
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_cnn1d(input_shape, num_classes):
    model = Sequential([
        Conv1D(128, kernel_size=5, activation='relu', padding='same', input_shape=(input_shape, 1)),
        MaxPooling1D(pool_size=2),
        BatchNormalization(),
        Dropout(0.3),
        
        Conv1D(64, kernel_size=5, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        BatchNormalization(),
        Dropout(0.3),
        
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_robust_models():
    logger.info("--- Data Augmentation ve Robust (Dış Ses) Tüm Modeller İçin Eğitim Başlıyor ---")
    df = load_data()
    if df is None: return

    target_col = 'label'
    drop_cols = [target_col]
    for col in ['filename', 'name', 'path', 'dosya_adi']:
        if col in df.columns: drop_cols.append(col)
            
    X_original = df.drop(columns=drop_cols, errors='ignore').to_numpy()
    y_original = df[target_col].to_numpy()

    # --- 1. DATA AUGMENTATION (ÖZNİTELİK SEVİYESİNDE GÜRÜLTÜ) ---
    logger.info("Orijinal veriye Gaussian Noise (Dış ortam/Mikrofon simülasyonu) ekleniyor...")
    
    X_noisy_light = X_original + np.random.normal(0, 0.03, X_original.shape)
    X_noisy_heavy = X_original + np.random.normal(0, 0.08, X_original.shape)
    
    X = np.vstack((X_original, X_noisy_light, X_noisy_heavy))
    y = np.concatenate((y_original, y_original, y_original))
    
    logger.info(f"Orijinal Örnek Sayısı: {len(X_original)} -> Yeni Artırılmış Örnek Sayısı: {len(X)}")

    le = LabelEncoder()
    y = le.fit_transform(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, y_train = X, y

    # --- 2. CLASSIC ML MODEL EĞİTİMLERİ ---
    models_to_train = [
        {
            "id": "CatBoost Robust", 
            "dir": "CatBoost_Robust",
            "model_name": "catboost_robust.pkl",
            "model": CatBoostClassifier(iterations=600, learning_rate=0.08, depth=6, random_seed=42, verbose=0)
        },
        {
             "id": "XGBoost Robust", 
             "dir": "XGBoost_Robust",
             "model_name": "xgboost_robust.pkl",
             "model": XGBClassifier(n_estimators=300, learning_rate=0.08, max_depth=6, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        },
        {
             "id": "LightGBM Robust", 
             "dir": "LightGBM_Robust",
             "model_name": "lightgbm_robust.pkl",
             "model": LGBMClassifier(n_estimators=300, learning_rate=0.08, max_depth=6, random_state=42, verbose=-1)
        },
        {
             "id": "RF Robust", 
             "dir": "Random Forest_Robust",
             "model_name": "random_forest_robust.pkl",
             "model": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
        },
        {
             "id": "KNN Robust", 
             "dir": "KNN_Robust",
             "model_name": "knn_robust.pkl",
             "model": KNeighborsClassifier(n_neighbors=5)
        },
        {
             "id": "SVM Robust", 
             "dir": "SVM_Robust",
             "model_name": "svm_robust.pkl",
             "model": SVC(kernel='rbf', probability=True, random_state=42)
        },
        {
             "id": "MLP Robust", 
             "dir": "MLP_Robust",
             "model_name": "mlp_robust.pkl",
             "model": MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, random_state=42)
        },
        {
             "id": "GradientBoosting Robust", 
             "dir": "GradientBoosting_Robust",
             "model_name": "gradboost_robust.pkl",
             "model": GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=3, random_state=42)
        }
    ]

    for config in models_to_train:
        logger.info(f"{config['id']} modeli gürültülü (Robust) veride eğitiliyor...")
        clf = config['model']
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_train)
        if hasattr(y_pred, 'flatten'):
             y_pred = y_pred.flatten()
             
        acc = accuracy_score(y_train, y_pred)
        logger.info(f"✅ {config['id']} Başarısı (Eğitim Verisi): %{acc * 100:.2f}")
        
        save_dir = os.path.join(BASE_MODELS_DIR, config['dir'])
        os.makedirs(save_dir, exist_ok=True)
        
        joblib.dump(clf, os.path.join(save_dir, config['model_name']))
        joblib.dump(scaler, os.path.join(save_dir, "scaler_robust.pkl"))
        joblib.dump(le, os.path.join(save_dir, "label_encoder_robust.pkl"))
        logger.info(f"Model Klasöre Kaydedildi: {save_dir}")

    # --- 3. DEEP LEARNING (DL) EĞİTİMLERİ ---
    if TF_AVAILABLE:
        logger.info("Deep Learning (DNN / CNN1D) modelleri gürültülü (Robust) veride eğitiliyor...")
        y_train_cat = to_categorical(y_train)
        input_shape = X_train.shape[1]
        num_classes = len(le.classes_)
        
        dl_models = [
            {
                "id": "DNN Robust",
                "dir": "DNN_Robust",
                "model_name": "dnn_robust.h5",
                "builder": lambda: build_dnn(input_shape, num_classes),
                "is_cnn": False
            },
            {
                "id": "CNN1D Robust",
                "dir": "CNN1D_Robust",
                "model_name": "cnn1d_robust.h5",
                "builder": lambda: build_cnn1d(input_shape, num_classes),
                "is_cnn": True
            }
        ]

        for config in dl_models:
            logger.info(f"{config['id']} modeli eğitiliyor...")
            model_dl = config['builder']()
            
            X_train_dl = X_train
            if config['is_cnn']:
                X_train_dl = np.expand_dims(X_train, axis=2)
                
            es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
            
            history = model_dl.fit(
                X_train_dl, y_train_cat, 
                epochs=50, batch_size=32, 
                callbacks=[es], verbose=0
            )

            predictions = model_dl.predict(X_train_dl, verbose=0)
            y_pred_dl = np.argmax(predictions, axis=1)
            y_true_dl = np.argmax(y_train_cat, axis=1)
            acc = accuracy_score(y_true_dl, y_pred_dl)
            
            logger.info(f"✅ {config['id']} Başarısı: %{acc * 100:.2f}")

            save_dir = os.path.join(BASE_MODELS_DIR, config['dir'])
            os.makedirs(save_dir, exist_ok=True)
            model_dl.save(os.path.join(save_dir, config['model_name']))
            joblib.dump(scaler, os.path.join(save_dir, "scaler_robust.pkl"))
            joblib.dump(le, os.path.join(save_dir, "label_encoder_robust.pkl"))
            logger.info(f"Model Klasöre Kaydedildi: {save_dir}")
    else:
        logger.warning("TensorFlow kurulu değil. DNN Robust ve CNN1D Robust atlanıyor.")

    logger.info("Tüm Seçili Robust model eğitimleri başarıyla tamamlandı!")

if __name__ == "__main__":
    train_robust_models()
