import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(BASE_DIR, ".."))
DATA_DIR     = os.path.join(PROJECT_ROOT, "All_Sounds", "Extracted_CSV")
MODEL_DIR    = os.path.join(PROJECT_ROOT, "Models3", "DNN")
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


def train_dnn():
    print("--- DNN Eğitimi Başlıyor ---")
    df = load_data()

    drop_cols = ['label'] + [c for c in ['filename', 'name', 'path', 'dosya_adi'] if c in df.columns]
    X = df.drop(columns=drop_cols, errors='ignore')
    y = df['label']

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    y_cat = to_categorical(y_enc)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model = Sequential([
        Dense(256, input_dim=X.shape[1], activation='relu'),
        BatchNormalization(), Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(), Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dense(4, activation='softmax'),
    ])
    model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        X, y_cat, epochs=100, batch_size=32,
        callbacks=[EarlyStopping(monitor='loss', patience=10, restore_best_weights=True),
                   ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1e-5)],
        verbose=1
    )

    model.save(os.path.join(MODEL_DIR, "dnn_model.h5"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_dnn.pkl"))
    joblib.dump(le,     os.path.join(MODEL_DIR, "label_encoder_dnn.pkl"))
    print(f"Model kaydedildi: {MODEL_DIR}")

    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.title('DNN Model Başarısı')
    plt.legend()
    plt.savefig(os.path.join(MODEL_DIR, "training_history.png"))
    plt.close()


if __name__ == "__main__":
    train_dnn()
