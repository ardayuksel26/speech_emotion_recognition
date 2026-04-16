import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# --- AYARLAR ---
DATA_DIR   = "Data_with_noise/Extracted CSV"
MODEL_DIR  = "Models2/CNN1D"
MODEL_NAME = "cnn1d_model.h5"
SCALER_NAME = "scaler_cnn1d.pkl"
LABEL_ENCODER_NAME = "label_encoder_cnn1d.pkl"

os.makedirs(MODEL_DIR, exist_ok=True)

EMOTION_FILES = {
    "angry.csv": "angry",
    "calm.csv":  "calm",
    "happy.csv": "happy",
    "sad.csv":   "sad"
}


def load_data():
    all_dfs = []
    for fname, label in EMOTION_FILES.items():
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['label'] = label
            all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else None


def train_cnn1d():
    print("--- 1D CNN Eğitimi Başlıyor ---")
    df = load_data()
    if df is None:
        return

    target_col = 'label'
    drop_cols  = [target_col] + [c for c in ['filename', 'name', 'path', 'dosya_adi'] if c in df.columns]

    X = df.drop(columns=drop_cols, errors='ignore')
    y = df[target_col]

    le = LabelEncoder()
    y  = le.fit_transform(y)
    y_cat = to_categorical(y)

    scaler = StandardScaler()
    X      = scaler.fit_transform(X)
    X      = np.expand_dims(X, axis=2)   # (N, features, 1)

    X_train, y_train = X, y_cat

    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    model.save(os.path.join(MODEL_DIR, MODEL_NAME))
    joblib.dump(scaler, os.path.join(MODEL_DIR, SCALER_NAME))
    joblib.dump(le,     os.path.join(MODEL_DIR, LABEL_ENCODER_NAME))
    print(f"Model kaydedildi: {MODEL_DIR}")

    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.title('1D CNN Model Başarısı')
    plt.legend()
    plt.close()


if __name__ == "__main__":
    train_cnn1d()
