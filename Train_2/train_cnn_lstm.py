import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense, Dropout,
                                     Flatten, LSTM, BatchNormalization,
                                     Permute, TimeDistributed)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(BASE_DIR, ".."))
DATA_ROOT    = os.path.join(PROJECT_ROOT, "All_Sounds")
MODEL_DIR    = os.path.join(PROJECT_ROOT, "Models3")
MODEL_PATH   = os.path.join(MODEL_DIR, "cnn_lstm_model.h5")
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    print("Spectrogram verisi yükleniyor...")
    X = np.load(os.path.join(DATA_ROOT, "features_spectrograms.npy"))
    y = np.load(os.path.join(DATA_ROOT, "labels_onehot.npy"))
    return X, y


def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(64,  (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(), MaxPooling2D((2, 2)), Dropout(0.4),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(), MaxPooling2D((2, 2)), Dropout(0.4),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(), MaxPooling2D((2, 2)), Dropout(0.4),
        Permute((2, 1, 3)),
        TimeDistributed(Flatten()),
        LSTM(64, return_sequences=False), Dropout(0.4),
        Dense(num_classes, activation='softmax'),
    ])
    model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train():
    print("--- CNN-LSTM Eğitimi Başlıyor ---")
    X, y = load_data()
    print(f"Eğitim Verisi: {X.shape}")

    model = build_model((X.shape[1], X.shape[2], 1), y.shape[1])
    model.summary()

    history = model.fit(
        X, y, epochs=50, batch_size=32,
        callbacks=[
            ReduceLROnPlateau(monitor='loss', patience=5, factor=0.1, min_lr=1e-5, verbose=1),
            EarlyStopping(monitor='loss', patience=10, restore_best_weights=True),
        ]
    )

    model.save(MODEL_PATH)
    print(f"Model kaydedildi: {MODEL_PATH}")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Eğitim Başarısı')
    plt.title('Model Başarısı')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Eğitim Kaybı')
    plt.title('Model Kaybı')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "cnn_lstm_history.png"))
    plt.close()


if __name__ == "__main__":
    train()
