import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense, Dropout,
                                      Flatten, LSTM, BatchNormalization,
                                      Permute, TimeDistributed)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# --- AYARLAR ---
# Önce Data_with_noise/prepare_dataset.py çalıştırılmalıdır.
DATA_ROOT  = "Data_with_noise"
MODEL_DIR  = "Models2"
MODEL_NAME = "cnn_lstm_model.h5"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    print("Spectrogram verileri yükleniyor...")
    X = np.load(os.path.join(DATA_ROOT, "features_spectrograms.npy"))
    y = np.load(os.path.join(DATA_ROOT, "labels_onehot.npy"))
    return X, y


def build_cnn_lstm_model(input_shape, num_classes):
    model = Sequential()

    # CNN Blokları
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))

    # CNN → LSTM geçiş
    model.add(Permute((2, 1, 3)))
    model.add(TimeDistributed(Flatten()))

    # LSTM
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.4))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train():
    X, y = load_data()

    X_train, y_train = X, y
    print(f"Eğitim Verisi: {X_train.shape}")

    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    num_classes = y.shape[1]

    model = build_cnn_lstm_model(input_shape, num_classes)
    model.summary()

    callbacks = [
        ReduceLROnPlateau(monitor='loss', patience=5, verbose=1, factor=0.1, min_lr=1e-5),
        EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    ]

    print("Eğitim başlıyor...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        callbacks=callbacks
    )

    model.save(MODEL_PATH)
    print(f"✅ Model kaydedildi: {MODEL_PATH}")

    plot_history(history)


def plot_history(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Eğitim Başarısı')
    plt.title('Model Başarısı')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Eğitim Kaybı')
    plt.title('Model Kaybı')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.close()


if __name__ == "__main__":
    train()
