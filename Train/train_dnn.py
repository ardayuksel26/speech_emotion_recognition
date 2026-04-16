import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- AYARLAR ---
DATA_DIR = "TurEV-DB/Extracted CSV"
MODEL_DIR = "Models2/DNN"
MODEL_NAME = "dnn_model.h5"  # Keras modeli olduğu için .h5
SCALER_NAME = "scaler_dnn.pkl"
LABEL_ENCODER_NAME = "label_encoder_dnn.pkl"

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

EMOTION_FILES = {"angry.csv": "angry", "calm.csv": "calm", "happy.csv": "happy", "sad.csv": "sad"}

def load_data():
    all_dfs = []
    for fname, label in EMOTION_FILES.items():
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['label'] = label
            all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else None

def train_dnn():
    print("--- DNN Eğitimi Başlıyor ---")
    df = load_data()
    
    # Ön İşleme
    target_col = 'label'
    drop_cols = [target_col]
    for col in ['filename', 'name', 'path', 'dosya_adi']:
        if col in df.columns: drop_cols.append(col)
    
    X = df.drop(columns=drop_cols, errors='ignore')
    y = df[target_col]

    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # One-Hot Encoding (Keras için gerekli)
    from tensorflow.keras.utils import to_categorical
    y_cat = to_categorical(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, y_train = X, y_cat

    # --- MODEL MİMARİSİ ---
    model = Sequential()
    
    # Giriş Katmanı
    model.add(Dense(256, input_dim=X.shape[1], activation='relu'))
    model.add(BatchNormalization()) # Veriyi dengeler
    model.add(Dropout(0.3))         # %30 unutma (Ezber önler)

    # Gizli Katman 1
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Gizli Katman 2
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    
    # Çıkış Katmanı (4 Duygu)
    model.add(Dense(4, activation='softmax'))

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Callback'ler
    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.00001)

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # Kaydet
    model.save(os.path.join(MODEL_DIR, MODEL_NAME))
    joblib.dump(scaler, os.path.join(MODEL_DIR, SCALER_NAME))
    joblib.dump(le, os.path.join(MODEL_DIR, LABEL_ENCODER_NAME))
    
    # Grafik
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.title('DNN Model Başarısı')
    plt.legend()
    plt.close() # plt.show()

if __name__ == "__main__":
    train_dnn()