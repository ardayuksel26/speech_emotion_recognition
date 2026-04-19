import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# --- AYARLAR ---
DATA_DIR = "../Test/sentencevoice_train/Extracted CSV"
MODEL_DIR = "../Models/Sentence_Models/CNN1D"
MODEL_NAME = "cnn1d_model.h5"
SCALER_NAME = "scaler_cnn1d.pkl"
LABEL_ENCODER_NAME = "label_encoder_cnn1d.pkl"
os.makedirs(MODEL_DIR, exist_ok=True)




EMOTION_FILES = {"angry.csv": "angry", "calm.csv": "calm", "happy.csv": "happy", "sad.csv": "sad"}


def apply_speaker_normalization(df):
    """Her konuşmacıyı kendi içinde normalize ederek bireysel farkları ortadan kaldırır."""
    target_col = 'label'
    speaker_col = 'speaker_id'
    
    # Sayısal sütunları bul (label ve speaker_id hariç)
    skip_cols = [target_col, speaker_col, 'filename', 'name', 'path', 'dosya_adi']
    feat_cols = [c for c in df.columns if c not in skip_cols and df[c].dtype in ['float64', 'int64']]
    
    # Grupla ve normalize et
    df[feat_cols] = df.groupby(speaker_col)[feat_cols].transform(lambda x: (x - x.mean()) / (x.std() + 1e-6))
    return df

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
    
    target_col = 'label'
    drop_cols = [target_col, 'speaker_id']
    for col in ['filename', 'name', 'path', 'dosya_adi']:
        if col in df.columns: drop_cols.append(col)
    
    X = df.drop(columns=drop_cols, errors='ignore')
    y = df[target_col]

    le = LabelEncoder()
    y = le.fit_transform(y)
    
    from tensorflow.keras.utils import to_categorical
    y_cat = to_categorical(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 1D CNN için veriyi şekillendir: (Örnek Sayısı, Özellik Sayısı, 1)
    X = np.expand_dims(X, axis=2)

    X_train, y_train = X, y_cat

    # --- MODEL MİMARİSİ ---
    model = Sequential()
    
    # CNN Katmanı 1
    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    
    # CNN Katmanı 2
    model.add(Conv1D(128, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Flatten()) # Düzleştir
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(4, activation='softmax'))

    optimizer = Adam(learning_rate=0.0005) # CNN için biraz daha yavaş öğrenme hızı
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    # Kaydet
    model.save(os.path.join(MODEL_DIR, MODEL_NAME))
    # Scaler kaydederken boyutu eski haline getirmek gerekmez, joblib halleder
    # Ancak scaler fit işlemi 2D data üzerinde yapıldı, bu doğru.
    joblib.dump(scaler, os.path.join(MODEL_DIR, SCALER_NAME))
    joblib.dump(le, os.path.join(MODEL_DIR, LABEL_ENCODER_NAME))
    
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.title('1D CNN Model Başarısı')
    plt.legend()
    plt.close() # plt.show()

if __name__ == "__main__":
    train_cnn1d()