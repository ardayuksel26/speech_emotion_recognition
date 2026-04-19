import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, LSTM, BatchNormalization, Permute, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# --- AYARLAR ---
DATA_ROOT = "Data"
MODEL_DIR = "../Models/Sentence_Models"
MODEL_NAME = "cnn_lstm_model.h5"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME) # Tam yol: Models2/cnn_lstm_model.h5

# Models klasörü yoksa oluştur (Hata almamak için)




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
    print("Veriler yükleniyor...")
    # Hazırladığımız npy dosyalarını yüklüyoruz
    X = np.load(os.path.join(DATA_ROOT, "features_spectrograms.npy"))
    y = np.load(os.path.join(DATA_ROOT, "labels_onehot.npy"))
    return X, y

def build_cnn_lstm_model(input_shape, num_classes):
    """
    Raporundaki Hybrid CNN-LSTM Mimarisi
    """
    model = Sequential()

    # --- CNN BLOKLARI (Görsel Özellik Çıkarımı) ---
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    # --- GEÇİŞ KATMANI (CNN -> LSTM) ---
    # Boyutları (Frekans, Zaman, Kanal) -> (Zaman, Frekans, Kanal) yapıyoruz
    model.add(Permute((2, 1, 3)))
    
    # Zaman adımını koruyarak özellikleri düzleştiriyoruz
    model.add(TimeDistributed(Flatten()))

    # --- LSTM KATMANI (Zamansal Özellik Çıkarımı) ---
    model.add(LSTM(64, return_sequences=False)) 
    model.add(Dropout(0.4))

    # --- ÇIKTI KATMANI ---
    model.add(Dense(num_classes, activation='softmax'))

    # Optimizasyon: Adam, Kayıp: Categorical Crossentropy
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train():
    # 1. Veriyi Yükle
    X, y = load_data()
    
    # 2. Use 100% data
    X_train, y_train = X, y
    
    print(f"Eğitim Verisi: {X_train.shape}")
    
    # 3. Modeli İnşa Et
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    num_classes = y.shape[1] # 4 sınıf
    
    model = build_cnn_lstm_model(input_shape, num_classes)
    model.summary() # Model özetini göster
    
    # 4. Callback'ler (Eğitimi İyileştiriciler)
    lr_reduction = ReduceLROnPlateau(monitor='loss', patience=5, verbose=1, factor=0.1, min_lr=0.00001)
    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    
    # 5. EĞİTİMİ BAŞLAT
    print("Eğitim başlıyor...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        callbacks=[lr_reduction, early_stop]
    )
    
    # 6. Modeli Kaydet (İşte burası Models klasörüne kaydediyor)
    model.save(MODEL_PATH)
    print("-" * 30)
    print(f"✅ Model başarıyla kaydedildi: {MODEL_PATH}")
    print("-" * 30)
    
    # 7. Grafikleri Çiz
    plot_history(history)

def plot_history(history):
    # Başarım ve Kayıp Grafikleri
    plt.figure(figsize=(12, 4))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Eğitim Başarısı')
    plt.title('Model Başarısı (Accuracy)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Eğitim Kaybı')
    plt.title('Model Kaybı (Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.close() # plt.show()

if __name__ == "__main__":
    train()