import numpy as np
import os

# Dosya yolları
DATA_ROOT = "Data"
features_path = os.path.join(DATA_ROOT, "features_spectrograms.npy")
labels_path = os.path.join(DATA_ROOT, "labels_onehot.npy")

# 1. Dosyaları Yükle
print("Dosyalar yükleniyor...")
X = np.load(features_path)
y = np.load(labels_path)

# 2. Boyutları (Shape) Kontrol Et
print("\n--- BOYUT KONTROLÜ ---")
print(f"X (Girdiler) Boyutu: {X.shape}") 
# Beklenen: (Örnek Sayısı, 128, 130, 1) -> (Adet, Frekans, Zaman, Kanal)

print(f"y (Etiketler) Boyutu: {y.shape}") 
# Beklenen: (Örnek Sayısı, 4) -> (Adet, Sınıf Sayısı)

# 3. İçerik Örnekleri
print("\n--- İÇERİK ÖRNEKLERİ ---")
print(f"İlk Sesin Etiketi (One-Hot): {y[0]}")
print(f"İlk Sesin Spektrogram Verisinden Ufak Bir Parça:\n{X[0, :5, :5, 0]}") # İlk 5x5 piksel