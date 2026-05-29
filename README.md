# Speech Emotion Recognition (SER) Project / Konuşmadan Duygu Tanıma Projesi

[English](#english) | [Türkçe](#türkçe)

---

## English

### 📌 Project Overview
This project is an advanced **Speech Emotion Recognition (SER)** system developed as a CMPE Final Project at Kadir Has University. It utilizes a state-of-the-art hybrid architecture called the **Master Ensemble**, combining word-level tabular models with global sentence-level Transformer models to achieve high-accuracy emotion detection in Turkish and English speech.

The system focuses on four core emotions: **Angry, Calm, Happy, and Sad**.

### 🚀 Key Features
- **Master Ensemble Architecture:** Fuses predictions from tabular models (CatBoost, XGBoost, LightGBM) and Deep Learning Transformers (HuBERT).
- **Real-time Processing:** Live microphone recording and instant analysis.
- **Multi-Model Support:** 18+ experimental models including CNN, DNN, and various HuggingFace models (SenseVoice, WavLM, etc.).
- **Word-Level Analysis:** Visualizes emotion changes throughout a sentence using Vosk STT integration.
- **Noise Robustness:** Models trained with noise-augmented datasets for real-world reliability.
- **Modern UI:** Responsive, glassmorphic 2026 design built with React and Tailwind CSS.

### 🛠️ Installation

#### 1. Backend Setup
The backend is built with Flask.
```bash
# Clone the repository
git clone <repository-url>
cd Speech_Emotion_Recognition_Project

# Install Python dependencies
pip install -r Backend/requirements.txt

# Start the Flask server
cd Backend
python app.py
```

#### 2. Frontend Setup
The frontend is built with React and Vite.
```bash
# Navigate to the frontend directory
cd Frontend

# Install NPM dependencies
npm install

# Start the development server
npm run dev
```

### 📊 Technical Specifications
- **Accuracy:** ~80.94% on real-world test data.
- **Preprocessing:** OpenSMILE (IS10 Functionals) for feature extraction (1582 dimensions).
- **Inference Time:** ~1.4 seconds end-to-end.

---

## Türkçe

### 📌 Proje Özeti
Bu proje, Kadir Has Üniversitesi CMPE Bitirme Projesi kapsamında geliştirilmiş ileri düzey bir **Konuşmadan Duygu Tanıma (SER)** sistemidir. Sistem, kelime bazlı tabüler modelleri ve cümle bazlı Transformer modellerini birleştiren **Master Ensemble** adlı hibrit bir mimari kullanarak yüksek doğrulukta duygu tespiti yapar.

Sistem dört temel duyguya odaklanır: **Öfkeli (Angry), Sakin (Calm), Mutlu (Happy) ve Üzgün (Sad)**.

### 🚀 Temel Özellikler
- **Master Ensemble Mimarisi:** Tabüler modeller (CatBoost, XGBoost, LightGBM) ve Derin Öğrenme Transformer modellerinin (HuBERT) tahminlerini harmanlar.
- **Gerçek Zamanlı İşleme:** Canlı mikrofon kaydı ve anlık analiz imkanı.
- **Çoklu Model Desteği:** CNN, DNN ve çeşitli HuggingFace modelleri (SenseVoice, WavLM vb.) dahil 18'den fazla deneysel model.
- **Kelime Bazlı Analiz:** Vosk STT entegrasyonu ile cümle içindeki duygu değişimlerini görselleştirir.
- **Gürültüye Dayanıklılık:** Gerçek dünya koşulları için gürültü ile artırılmış veri setleriyle eğitilmiş modeller.
- **Modern Arayüz:** React ve Tailwind CSS ile oluşturulmuş, kullanıcı dostu cam (glassmorphic) tasarım.

### 🛠️ Kurulum

#### 1. Backend (Sunucu) Kurulumu
Backend Flask ile geliştirilmiştir.
```bash
# Depoyu klonlayın
git clone <repository-url>
cd Speech_Emotion_Recognition_Project

# Python bağımlılıklarını yükleyin
pip install -r Backend/requirements.txt

# Flask sunucusunu başlatın
cd Backend
python app.py
```

#### 2. Frontend (Arayüz) Kurulumu
Frontend React ve Vite ile geliştirilmiştir.
```bash
# Frontend dizinine gidin
cd Frontend

# NPM bağımlılıklarını yükleyin
npm install

# Geliştirme sunucusunu başlatın
npm run dev
```

### 📊 Teknik Özellikler
- **Doğruluk Oranı:** Gerçek dünya test verilerinde ~%80.94.
- **Önişleme:** Özellik çıkarımı için OpenSMILE (IS10 Functionals) kullanımı (1582 boyut).
- **Tahmin Süresi:** Uçtan uca yaklaşık 1.4 saniye.

---

## 📸 Screenshots / Uygulama Ekran Görüntüleri

### 1. Main Page / Ana Sayfa
![Main Page](https://via.placeholder.com/800x450?text=Main+Page)

### 2. Experimental Models / Deneysel Modeller
![Experimental Models](https://via.placeholder.com/800x450?text=Experimental+Models)

### 3. About Us Page / Hakkımızda Sayfası
![About Us](https://via.placeholder.com/800x450?text=About+Us+Page)

### 4. Use Cases / Kullanım Alanları
![Use Cases](https://via.placeholder.com/800x450?text=Use+Cases)

### 5. Technical Info Page / Teknik Bilgi Sayfası
![Technical Info](https://via.placeholder.com/800x450?text=Technical+Info+Page)

### 6. Result Screen / Sonuç Ekranı
![Result Screen](https://via.placeholder.com/800x450?text=Result+Screen)

---
© 2026 Speech Emotion Recognition Project