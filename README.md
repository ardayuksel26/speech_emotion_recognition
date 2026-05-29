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
<img width="1918" height="907" alt="Main_Page" src="https://github.com/user-attachments/assets/5db97759-de7f-48b6-9eb3-1429d552d476" />


### 2. Experimental Models / Deneysel Modeller
<img width="1918" height="910" alt="Experimental_Models_Page" src="https://github.com/user-attachments/assets/9bbb1d04-c0ed-4044-879c-1d3bdf0676bf" />


### 3. About Us Page / Hakkımızda Sayfası
<img width="1896" height="896" alt="About_Us_Page" src="https://github.com/user-attachments/assets/11e5f300-f5f8-49cf-85fe-fb37672fba38" />


### 4. Use Cases / Kullanım Alanları
<img width="1918" height="907" alt="Use_Cases_Page" src="https://github.com/user-attachments/assets/27be6228-5227-4d54-bf6a-6f62d479b289" />


### 5. Technical Info Page / Teknik Bilgi Sayfası
<img width="1895" height="887" alt="Technical_info_page" src="https://github.com/user-attachments/assets/d7bd67f6-f897-4a33-bf62-73b2fc87d10a" />


### 6. Result Screen / Sonuç Ekranı
<img width="1918" height="907" alt="Result_Screen" src="https://github.com/user-attachments/assets/ade849c0-d804-417c-a11f-3d58a16197a8" />


---
© 2026 Speech Emotion Recognition Project
