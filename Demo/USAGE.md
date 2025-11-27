# Türkçe Sesten Duygu Tanıma Projesi - Kullanım Kılavuzu

## 🚀 Hızlı Başlangıç

### Seçenek 1: Otomatik Çalıştırma (Önerilen)
```bash
python run_project.py
```
Bu komut:
1. Eğer model yoksa, önce en iyi modeli eğitir
2. Sonra demo uygulamasını başlatır

### Seçenek 2: Manuel Çalıştırma

#### 1. Model Eğitimi
```bash
python speech_emotion_recognition.py
```
Bu komut:
- Farklı algoritmaları (Random Forest, SVM, Neural Network, Gradient Boosting) eğitir
- En iyi performans gösteren modeli seçer
- Modeli `models/best_emotion_model.pkl` dosyasına kaydeder
- Sonuçları görselleştirir

#### 2. Demo Uygulaması
```bash
streamlit run demo_app.py
```
Bu komut:
- Eğitilmiş en iyi modeli yükler
- Web tabanlı demo arayüzünü başlatır
- Ses dosyası yükleme ve analiz özelliği sunar

## 📁 Proje Yapısı

```
├── speech_emotion_recognition.py  # Ana eğitim scripti
├── demo_app.py                   # Streamlit demo uygulaması
├── model_utils.py                # Model yükleme/kaydetme fonksiyonları
├── run_project.py                # Otomatik çalıştırma scripti
├── models/                       # Eğitilmiş modeller
│   └── best_emotion_model.pkl    # En iyi model
├── Extracted_CSV/                # Özellik verileri
│   ├── angry.csv
│   ├── calm.csv
│   ├── happy.csv
│   └── sad.csv
└── Sound_Source/                 # Ses dosyaları
    ├── Angry/
    ├── Calm/
    ├── Happy/
    └── Sad/
```

## 🎯 Özellikler

### Model Eğitimi
- **Çoklu algoritma desteği**: Random Forest, SVM, Neural Network, Gradient Boosting
- **Otomatik model seçimi**: En yüksek doğruluk skoruna sahip model seçilir
- **Çapraz doğrulama**: 5-fold cross validation ile güvenilir performans ölçümü
- **Özellik mühendisliği**: Korelasyon analizi ve normalizasyon
- **Model kaydetme**: En iyi model otomatik olarak kaydedilir

### Demo Uygulaması
- **Gerçek zamanlı analiz**: WAV dosyalarını yükleyip analiz edin
- **Görsel analiz**: Dalga formu ve spektrogram görselleştirmesi
- **Duygu tahmini**: 4 duygu sınıfı (Kızgın, Sakin, Mutlu, Üzgün)
- **Güven skoru**: Tahmin güvenilirlik oranı
- **Model performansı**: Eğitilmiş modelin detaylı performans bilgileri

## 🔧 Gereksinimler

Gerekli paketleri yüklemek için:
```bash
pip install -r requirements.txt
```

## 📊 Model Performansı

Eğitim tamamlandıktan sonra:
- En iyi model otomatik olarak seçilir ve kaydedilir
- Performans metrikleri (doğruluk, F1-score, vb.) hesaplanır
- Confusion matrix ve özellik önemleri görselleştirilir
- Sonuçlar PNG dosyası olarak kaydedilir

## 🎵 Ses Dosyası Formatı

- **Format**: WAV
- **Örnekleme oranı**: 16 kHz (otomatik dönüştürülür)
- **Kanal**: Mono (stereo dosyalar otomatik dönüştürülür)
- **Dil**: Türkçe konuşma

## 🚨 Sorun Giderme

### Model bulunamadı hatası
```bash
python speech_emotion_recognition.py
```
Önce modeli eğitin.

### CSV dosyaları bulunamadı
`Extracted_CSV/` klasörünün var olduğunu ve içinde CSV dosyalarının bulunduğunu kontrol edin.

### Streamlit hatası
```bash
pip install streamlit
```
Streamlit'in yüklü olduğundan emin olun.

## 📈 Sonuçlar

- Model performans grafikleri: `speech_emotion_recognition_results.png`
- Eğitilmiş model: `models/best_emotion_model.pkl`
- Demo uygulaması: Web tarayıcısında açılır (genellikle http://localhost:8501)