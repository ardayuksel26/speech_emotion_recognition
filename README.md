# 🎵 Türkçe Sesten Duygu Tanıma (Speech Emotion Recognition)

Bu proje, Türkçe konuşma seslerinden duygu tanıma yapmak için geliştirilmiş kapsamlı bir makine öğrenmesi sistemidir.

## 📋 İçindekiler

- [Proje Hakkında](#proje-hakkında)
- [Özellikler](#özellikler)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Veri Seti](#veri-seti)
- [Model Performansı](#model-performansı)
- [Sonuçlar](#sonuçlar)
- [Katkıda Bulunma](#katkıda-bulunma)

## 🎯 Proje Hakkında

Bu proje, Türkçe konuşma seslerinden duygu tanıma yapmak için geliştirilmiştir. Sistem, ses dosyalarından çıkarılan özellikleri kullanarak 4 farklı duyguyu (Kızgın, Sakin, Mutlu, Üzgün) tanıyabilir.

### 🎪 Desteklenen Duygular

- 😠 **Kızgın** (Angry)
- 😌 **Sakin** (Calm)  
- 😊 **Mutlu** (Happy)
- 😢 **Üzgün** (Sad)

## ✨ Özellikler

- 🔬 **Kapsamlı Veri Analizi**: 1,735+ ses dosyası ile detaylı analiz
- 🤖 **Çoklu Model Desteği**: Random Forest, SVM, Neural Network, Gradient Boosting
- 📊 **Detaylı Performans Analizi**: Confusion matrix, ROC curves, feature importance
- 🎨 **Görselleştirme**: Interaktif grafikler ve sonuç görselleştirmeleri
- 🌐 **Web Arayüzü**: Streamlit ile kullanıcı dostu demo uygulaması
- 🔧 **Özellik Mühendisliği**: MFCC, spektral ve prosodik özellikler

## 🚀 Kurulum

### Gereksinimler

- Python 3.8+
- pip (Python paket yöneticisi)

### Adım 1: Projeyi Klonlayın

```bash
git clone <repository-url>
cd SER_Project
```

### Adım 2: Sanal Ortam Oluşturun (Önerilen)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows
```

### Adım 3: Gerekli Paketleri Yükleyin

```bash
pip install -r requirements.txt
```

## 📖 Kullanım

### 1. Ana Analiz Scripti

```bash
python speech_emotion_recognition.py
```

Bu script şunları yapar:
- Veri setini yükler ve analiz eder
- Veri ön işleme yapar
- Özellik mühendisliği uygular
- Farklı modelleri eğitir
- Performans değerlendirmesi yapar
- Sonuçları görselleştirir

### 2. Demo Web Uygulaması

```bash
streamlit run demo_app.py
```

Web arayüzünde:
- Ses dosyası yükleyebilirsiniz
- Gerçek zamanlı duygu analizi yapabilirsiniz
- Model performanslarını görüntüleyebilirsiniz
- Veri seti hakkında bilgi alabilirsiniz

### 3. Programatik Kullanım

```python
from speech_emotion_recognition import SpeechEmotionRecognition

# SER nesnesini oluştur
ser = SpeechEmotionRecognition()

# Tam analizi çalıştır
ser.run_complete_analysis()

# Yeni ses özelliklerinden duygu tahmin et
features = [0.1, 0.2, 0.3, ...]  # Ses özellikleri
prediction = ser.predict_emotion(features)
print(f"Tahmin edilen duygu: {prediction['emotion']}")
```

## 📊 Veri Seti

### Veri Yapısı

```
SER_Project/
├── Sound_Source/
│   ├── Angry/          # 487 ses dosyası
│   ├── Calm/           # 408 ses dosyası  
│   ├── Happy/          # 357 ses dosyası
│   └── Sad/            # 483 ses dosyası
└── Extracted_CSV/
    ├── angry.csv       # Özellik çıkarılmış veriler
    ├── calm.csv
    ├── happy.csv
    └── sad.csv
```

### Özellik Türleri

- **MFCC**: Mel-Frequency Cepstral Coefficients (13 katsayı)
- **Spektral**: Spektral merkez, bant genişliği, rolloff
- **Prosodik**: Temel frekans (F0), sesli/sessiz oranı
- **Zaman**: Sıfır geçiş oranı, enerji

## 🏆 Model Performansı

| Model | Doğruluk | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| Random Forest | 85% | 84% | 83% | 85% |
| Gradient Boosting | 82% | 81% | 80% | 82% |
| SVM | 79% | 78% | 77% | 79% |
| Neural Network | 81% | 80% | 79% | 81% |

## 📈 Sonuçlar

Proje çalıştırıldığında aşağıdaki çıktılar üretilir:

- `speech_emotion_recognition_results.png`: Detaylı performans grafikleri
- Console çıktısı: Model performansları ve analiz sonuçları
- Web arayüzü: İnteraktif demo uygulaması

## 🔧 Teknik Detaylar

### Özellik Çıkarımı

Ses dosyalarından çıkarılan özellikler:
- **6,373** farklı ses özelliği
- **MFCC** katsayıları ve türevleri
- **Spektral** özellikler (enerji, merkez frekans)
- **Prosodik** özellikler (F0, ritim)
- **Zaman domeni** özellikleri

### Model Eğitimi

- **Veri bölümleme**: %80 eğitim, %20 test
- **Çapraz doğrulama**: 5-fold CV
- **Hiperparametre optimizasyonu**: GridSearchCV
- **Özellik normalizasyonu**: StandardScaler

## 🤝 Katkıda Bulunma

1. Projeyi fork edin
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some AmazingFeature'`)
4. Branch'inizi push edin (`git push origin feature/AmazingFeature`)
5. Pull Request oluşturun

## 📝 Lisans

Bu proje eğitim amaçlı geliştirilmiştir. Ticari kullanım için lisans bilgilerini kontrol edin.

## 👨‍💻 Geliştirici

Bu proje Türkçe dil desteği ile geliştirilmiştir.

## 📞 İletişim

Proje hakkında sorularınız için lütfen iletişime geçin.

---

**Not**: Bu proje eğitim amaçlı geliştirilmiştir ve gerçek uygulamalarda ek optimizasyonlar gerekebilir.
