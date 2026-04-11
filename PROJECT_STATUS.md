# SER Project - Speech Emotion Recognition
**Yapay Zeka Destekli Çoklu-Model Sesten Duygu Tanıma (SER) Sistemi 2026**

## 📌 Proje Özeti
Bu proje, sadece tekil sesten değil, cümle ve kelime düzeyindeki aralıklardan duygu geçişlerini yüksek isabetle tanıyan gelişmiş bir ensemble (harmanlanmış) derin öğrenme projesidir. Sistem, **Mastermind (Üst Akıl)** adını verdiğimiz özel bir yönetim mekanizmasına ve "Experimental (Deneysel)" adını taşıyan bir laboratuvar arayüzüne sahiptir.

Proje 4 ana sınıf üzerinde (**Angry, Calm, Happy, Sad**) odaklanmıştır ve yüksek kaliteli (1584 boyutlu vektör) özellik çıkarım yöntemleriyle (Librosa, VAD, MFCC) eğitilmiştir.

---

## 🚀 Proje Mimarisi (2026 Vizyonu)

### 1- Mastermind (Üst Akıl) Yapısı
Ana sayfa (Hero.tsx), üretim ortamını (Production) temsil eder. Kullanıcıdan alınan mikrofon sesi veya wav dosyası arka planda şu işlemlerden geçer:
1. **Silero VAD & VOSK Segmentasyonu:** Ses kelime parçalarına bölünür ve gürültüden arındırılır.
2. **Feature Extraction:** MFCC, Chroma, Spectral Contrast analizleri çıkartılır.
3. **CatBoost & XGBoost Tahmini:** Ağırlıklı modeller kelimenin duygu ihtimallerini hesaplar.
4. **RF_Robust Detektörü (VETO Sistemi):** "Sad" (Üzgün) olarak işaretlenen segmentler yapay veya gürültülü sessizlikler olabileceğinden, Random Forest Robust modeli tarafından denetlenip gerekirse veto edilip "Calm" veya "Happy" sınıflarına dönüştürülür.
5. **Arayüz (React):** Sonuçlar, "Timeline (Zaman Çizelgesi)" üzerinden kelime kelime gösterilir.

### 2- Experimental (Deneysel) Laboratuvar
Testleri şeffaflaştırmak amacıyla projenin içine bir laboratuvar arayüzü kurulmuştur:
- **10 Farklı Model:** XGBoost, CatBoost, RF_Robust, CNN1D, SVM, KNN gibi modellerden herhangi biri manuel seçilerek test edilebilir.
- **Segmentasyon Aracı Seçimi:** VOSK, WhisperX veya basit Silero VAD arasında geçiş yapılarak modellerin tepkisi ölçülebilir.
- **Stüdyo vs Gürültü Modu:** Ses girişlerine gerçek zamanlı gürültü enjekte edilerek dayanıklılık (robustness) simülasyonları yapılır.

---

## 📁 Dizin Yapısı (Project Structure)

- **/Frontend:** Projenin son kullanıcı ile etkileşime girdiği yerdir. React (Vite, TypeScript), Tailwind CSS ve Framer Motion ile tasarlanan yüksek görselliğe (Glassmorphic) sahip modern 2026 estetiğine ve animasyonlarına sahiptir.
- **/Backend:** FastAPI ile yazılmış Python sunucusu (`app.py`). `BackgroundTasks` kullanarak ağır ses işlemlerini asenkron çözer.
- **/Test:** Projenin doğruluğunu kanıtlayan, Confusion Matrix, Recall, F1 skorlarını çıkaran script dizini. Frontend için `realWorldResults.ts` buradan beslenir.
- **/Models:** Eğitilmiş algoritmaların pkl veya keras sürümleri bulunur.
- **/Data:** RAVDESS tabanlı data augmentation uygulanmış ses ve özellik (feature.npy) setleri bulunur.

---

## 📈 Güncel Performans (Mastermind Production)
Sentetik Cümle Benchmark sonuçlarına göre sistem:
- **Genel Doğruluk (Accuracy):** ~0.6456 (Zorlu Şartlarda)
- **Güçlü Halka:** "Angry" ve "Calm" sınıflarında oldukça kararlı ve %80 üzeri Recall üretmektedir. "Sad" için özel Veto uygulamaktadır.

## 🔮 Gelecek Geliştirmeler (TODO)
- Deneysel alanda, eksik olan modeller için kelime sonuçlarının kaydedilmesi (benchmark testlerinin yenilenmesi).
- Timeline üzerindeki beyaz alanların, karanlık model (dark mode) sistemine tam entegre çalıştırılarak responsive tasarımların mobil için bitirilmesi.
- Yeni Türkçe Veri Seti (TurEV-DB vb.) dahil edilerek kelime öbeklerinin Türkçe spesifik vurgu analizine geçirilmesi.
