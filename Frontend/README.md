# Mastermind SER - Frontend Interface (2026)

Bu klasör, çoklu-model Sesten Duygu Tanıma (SER) projesinin kullanıcı arayüzünü barındırır. Modern, cam efektli (glassmorphic) 2026 akademik estetiğine sadık kalınarak React 18, Vite ve TailwindCSS kullanılarak inşa edilmiştir.

## 🌟 Ana Özellikler

1.  **Mastermind (Kokpit) Arayüzü (`Hero.tsx` & `Result.tsx`):**
    *   Yüklenen veya kaydedilen sese VOSK + CatBoost/XGBoost üzerinden üretim kalitesinde "Üst Akıl" tahmini uygular.
    *   Timeline bazlı *Kelime Kelime Analiz* ekranı bulunur.
    *   Yanlış okumaları engelleyen tam donanımlı *RF_Robust Veto* entegrasyonu vardır.
    *   Hızlı, duyarlı ve devasa animasyonlarla kaplı Dashboard sunar.

2.  **Deneysel Laboratuvar (`ExperimentalHero.tsx`):**
    *   Mühendislik için tasarlandı. Ses dosyalarını WhisperX, VAD veya VOSK ile parçalayıp, sisteme kayıtlı 10 farklı otonom öğrenme algoritmasıyla ayrı ayrı sınamanızı sağlar.
    *   Stüdyo ve Gürültülü simülasyon ortamları içerir.

3.  **Akademik Raporlama Yüzü (`TechnicalInfoPage.tsx`):**
    *   Tam sayfa scroll bazlı akademik araştırma özetidir. Recharts kütüphanesi kullanarak *Test* klasöründen okunan t-SNE, Mastermind Accuracy, Recall ve Precision skorlarını çizer. (Makine öğrenim modelinin başarı endeksi buradadır).

## 🛠 Kullanılan Teknolojiler
*   **React + TypeScript (Vite):** Yüksek süratli component hiyerarşisi.
*   **TailwindCSS + Framer Motion:** Karmaşık CSS animasyonlarının akıcı sekilde çalıştırılması (Scroll bar detayı ve Dashboard).
*   **Recharts:** Eğitim eğrisi ve Test Sonucu grafiklerinin dinamik oluşturulması.

## 🚀 Başlatma
1. Tüm Node gereksinimlerini kurun: `npm install`
2. Geliştirme sunucusunu çalıştırın: `npm run dev`
    *(Not: Backend API olan `app.py` sunucusunun da arkada 5000 portundan çalışıyor olması zorunludur).*
