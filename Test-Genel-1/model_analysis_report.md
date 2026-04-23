# 🎯 SER Projesi — Kapsamlı Model Analiz Raporu

**Tarih:** 23 Nisan 2026  
**Test Dataseti:** `sentencevoice_test` (320 dosya, 80 per emotion)  
**Test Yöntemi:** Word modelleri → Vosk + Majority Voting | Sentence modelleri → Tam cümle

---

## 📊 1. Tüm Modeller — Accuracy Genel Tablosu

### 🟦 Models/ (Kelime ile Eğitildi) — Word-Level Majority Voting

| Model | Accuracy | Angry R | Calm R | Happy R | Sad R |
|---|---|---|---|---|---|
| GradientBoosting_Robust | %70.31 | 0.99 | 0.75 | 0.76 | 0.31 |
| XGBoost_Robust | %69.69 | 0.95 | 0.80 | 0.64 | 0.40 |
| LightGBM_Robust | %68.44 | 0.95 | 0.80 | 0.59 | 0.40 |
| XGBoost | %65.62 | 0.97 | 0.70 | 0.51 | 0.44 |
| Random Forest_Robust | %63.75 | 0.90 | 0.57 | 0.39 | 0.69 |
| GradientBoosting | %60.62 | 0.97 | 0.45 | 0.84 | 0.16 |
| LightGBM | %53.75 | 0.90 | 0.86 | 0.15 | 0.24 |
| Random Forest | %50.31 | 0.88 | 0.56 | 0.24 | 0.34 |
| MLP_Robust | %44.38 | 0.95 | 0.49 | 0.30 | 0.04 |
| CatBoost_Robust | %46.25 | 0.99 | 0.38 | 0.28 | 0.21 |
| CatBoost | %43.12 | 1.00 | 0.34 | 0.26 | 0.12 |
| MLP | %25.94 | 1.00 | 0.01 | 0.03 | 0.00 |
| SVM / SVM_Robust | %25.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| KNN_Robust | %24.06 | 0.21 | 0.59 | 0.00 | 0.16 |
| KNN | ~%22 | düşük | — | — | — |

### 🟩 Models_2/ (V2 Serisi) — Word-Level Majority Voting

| Model | Accuracy | Angry R | Calm R | Happy R | Sad R |
|---|---|---|---|---|---|
| **CatBoost_V2** | **%91.56** ⭐ | 1.00 | 0.89 | 0.89 | 0.89 |
| **LightGBM_V2** | **%90.94** ⭐ | 1.00 | 0.86 | 0.86 | 0.91 |
| **XGBoost_V2** | **%90.31** ⭐ | 1.00 | 0.88 | 0.82 | 0.91 |
| GradientBoosting_V2 | ~%89 | yüksek | yüksek | yüksek | yüksek |
| RandomForest_V2 | %75.62 | 1.00 | 0.50 | 0.79 | 0.74 |

> ⭐ Models_2 serisi açık ara en güçlü. Kelime modellerinde zirvede.

### 🟨 Sentence_Models/ — Full Sentence Input

| Model | Accuracy | Durum |
|---|---|---|
| Random Forest | %40.94 | En iyi ama hala yetersiz |
| MLP | %26.25 | Sad'e kilitlenmiş |
| KNN | %25.31 | Dengesiz |
| SVM | %25.00 | Tamamen sad'e kilitlenmiş |
| XGBoost | %25.31 | Tamamen happy'e kilitlenmiş |
| CatBoost | %22.19 | Tamamen happy'e kilitlenmiş |
| GradientBoosting | %24.06 | Tamamen happy'e kilitlenmiş |
| LightGBM | %20.62 | En kötü |

> ⚠️ Sentence_Models serisi büyük ölçüde başarısız. Feature mismatch sebebi aşağıda.

### 🟪 HuggingFace Modelleri

| Model | Accuracy | Durum |
|---|---|---|
| **HuBERT** | **%82.81** ⭐ | Angry precision=1.00, çok güçlü |
| WavLM | %11.56 | Yanlış etiket seti (fear, contempt...) |
| Wav2Vec2Turkish | %0.00 | "negative/positive" döndürüyor |
| Wav2Vec2English | %0.00 | Kısaltılmış etiketler (ang, hap) |
| XLSR | %0.00 | 7 farklı etiket, mapping yok |
| WavLMBasePlus | %0.00 | Hata verdi |
| ExHuBERT | %0.00 | Tüm tahminler hatalı |

---

## 🔍 2. Kritik Bulgular

### 🚨 Bulgu 1: Sentence_Models Neden Çöktü?

Bu modeller eğitilirken farklı bir feature pipeline kullanılmış olmalı (farklı normalizasyon, mel spektrum eklentisi veya farklı IS10 konfigürasyonu). Test sırasında 1582-dim plain IS10 verildi ama modellerin beklediği format bu değil. Sonuç: neredeyse rasgele tahmin düzeyinde (%20-25).

### 🚨 Bulgu 2: Sad Recall Sistematik Sorunu

Tüm Models/ serisinde Angry Recall = 0.95-1.00 ama Sad Recall = 0.04-0.40. Sad sesler büyük ihtimalle Calm veya Happy olarak etiketleniyor. Bu real-time'daki temel sorunun kaynağı.

**Models_2 serisi bu sorunu büyük ölçüde çözüyor:** LightGBM_V2 Sad Recall = 0.91!

### 🚨 Bulgu 3: HF Modellerin %85'i Uyumsuz

Sadece HuBERT gerçekten çalışıyor. Diğerleri yanlış etiket seti döndürüyor veya hata veriyor. Sistem kaynaklarını boşa harcıyorlar.

---

## 🏆 3. Kullanılabilir Modeller Sıralaması

```
1. 🥇 Models_2/CatBoost_V2      → %91.56
2. 🥈 Models_2/LightGBM_V2      → %90.94
3. 🥉 Models_2/XGBoost_V2       → %90.31
4.    Models_2/GradientBoost_V2  → ~%89
5.    HuBERT (HuggingFace)       → %82.81 (cümle tabanlı)
6.    Models_2/RandomForest_V2   → %75.62
7.    Models/GradBoost_Robust    → %70.31
8.    Models/XGBoost_Robust      → %69.69
```

---

## 🗺️ 4. Stratejik Yol Haritası

### 🔴 Adım 1 — Hemen: Backend'i Models_2'ye Geçir

`app.py` içindeki `quality="studio"` modu hâlâ eski Models/ klasörünü kullanıyor.
CatBoost_V2, XGBoost_V2, LightGBM_V2 ile değiştirmek accuracy'yi **%45 → %90+** yapar.

### 🔴 Adım 2 — Hemen: Çalışmayan HF Modellerini Kaldır

WavLM, XLSR, Wav2Vec2Turkish, Wav2Vec2English, ExHuBERT → sistemden çıkar.
Sadece **HuBERT** kalsın. Real-time hızı ciddi artar.

### 🟡 Adım 3 — Önemli: HuBERT'i Majority Voting'e Ekle

HuBERT cümle bazlı çalışıyor ve Angry'de precision=1.00.
V2 modelleriyle (kelime bazlı) birlikte Weighted Voting yapılırsa boşluklar kapanır.

### 🟡 Adım 4 — Önemli: Sad Recall'u Güçlendir

Models_2 modellerini `class_weight` ile yeniden eğit ya da
`VOCAL_CALIBRATION` içinde Sad'e +15-20% boost ekle.

### ⚡ Adım 5 — Master Ensemble Kur

```
CatBoost_V2 + LightGBM_V2 + HuBERT
→ Weighted Majority Voting
→ Hedef Accuracy: %93-95+
```

### 🔬 Adım 6 — Sentence_Models Karar

Ya doğru pipeline ile yeniden eğit ya da sistemden geçici olarak çıkar.
Şu haliyle sisteme zarar veriyorlar.

---

## 📋 5. Özet Eylem Planı

| Öncelik | Eylem | Beklenen Etki |
|---|---|---|
| 🔴 Acil | Backend'i Models_2 klasörüne geçir | %45 → %90+ accuracy |
| 🔴 Acil | Çalışmayan HF modellerini devre dışı bırak | Real-time hız artışı |
| 🟡 Önemli | HuBERT'i Majority Voting'e ekle | Angry precision sabitlenir |
| 🟡 Önemli | Sad için VOCAL_CALIBRATION boost | Sad recall artışı |
| 🟢 Sonra | Sentence_Models doğru pipeline ile yeniden eğit | Potansiyel %60-70+ |
| 🟢 Sonra | 3 modeli Weighted Ensemble ile birleştir | Hedef %93-95+ |
