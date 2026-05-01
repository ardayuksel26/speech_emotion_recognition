# SER Project — Speech Emotion Recognition
**AI-Powered Multi-Model Real-Time Speech Emotion Recognition System (2026)**
**Kadir Has University — CMPE Final Project**

---

## 📌 Project Summary

A full-stack, production-grade Speech Emotion Recognition (SER) system targeting four emotion classes: **Angry, Calm, Happy, Sad**. The system operates using a state-of-the-art hybrid architecture called the **Master Ensemble**, which fuses Word-Level Majority Voting from tabular models (CatBoost, XGBoost, LightGBM) with Sentence-Level Global analysis from a deep learning Transformer (HuBERT). 

---

## 🏗️ Final Architecture (Master Ensemble)

### Production Pipeline (Main Page)
1. **Audio Input:** Microphone stream (real-time) or WAV file upload via dynamic `AudioInput` UI.
2. **Segmentation (Word-Level):** Vosk STT engine divides the audio into exact word boundaries.
3. **Feature Extraction (Tabular):** OpenSMILE IS10 Functionals extract a 1582-dim vector for each segmented word.
4. **Majority Voting (Models_2):** CatBoost V2, LightGBM V2, and XGBoost V2 evaluate each word. Results are aggregated using F1-score weighted fusion to determine the strongest tabular baseline.
5. **Global Transformer Jury:** HuggingFace `HuBERT` (SeaBenSea/hubert-large-turkish-speech-emotion-recognition) evaluates the entire unsegmented sentence to grasp overall prosodic intent.
6. **Master Fusion & Calibration:** The tabular majority voting results are fused with HuBERT's prediction. A Turkish-specific vocal calibration matrix is applied to prevent bias (`{'angry': 1.25, 'happy': 1.75, 'sad': 0.5, 'calm': 1.4}`).
7. **Frontend Dashboard:** React (Vite + TypeScript) displays an interactive, real-time emotion timeline, probability charts, and the final predicted emotion with a Glassmorphic 2026 UI.

### Experimental Laboratory
- **18 Selectable Models:** Includes robust variations, full-sentence baseline models, and cutting-edge HuggingFace integrations (SenseVoice, WavLM, ExHuBERT, Wav2Vec2 Turkish, SUPERB).
- **Joint Testing Mode:** Allows running an audio file through all top-tier models simultaneously for academic comparison.

---

## 📁 Directory Structure

| Directory | Purpose |
|---|---|
| `/Backend` | Flask API (`app.py`, 1989 lines). Core endpoint: `/analyze_master`. |
| `/Frontend` | React + Vite + TypeScript. Pages: MainPage (Production), ExperimentalPage, TechnicalInfoPage, AboutPage. |
| `/Models` | Initial baseline models (Standard + Robust variants). |
| `/Models_2` | Advanced noise-augmented V2 models (CatBoost_V2, XGBoost_V2, LightGBM_V2) used for word-level majority voting. |
| `/Huggingface` | Integrations for advanced DL models (HuBERT, WavLM, SenseVoice, Wav2Vec2). |
| `/Data`, `/Data_with_noise`, `/TurEV-DB` | Training datasets, including noise-augmented RAVDESS and Turkish Emotion Voice Database. |
| `/Train`, `/Train_2`, `/Train_3` | Historical training scripts for all model generations. |
| `/Test-Genel-1` | The comprehensive evaluation and testing suite containing real-world human recordings (320 files) and the final Master Ensemble benchmarking tools. |
| `/Old-Test-1`, `/Old-Test-2`, `/Old-Test-3` | Legacy evaluation data and benchmarking scripts from earlier project phases. |

---

## 📈 Final Performance Metrics (Master Ensemble)

**Test Dataset:** Real-world multi-speaker recordings (`Test-Genel-1/our_voices_for_test` — 320 files)
**Overall Accuracy:** **80.94%**
**Macro F1-Score:** **81.01%**
**Processing Speed:** ~1.4s per file (End-to-End)

### Emotion-Specific Breakdown
| Emotion | Precision | Recall | F1-Score |
|---|---|---|---|
| **Angry** | 91.30% | 78.75% | 84.56% |
| **Calm** | 69.57% | 80.00% | 74.42% |
| **Happy** | 81.94% | 73.75% | 77.63% |
| **Sad** | 83.91% | 91.25% | 87.43% |

### Confusion Matrix
| Actual \ Predicted | Angry | Calm | Happy | Sad |
|---|---|---|---|---|
| **Angry** | **63** | 8 | 9 | 0 |
| **Calm** | 0 | **64** | 4 | 12 |
| **Happy** | 6 | 13 | **59** | 2 |
| **Sad** | 0 | 7 | 0 | **73** |

---

