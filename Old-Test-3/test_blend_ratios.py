"""
Test3/test_blend_ratios.py
Models_2 ensemble ağırlık oranlarını test eder (VOSK yok, global IS10).
LightGBM (en iyi bireysel model) ile diğer 4 modelin ağırlık oranı 20 adımda test edilir.
Oran r: LightGBM ağırlığı = r, diğer 4 model eşit paylaşır (1-r)/4.
r = 0.05, 0.10, ..., 1.00 → 20 test noktası.
"""

import os, sys, time, warnings
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, classification_report

warnings.filterwarnings('ignore')

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(BASE_DIR, ".."))

import opensmile

TEST_SOUNDS_DIR = os.path.join(PROJECT_ROOT, "All_Sounds", "Test_Sounds")
MODELS_2_DIR    = os.path.join(PROJECT_ROOT, "Models_2")
RESULTS_DIR     = os.path.join(BASE_DIR, "Results")
RESULTS_FILE    = os.path.join(RESULTS_DIR, "blend_ratio_results.txt")
os.makedirs(RESULTS_DIR, exist_ok=True)

EMOTION_ORDER = ['angry', 'calm', 'happy', 'sad']
# LightGBM ağırlığı: %5 → %100, adım %5 (20 oran)
RATIOS = [round(r, 2) for r in np.arange(0.05, 1.01, 0.05)]

# ── Logger ────────────────────────────────────────────────────────────────────
class Logger:
    def __init__(self, path):
        self.f = open(path, 'w', encoding='utf-8')
    def log(self, msg=""):
        print(msg)
        self.f.write(msg + "\n")
    def close(self):
        self.f.close()

log = Logger(RESULTS_FILE)

# ── Modelleri Yükle ──────────────────────────────────────────────────────────
MODELS_2_CONFIG = {
    'lgbm_v2': ('LightGBM/lightgbm_model.pkl',                  'LightGBM/scaler_lgbm.pkl',            'LightGBM/label_encoder_lgbm.pkl'),
    'rf_v2':   ('RandomForest/random_forest_model.pkl',          'RandomForest/scaler_rf.pkl',          'RandomForest/label_encoder_rf.pkl'),
    'xgb_v2':  ('XGBoost/xgboost_model.pkl',                    'XGBoost/scaler_xgb.pkl',              'XGBoost/label_encoder_xgb.pkl'),
    'cb_v2':   ('CatBoost/catboost_model.pkl',                   'CatBoost/scaler_catboost.pkl',        'CatBoost/label_encoder_catboost.pkl'),
    'gb_v2':   ('GradientBoosting/gradient_boosting_model.pkl',  'GradientBoosting/scaler_gb.pkl',      'GradientBoosting/label_encoder_gb.pkl'),
}
MODEL_NAMES = {
    'lgbm_v2': 'LightGBM',
    'rf_v2':   'Random Forest',
    'xgb_v2':  'XGBoost',
    'cb_v2':   'CatBoost',
    'gb_v2':   'Gradient Boosting',
}

log.log("Modeller yükleniyor...")
models = {}
for key, (mp, sp, ep) in MODELS_2_CONFIG.items():
    m_path = os.path.join(MODELS_2_DIR, mp)
    if os.path.exists(m_path):
        models[key] = {
            'model':   joblib.load(m_path),
            'scaler':  joblib.load(os.path.join(MODELS_2_DIR, sp)),
            'encoder': joblib.load(os.path.join(MODELS_2_DIR, ep)),
        }
        log.log(f"  ✓ {MODEL_NAMES[key]}")
    else:
        log.log(f"  ✗ {key} bulunamadı: {m_path}")

log.log(f"  {len(models)} model yüklendi.\n")
if len(models) == 0:
    log.log("HATA: Hiç model yüklenemedi!"); log.close(); sys.exit(1)

# ── OpenSMILE ────────────────────────────────────────────────────────────────
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.IS10,
    feature_level=opensmile.FeatureLevel.Functionals,
)

def extract_is10(file_path):
    try:
        df = smile.process_file(file_path)
        f  = df.to_numpy().flatten().astype(np.float32)
        return f if len(f) == 1582 else None
    except:
        return None

# ── Test Seslerini Yükle ─────────────────────────────────────────────────────
wav_files = sorted(Path(TEST_SOUNDS_DIR).glob("*.wav"))
log.log(f"Test sesleri: {len(wav_files)} dosya")
if not wav_files:
    log.log("HATA: Test sesi bulunamadı!"); log.close(); sys.exit(1)

label_counts = {}
for wf in wav_files:
    lbl = wf.stem.split('_')[0].lower()
    label_counts[lbl] = label_counts.get(lbl, 0) + 1
for lbl in sorted(label_counts):
    log.log(f"  {lbl:<10}: {label_counts[lbl]}")

# ── Aşama 1: IS10 Çıkar + Tüm Modelleri Bir Kez Çalıştır ────────────────────
log.log("\n" + "="*70)
log.log("  AŞAMA 1 — IS10 Çıkarımı + Model Tahminleri (bir kez çalışır)")
log.log("="*70)

# Her örnek için: true_label + her modelin 4-boyutlu olasılık vektörü
# {model_key: probs[4]}
all_true   = []
all_probs  = {k: [] for k in models}   # model_key → list of np.array(4)
errors     = 0
t_start    = time.time()

for idx, wav_path in enumerate(wav_files):
    true_label = wav_path.stem.split('_')[0].lower()
    if true_label not in EMOTION_ORDER:
        errors += 1
        continue

    feat = extract_is10(str(wav_path))
    if feat is None:
        errors += 1
        continue

    all_true.append(true_label)
    for key, tools in models.items():
        m, s, e = tools['model'], tools['scaler'], tools['encoder']
        f_scaled = s.transform(feat.reshape(1, -1))
        p_raw    = m.predict_proba(f_scaled)[0]
        p_mapped = np.zeros(4)
        for i, cls in enumerate(e.classes_):
            if cls.lower() in EMOTION_ORDER:
                p_mapped[EMOTION_ORDER.index(cls.lower())] = p_raw[i]
        all_probs[key].append(p_mapped)

    if (idx + 1) % 300 == 0:
        log.log(f"  [{idx+1}/{len(wav_files)}] — {time.time()-t_start:.0f}s")

elapsed = time.time() - t_start
n = len(all_true)
log.log(f"\n  Tamamlandı — {n} örnek, {errors} hata, {elapsed:.1f}s")

# numpy dizilerine çevir
for key in all_probs:
    all_probs[key] = np.array(all_probs[key])   # shape (n, 4)

# ── Aşama 2: Bireysel Model Sonuçları ────────────────────────────────────────
log.log("\n" + "="*70)
log.log("  AŞAMA 2 — Bireysel Model Performansı")
log.log("="*70)
log.log(f"  {'Model':<22} {'Accuracy':>10} {'F1-Score':>10}")
log.log("-"*46)

individual_acc = {}
for key, probs in all_probs.items():
    preds = [EMOTION_ORDER[int(np.argmax(p))] for p in probs]
    acc   = accuracy_score(all_true, preds) * 100
    f1    = f1_score(all_true, preds, average='weighted', labels=EMOTION_ORDER) * 100
    individual_acc[key] = acc
    log.log(f"  {MODEL_NAMES[key]:<22} {acc:>9.2f}% {f1:>9.2f}%")

# ── Aşama 3: Eşit Ağırlık Ensemble (referans) ───────────────────────────────
log.log("\n" + "="*70)
log.log("  AŞAMA 3 — Eşit Ağırlık Ensemble (referans)")
log.log("="*70)

equal_ensemble = sum(all_probs[k] for k in models) / len(models)
equal_preds    = [EMOTION_ORDER[int(np.argmax(p))] for p in equal_ensemble]
equal_acc      = accuracy_score(all_true, equal_preds) * 100
equal_f1       = f1_score(all_true, equal_preds, average='weighted', labels=EMOTION_ORDER) * 100
log.log(f"  Eşit Ağırlık — Accuracy: %{equal_acc:.2f}  F1: %{equal_f1:.2f}")

# ── Aşama 4: LightGBM Ağırlık Oranı Testi ───────────────────────────────────
log.log("\n" + "="*70)
log.log("  AŞAMA 4 — LightGBM Ağırlık Oranı Testi")
log.log("  (r = LightGBM ağırlığı, diğer 4 model eşit: (1-r)/4 her biri)")
log.log("="*70)
log.log(f"  {'LightGBM Ağırlığı':<22} {'Diğerleri':>10} {'Accuracy':>10} {'F1-Score':>10}")
log.log("-"*56)

OTHER_KEYS = [k for k in models if k != 'lgbm_v2']
ratio_results = {}

for r in RATIOS:
    other_w = (1.0 - r) / len(OTHER_KEYS) if OTHER_KEYS else 0.0
    blended = all_probs['lgbm_v2'] * r
    for ok in OTHER_KEYS:
        blended = blended + all_probs[ok] * other_w
    # normalize her satırı
    row_sums = blended.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    blended /= row_sums

    preds = [EMOTION_ORDER[int(np.argmax(p))] for p in blended]
    acc   = accuracy_score(all_true, preds) * 100
    f1    = f1_score(all_true, preds, average='weighted', labels=EMOTION_ORDER) * 100
    ratio_results[r] = {'acc': acc, 'f1': f1, 'preds': preds}

best_r   = max(ratio_results, key=lambda x: ratio_results[x]['acc'])
best_acc = ratio_results[best_r]['acc']

for r in RATIOS:
    res     = ratio_results[r]
    other_w = (1.0 - r) / len(OTHER_KEYS) if OTHER_KEYS else 0.0
    marker  = "  ◄ EN İYİ" if r == best_r else ""
    log.log(f"  %{r*100:>5.1f}             %{other_w*100:>5.1f} (x4)  {res['acc']:>9.2f}% {res['f1']:>9.2f}%{marker}")

# ── Aşama 5: Duygu Bazlı Analiz (en iyi oran) ───────────────────────────────
log.log("\n" + "="*70)
log.log(f"  AŞAMA 5 — En İyi Oran Duygu Bazlı Analiz")
log.log(f"  LightGBM ağırlığı: %{best_r*100:.0f}  —  Accuracy: %{best_acc:.2f}")
log.log("="*70)
best_preds = ratio_results[best_r]['preds']
log.log(classification_report(all_true, best_preds, labels=EMOTION_ORDER))

# ── Aşama 6: Accuracy-Weighted Ensemble ─────────────────────────────────────
log.log("\n" + "="*70)
log.log("  AŞAMA 6 — Accuracy-Weighted Ensemble (bireysel doğruluğa göre ağırlık)")
log.log("="*70)
total_acc = sum(individual_acc.values())
acc_weighted = sum(all_probs[k] * (individual_acc[k] / total_acc) for k in models)
row_sums = acc_weighted.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
acc_weighted /= row_sums
aw_preds = [EMOTION_ORDER[int(np.argmax(p))] for p in acc_weighted]
aw_acc   = accuracy_score(all_true, aw_preds) * 100
aw_f1    = f1_score(all_true, aw_preds, average='weighted', labels=EMOTION_ORDER) * 100
log.log(f"  Accuracy-Weighted — Accuracy: %{aw_acc:.2f}  F1: %{aw_f1:.2f}")
for k in models:
    log.log(f"    {MODEL_NAMES[k]:<22} ağırlık: %{individual_acc[k]/total_acc*100:.1f}")

# ── Özet ────────────────────────────────────────────────────────────────────
log.log("\n" + "="*70)
log.log("  ÖZET")
log.log("="*70)
log.log(f"  Eşit Ağırlık Ensemble  : %{equal_acc:.2f}")
log.log(f"  En İyi LightGBM Oranı  : %{best_acc:.2f}  (LightGBM %{best_r*100:.0f})")
log.log(f"  Accuracy-Weighted      : %{aw_acc:.2f}")
log.log(f"  En İyi Bireysel Model  : LightGBM %{individual_acc['lgbm_v2']:.2f}")
log.log("="*70)
log.log(f"\nSonuçlar: {RESULTS_FILE}")
log.close()
