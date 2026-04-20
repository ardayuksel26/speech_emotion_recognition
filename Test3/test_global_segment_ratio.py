"""
Test3/test_global_segment_ratio.py
Cümle Analizi için Global (G) ve Segment (S) ağırlık oranlarını test eder.
En iyi model dağılımını (LightGBM %35, diğerleri %16.25) sabit tutar.
Global ağırlığı G = 0.1, 0.2, ..., 0.9 arasında taranır. (S = 1-G)
Veriseti: Test/sentencevoice (79 dosya)
"""

import os, sys, time, warnings
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, classification_report
import librosa
import soundfile as sf
import tempfile

warnings.filterwarnings('ignore')

# Proje kök dizinini ekle
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(BASE_DIR, ".."))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "Backend"))

import opensmile
from stt_service import transcribe

TEST_SENTENCE_DIR = os.path.join(PROJECT_ROOT, "Test", "sentencevoice")
MODELS_2_DIR      = os.path.normpath(os.path.join(PROJECT_ROOT, "Models_2"))
RESULTS_DIR       = os.path.join(BASE_DIR, "Results")
RESULTS_FILE      = os.path.join(RESULTS_DIR, "global_segment_ratio_results.txt")
os.makedirs(RESULTS_DIR, exist_ok=True)

EMOTION_ORDER = ['angry', 'calm', 'happy', 'sad']

# En iyi model oylama ağırlıkları (Önceki testten gelen)
MODEL_WEIGHTS = {
    'lgbm_v2': 0.35,
    'rf_v2':   0.1625,
    'xgb_v2':  0.1625,
    'cb_v2':   0.1625,
    'gb_v2':   0.1625,
}

# Global ağırlıkları: %10 → %90, adım %10
GLOBAL_RATIOS = [round(r, 1) for r in np.arange(0.1, 1.0, 0.1)]

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
        log.log(f"  ✓ {key}")
    else:
        log.log(f"  ✗ {key} bulunamadı: {m_path}")

log.log(f"  {len(models)} model yüklendi.\n")

# ── OpenSMILE & Feature Extractor ──────────────────────────────────────────
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.IS10,
    feature_level=opensmile.FeatureLevel.Functionals,
)

def extract_features_plain(file_path):
    try:
        df = smile.process_file(file_path)
        f  = df.to_numpy().flatten().astype(np.float32)
        return f if len(f) == 1582 else None
    except:
        return None

# ── Test Seslerini Yükle ─────────────────────────────────────────────────────
wav_files = []
for emo_folder in ["Angry", "Calm", "Happy", "Sad"]:
    folder = os.path.join(TEST_SENTENCE_DIR, emo_folder)
    if os.path.exists(folder):
        wav_files.extend(list(Path(folder).glob("*.wav")))

log.log(f"Test sesleri: {len(wav_files)} dosya")

# ── Aşama 1: Öznitelikleri Çıkar (Global + Segment) ─────────────────────────
log.log("\n" + "="*70)
log.log("  AŞAMA 1 — Öznitelik Çıkarımı (VOSK Segmentation + IS10)")
log.log("="*70)

all_data = [] # List of {'true': label, 'global': np(4, models), 'segment_avg': np(4, models)}
errors = 0
t_start = time.time()

for idx, wav_path in enumerate(wav_files):
    true_label = wav_path.parent.name.lower()
    
    # 1. Global Özellik
    g_feat = extract_features_plain(str(wav_path))
    if g_feat is None:
        errors += 1; continue
        
    # 2. VOSK Segmentasyon
    segments_info = transcribe(str(wav_path), engine="vosk")
    if not segments_info:
        # Kelime bulamazsa tüm dosyayı segment say
        segments_info = [{'start': 0.0, 'end': librosa.get_duration(path=str(wav_path))}]
        
    # Segment Özellikleri
    y, sr = librosa.load(str(wav_path), sr=22050)
    seg_feats = []
    for seg in segments_info:
        start_idx = int(seg['start'] * sr)
        end_idx = int((seg['end'] + 0.1) * sr)
        seg_audio = y[start_idx:end_idx]
        if len(seg_audio) > int(0.1 * sr):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                sf.write(tf.name, seg_audio, sr)
                f = extract_features_plain(tf.name)
                if f is not None:
                    seg_feats.append(f)
            os.remove(tf.name)

    # Her model için Global ve Segment_Avg olasılıklarını hesapla
    sample_res = {'true': true_label, 'models': {}}
    
    for m_key, tools in models.items():
        m, s, e = tools['model'], tools['scaler'], tools['encoder']
        
        # Global Prob
        gf_scaled = s.transform(g_feat.reshape(1, -1))
        gp_raw = m.predict_proba(gf_scaled)[0]
        gp_mapped = np.zeros(4)
        for i, cls in enumerate(e.classes_):
            if cls.lower() in EMOTION_ORDER:
                gp_mapped[EMOTION_ORDER.index(cls.lower())] = gp_raw[i]
        
        # Segment Avg Prob
        if seg_feats:
            sp_sum = np.zeros(4)
            for sf_vec in seg_feats:
                sf_scaled = s.transform(sf_vec.reshape(1, -1))
                sp_raw = m.predict_proba(sf_scaled)[0]
                sp_m = np.zeros(4)
                for i, cls in enumerate(e.classes_):
                    if cls.lower() in EMOTION_ORDER:
                        sp_m[EMOTION_ORDER.index(cls.lower())] = sp_raw[i]
                sp_sum += sp_m
            sp_avg = sp_sum / len(seg_feats)
        else:
            sp_avg = gp_mapped.copy()
            
        sample_res['models'][m_key] = {'global': gp_mapped, 'segment': sp_avg}
        
    all_data.append(sample_res)
    if (idx + 1) % 20 == 0 or (idx + 1) == len(wav_files):
        log.log(f"  [{idx+1}/{len(wav_files)}] işlendi...")

elapsed = time.time() - t_start
log.log(f"\n  Tamamlandı — {len(all_data)} örnek, {elapsed:.1f}s")

# ── Aşama 2: Oranları Test Et ────────────────────────────────────────────────
log.log("\n" + "="*70)
log.log("  AŞAMA 2 — Global vs Segment Ağırlığı Testi")
log.log("  (G = Global Ağırlığı, S = 1-G Kelime Ağırlığı)")
log.log("="*70)
log.log(f"  {'Global (G)':<12} {'Segment (S)':<12} {'Accuracy':>10} {'F1-Score':>10}")
log.log("-"*50)

ratio_results = []
all_true = [d['true'] for d in all_data]

for g_ratio in GLOBAL_RATIOS:
    s_ratio = round(1.0 - g_ratio, 1)
    
    y_pred = []
    for sample in all_data:
        ensemble_probs = np.zeros(4)
        
        for m_key, weight in MODEL_WEIGHTS.items():
            m_res = sample['models'][m_key]
            # 60/40 yerine G/S kullan
            m_combined = g_ratio * m_res['global'] + s_ratio * m_res['segment']
            # Olasılıkları normalize et
            c_sum = np.sum(m_combined)
            if c_sum > 0: m_combined /= c_sum
            
            # Ensemble'a ekle (Optimal model ağırlığıyla)
            ensemble_probs += (m_combined * weight)
            
        final_idx = np.argmax(ensemble_probs)
        y_pred.append(EMOTION_ORDER[final_idx])
        
    acc = accuracy_score(all_true, y_pred) * 100
    f1  = f1_score(all_true, y_pred, average='weighted', labels=EMOTION_ORDER) * 100
    ratio_results.append({'g': g_ratio, 's': s_ratio, 'acc': acc, 'f1': f1, 'preds': y_pred})

# En iyi oranı bul
best_res = max(ratio_results, key=lambda x: x['acc'])

for res in ratio_results:
    marker = "  ◄ EN İYİ" if res['g'] == best_res['g'] else ""
    log.log(f"  %{res['g']*100:<10.0f} %{res['s']*100:<10.0f} {res['acc']:>9.2f}% {res['f1']:>9.2f}%{marker}")

log.log("\n" + "="*70)
log.log(f"  SONUÇ: En İyi Global Ağırlığı: %{best_res['g']*100:.0f}")
log.log(f"  Accuracy: %{best_res['acc']:.2f}")
log.log("="*70)
log.log(classification_report(all_true, best_res['preds'], labels=EMOTION_ORDER))

log.log(f"\nDetaylı rapor: {RESULTS_FILE}")
log.close()
