"""
Majority Voting Ağırlık Optimizasyonu — Ses Dosyası Bazlı
===========================================================
TurEV-DB Sound Source klasöründeki WAV dosyalarından özellik çıkarır,
her modeli ayrı ayrı değerlendirir ve doğruluk bazlı ağırlıkları hesaplar.

Metrikler: Accuracy, Precision, Recall, f1-Score (sınıf bazlı + macro)

İlk çalıştırmada özellikler çıkarılır ve cache'lenir (features_cache.npz).
Sonraki çalıştırmalarda cache'den yüklenir — çok daha hızlı.

Kullanım:
    python optimize_weights.py
    python optimize_weights.py --no-cache   (cache'i yoksay, yeniden çıkar)
"""

import os
import sys
import argparse
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# ── Logger Yapılandırması ──────────────────────────────────────────────────
class Logger(object):
    """Hem terminale hem de dosyaya yazdırmak için yardımcı sınıf."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)

# ── Yollar ───────────────────────────────────────────────────────────────────
ROOT        = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
AUDIO_DIR   = os.path.join(ROOT, 'TurEV-DB', 'Sound Source')
MODELS_DIR  = os.path.join(ROOT, 'Models')
CACHE_FILE  = os.path.join(os.path.dirname(__file__), 'features_cache.npz')

sys.path.insert(0, os.path.join(ROOT, 'Backend'))
from preprocessing import extract_features

try:
    from tensorflow.keras.models import load_model as keras_load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow bulunamadı — DNN ve CNN1D atlanacak.")

# ── Sabitler ─────────────────────────────────────────────────────────────────
EMOTIONS = ['angry', 'calm', 'happy', 'sad']

EMOTION_FOLDERS = {
    'angry': 'Angry',
    'calm':  'Calm',
    'happy': 'Happy',
    'sad':   'Sad',
}

MODEL_CONFIG = {
    'rf':                ('Random Forest/random_forest_model.pkl', 'Random Forest/scaler_rf.pkl',       'Random Forest/label_encoder_rf.pkl'),
    'xgboost':           ('XGBoost/xgboost_model.pkl',             'XGBoost/scaler_xgb.pkl',            'XGBoost/label_encoder_xgb.pkl'),
    'lightgbm':          ('LightGBM/lightgbm_model.pkl',           'LightGBM/scaler_lgb.pkl',           'LightGBM/label_encoder_lgb.pkl'),
    'catboost':          ('CatBoost/catboost_model.pkl',           'CatBoost/scaler_cb.pkl',            'CatBoost/label_encoder_cb.pkl'),
}


# ── 1. Özellik Çıkarımı / Cache ───────────────────────────────────────────────
def extract_all_features(use_cache=True):
    """
    Tüm WAV dosyalarından özellik çıkar.
    use_cache=True ise önceki çalıştırmanın sonuçları varsa onları kullan.
    """
    if use_cache and os.path.exists(CACHE_FILE):
        print(f"📦 Cache bulundu, yükleniyor: {CACHE_FILE}")
        data = np.load(CACHE_FILE, allow_pickle=True)
        X = data['X']
        y = data['y'].tolist()
        names = data['names'].tolist() if 'names' in data else []
        
        # Eğer isimler eksikse veya sayı tutmuyorsa (eski cache formatı)
        if len(names) != len(X):
            print(f"  ⚠️  Uyarı: Cache'de dosya isimleri eksik veya hatalı ({len(names)} vs {len(X)}).")
            print("     Dosya bazlı detaylı rapor için --no-cache ile çalıştırın.")
            names = [f"unknown_{i}.wav" for i in range(len(X))]

        print(f"✅ Cache'den yüklendi: {X.shape[0]} örnek")
        return X, y, names

    print("🎙️  Ses dosyalarından özellik çıkarılıyor (bu işlem birkaç dakika sürebilir)...")
    print("   Sonraki çalıştırmalar cache sayesinde anında başlayacak.\n")

    X_list, y_list, name_list = [], [], []
    skipped = 0

    for emotion, folder in EMOTION_FOLDERS.items():
        folder_path = os.path.join(AUDIO_DIR, folder)
        wav_files   = sorted([f for f in os.listdir(folder_path) if f.endswith('.wav')])
        total       = len(wav_files)
        print(f"  [{emotion.upper()}] {total} dosya işleniyor...")

        for i, fname in enumerate(wav_files, 1):
            fpath    = os.path.join(folder_path, fname)
            features = extract_features(fpath)

            if features is None:
                print(f"    ⚠️  Atlandı: {fname}")
                skipped += 1
                continue

            X_list.append(features)
            y_list.append(emotion)
            name_list.append(fname)

            if i % 50 == 0 or i == total:
                print(f"    {i}/{total} tamamlandı", end='\r')

        print(f"    {total}/{total} tamamlandı ✓")

    X = np.array(X_list, dtype=np.float32)
    y = y_list

    np.savez(CACHE_FILE, X=X, y=np.array(y), names=np.array(name_list))
    print(f"\n✅ Toplam {X.shape[0]} örnek çıkarıldı. ({skipped} dosya atlandı)")
    print(f"💾 Cache kaydedildi: {CACHE_FILE}")

    return X, y, name_list


# ── 2. Model Yükleme ──────────────────────────────────────────────────────────
def load_models():
    models = {}
    for key, (mp, sp, ep) in MODEL_CONFIG.items():
        m_path = os.path.join(MODELS_DIR, mp)
        if not os.path.exists(m_path):
            print(f"  ⚠️  [{key}] bulunamadı, atlanıyor.")
            continue
        try:
            obj = keras_load_model(m_path) if m_path.endswith('.h5') else joblib.load(m_path)
            if m_path.endswith('.h5') and not TF_AVAILABLE:
                continue
            models[key] = {
                'model':   obj,
                'scaler':  joblib.load(os.path.join(MODELS_DIR, sp)),
                'encoder': joblib.load(os.path.join(MODELS_DIR, ep)),
            }
        except Exception as ex:
            print(f"  ❌ [{key}] yüklenemedi: {ex}")
    return models


# ── 3. Tahmin & Metrikler ─────────────────────────────────────────────────────
def evaluate_model(key, tools, X_test, y_idx, names=None):
    """Tek bir modeli değerlendir, tüm metrikleri döndür."""
    model, scaler, encoder = tools['model'], tools['scaler'], tools['encoder']

    Xs = scaler.transform(X_test)

    if key == 'cnn1d':
        raw = model.predict(np.expand_dims(Xs, axis=2), verbose=0)
    elif key == 'dnn':
        raw = model.predict(Xs, verbose=0)
    else:
        raw = model.predict_proba(Xs)

    # Encoder sırasını EMOTIONS sırasına hizala
    classes = [c.lower() for c in encoder.classes_]
    aligned = np.zeros((raw.shape[0], len(EMOTIONS)), dtype=np.float32)
    for i, emo in enumerate(EMOTIONS):
        if emo in classes:
            aligned[:, i] = raw[:, classes.index(emo)]

    preds = np.argmax(aligned, axis=1)

    # Karışıklık Analizi Hesaplama
    confusion = {e: {target: 0 for target in EMOTIONS} for e in EMOTIONS}
    counts = {e: 0 for e in EMOTIONS}
    
    for true_i, pred_i in zip(y_idx, preds):
        true_emo = EMOTIONS[true_i]
        pred_emo = EMOTIONS[pred_i]
        confusion[true_emo][pred_emo] += 1
        counts[true_emo] += 1

    if names is not None:
        print(f"\n🔍 [{key.upper()}] Detaylı Test Sonuçları:")
        print(f"{'Dosya Adı':<45} | {'Gerçek':<10} | {'Tahmin':<10} | {'Durum'}")
        print("-" * 85)
        for i in range(len(names)):
            true_emo = EMOTIONS[y_idx[i]]
            pred_emo = EMOTIONS[preds[i]]
            status = "✅" if true_emo == pred_emo else "❌"
            print(f"{names[i]:<45} | {true_emo:<10} | {pred_emo:<10} | {status}")
        print("-" * 85)

    return {
        'probs':     aligned,
        'preds':     preds,
        'accuracy':  accuracy_score(y_idx, preds),
        'precision': precision_score(y_idx, preds, average='macro', zero_division=0),
        'recall':    recall_score(y_idx, preds, average='macro', zero_division=0),
        'f1':        f1_score(y_idx, preds, average='macro', zero_division=0),
        'f1_per_class': f1_score(y_idx, preds, average=None, labels=list(range(len(EMOTIONS))), zero_division=0),
        'report':    classification_report(y_idx, preds, target_names=EMOTIONS, digits=4, zero_division=0),
        'confusion': confusion,
        'counts':    counts
    }


# ── 4. Ağırlıklı Oylama ───────────────────────────────────────────────────────
def weighted_vote(model_results, weights, y_idx):
    N        = list(model_results.values())[0]['probs'].shape[0]
    combined = np.zeros((N, len(EMOTIONS)), dtype=np.float64)
    for key, w in weights.items():
        if key in model_results:
            combined += w * model_results[key]['probs']
    preds = np.argmax(combined, axis=1)

    # Karışıklık Analizi
    confusion = {e: {target: 0 for target in EMOTIONS} for e in EMOTIONS}
    counts = {e: 0 for e in EMOTIONS}
    for true_i, pred_i in zip(y_idx, preds):
        true_emo = EMOTIONS[true_i]
        pred_emo = EMOTIONS[pred_i]
        confusion[true_emo][pred_emo] += 1
        counts[true_emo] += 1

    return {
        'preds':     preds,
        'accuracy':  accuracy_score(y_idx, preds),
        'precision': precision_score(y_idx, preds, average='macro', zero_division=0),
        'recall':    recall_score(y_idx, preds, average='macro', zero_division=0),
        'f1':        f1_score(y_idx, preds, average='macro', zero_division=0),
        'report':    classification_report(y_idx, preds, target_names=EMOTIONS, digits=4, zero_division=0),
        'confusion': confusion,
        'counts':    counts
    }


# ── 5. Yazdırma Yardımcıları ──────────────────────────────────────────────────
def print_separator(char='=', width=70):
    print(char * width)

def print_model_row(name, acc, prec, rec, f1):
    print(f"  {name:<25} {acc*100:6.2f}%  {prec*100:6.2f}%  {rec*100:6.2f}%  {f1*100:6.2f}%")

def print_confusion_detail(key, result):
    print(f"\n📈 [{key.upper()}] Sınıflandırma Detayları:")
    print("-" * 75)
    print(f"  {'Gerçek Sınıf':<15} | {'Doğru':<10} | {'Yanlış':<10} | {'En Çok Karıştırılan'}")
    print("-" * 75)
    
    confusion = result['confusion']
    counts    = result['counts']
    
    for emo in EMOTIONS:
        total = counts[emo]
        correct = confusion[emo][emo]
        wrong = total - correct
        
        # Karıştırılanlar listesi (doğru olan hariç)
        others = {k: v for k, v in confusion[emo].items() if k != emo}
        confused_with = "Yok"
        if wrong > 0:
            top_wrong = max(others, key=others.get)
            if others[top_wrong] > 0:
                confused_with = f"{top_wrong} ({others[top_wrong]})"
        
        perc_correct = (correct / total * 100) if total > 0 else 0
        perc_wrong   = (wrong / total * 100) if total > 0 else 0
        
        row_str = f"  {emo.capitalize():<15} | {correct:>3}/{total:<4} ({perc_correct:5.1f}%) | {wrong:>3}/{total:<4} ({perc_wrong:5.1f}%) | {confused_with}"
        print(row_str)
    print("-" * 75)


# ── Ana Akış ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Log dosyasını ayarla
    log_path = os.path.join(os.path.dirname(__file__), 'optimization_results.txt')
    sys.stdout = Logger(log_path)

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cache', action='store_true', help='Cache\'i yoksay, yeniden çıkar')
    args = parser.parse_args()

    print_separator()
    print("   Majority Voting — Ses Dosyası Bazlı Model Değerlendirmesi")
    print_separator()

    # 1. Özellikler
    X, y, names = extract_all_features(use_cache=not args.no_cache)

    y_idx = np.array([EMOTIONS.index(e) for e in y])

    # Tüm veriyi değerlendirme seti olarak kullan
    X_test, y_test_idx, names_test = X, y_idx, names
    
    print(f"\n📂 Değerlendirme seti: {X_test.shape[0]} örnek (Tüm dosyalar)")
    for i, e in enumerate(EMOTIONS):
        print(f"   {e}: {np.sum(y_test_idx == i)} örnek")

    # 2. Modeller
    print("\n🔃 Modeller yükleniyor...")
    models = load_models()
    if not models:
        print("❌ Hiç model yüklenemedi.")
        sys.exit(1)
    print(f"   {len(models)} model yüklendi.")

    # 3. Her modeli değerlendir
    print("\n📡 Modeller değerlendiriliyor...")
    results = {}
    for key, tools in models.items():
        try:
            # Burası detaylı yazdırmayı tetikler
            results[key] = evaluate_model(key, tools, X_test, y_test_idx, names_test)
            print(f"   ✅ [{key}] tamamlandı.")
        except Exception as ex:
            print(f"   ❌ [{key}] hata: {ex}")

    # 4. Bireysel model tablosu
    print("\n")
    print_separator()
    print("📊 Her Modelin Bireysel Performansı (Test Seti)")
    print_separator()
    print(f"  {'Model':<25} {'Accuracy':>9}  {'Precision':>9}  {'Recall':>9}  {'F1-Score':>9}")
    print_separator('-')
    for key, res in sorted(results.items(), key=lambda x: -x[1]['f1']):
        print_model_row(key, res['accuracy'], res['precision'], res['recall'], res['f1'])

    # 5. Her model için detaylı analiz
    print("\n")
    print_separator()
    print("🔬 Modeller İçin Duygu Bazlı Karışıklık Analizi")
    print_separator()
    for key, res in sorted(results.items(), key=lambda x: -x[1]['f1']):
        print_confusion_detail(key, res)

    # 6. Sınıf bazlı F1 tablosu
    print("\n")
    print_separator()
    print("📊 Sınıf Bazlı F1-Score")
    print_separator()
    header = f"  {'Model':<25}" + "".join(f"  {e.capitalize():>8}" for e in EMOTIONS)
    print(header)
    print_separator('-')
    for key, res in sorted(results.items(), key=lambda x: -x[1]['f1']):
        row = f"  {key:<25}"
        for i in range(len(EMOTIONS)):
            row += f"  {res['f1_per_class'][i]*100:7.2f}%"
        print(row)

    # 7. Ağırlık şemalarını karşılaştır
    accuracies = {k: v['accuracy'] for k, v in results.items()}

    schemas = {
        "Eşit (1.0)":     {k: 1.0    for k in accuracies},
        "Doğrusal (acc)": {k: v      for k, v in accuracies.items()},
        "Kare (acc²)":    {k: v**2   for k, v in accuracies.items()},
        "Küp (acc³)":     {k: v**3   for k, v in accuracies.items()},
    }

    print("\n")
    print_separator()
    print("⚖️  Majority Voting — Ağırlık Şeması Karşılaştırması")
    print_separator()
    print(f"  {'Şema':<22} {'Accuracy':>9}  {'Precision':>9}  {'Recall':>9}  {'F1-Score':>9}")
    print_separator('-')

    best_name, best_acc, best_weights, best_result = None, 0.0, None, None
    schema_results = {}

    for name, w in schemas.items():
        r = weighted_vote(results, w, y_test_idx)
        schema_results[name] = r
        marker = ""
        if r['accuracy'] > best_acc:
            best_acc, best_name, best_weights, best_result = r['accuracy'], name, w, r
            marker = "  ◄ EN İYİ"
        print(f"  {name:<22} {r['accuracy']*100:6.2f}%  {r['precision']*100:6.2f}%  {r['recall']*100:6.2f}%  {r['f1']*100:6.2f}%{marker}")

    # 8. En iyi şema detayları
    print("\n")
    print_separator()
    print(f"🏆  En İyi Şema: {best_name}")
    print_separator()
    print_confusion_detail(best_name, best_result)
    print("\n" + best_result['report'])

    # 9. Normalize ağırlıklar
    max_w   = max(best_weights.values())
    norm_w  = {k: v / max_w for k, v in best_weights.items()}

    print_separator()
    print(f"🔧 '{best_name}' Şeması — Normalize Ağırlıklar (max=1.00):")
    print_separator('-')
    for key, nw in sorted(norm_w.items(), key=lambda x: -x[1]):
        bar = "█" * int(nw * 25)
        print(f"  {key:<25}  {nw:.4f}  {bar}")

    # 10. majority_voting.py çıktısı
    print("\n")
    print_separator()
    print("📋 majority_voting.py için MODEL_WEIGHTS (kopyala-yapıştır):")
    print_separator('-')
    print("MODEL_WEIGHTS = {")
    for key, nw in sorted(norm_w.items(), key=lambda x: -x[1]):
        pad = " " * (25 - len(f"'{key}'"))
        print(f"    '{key}':{pad}{nw:.4f},")
    print("}")
    print_separator()
