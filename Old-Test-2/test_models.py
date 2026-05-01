"""
Test2/test_models.py
====================
Models_2 altındaki 5 modeli iki farklı yöntemle test eder:
  1. CSV  : All_Sounds/Test_CSV'den önceden çıkarılmış IS10 özellikleri
  2. Audio: All_Sounds/Test_Sounds WAV dosyalarından anlık IS10 çıkarma

Her model için accuracy, precision, recall, F1, confusion analizi ve
sınıf bazlı F1 tablosu üretir. Majority voting karşılaştırması yapar.
Sonuçlar Test2/Results/test_results.txt dosyasına kaydedilir.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import warnings
import opensmile
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)

warnings.filterwarnings('ignore')

# ── Yollar ────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(BASE_DIR, ".."))
TEST_SOUNDS  = os.path.join(PROJECT_ROOT, "All_Sounds", "Test_Sounds")
TEST_CSV_DIR = os.path.join(PROJECT_ROOT, "All_Sounds", "Test_CSV")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "Models_2")
RESULTS_DIR  = os.path.join(BASE_DIR, "Results")
os.makedirs(RESULTS_DIR, exist_ok=True)

EMOTIONS = ["angry", "calm", "happy", "sad"]

MODELS = [
    {
        "name":        "Random Forest",
        "subdir":      "RandomForest",
        "model_file":  "random_forest_model.pkl",
        "scaler_file": "scaler_rf.pkl",
        "le_file":     "label_encoder_rf.pkl",
    },
    {
        "name":        "LightGBM",
        "subdir":      "LightGBM",
        "model_file":  "lightgbm_model.pkl",
        "scaler_file": "scaler_lgbm.pkl",
        "le_file":     "label_encoder_lgbm.pkl",
    },
    {
        "name":        "XGBoost",
        "subdir":      "XGBoost",
        "model_file":  "xgboost_model.pkl",
        "scaler_file": "scaler_xgb.pkl",
        "le_file":     "label_encoder_xgb.pkl",
    },
    {
        "name":        "CatBoost",
        "subdir":      "CatBoost",
        "model_file":  "catboost_model.pkl",
        "scaler_file": "scaler_catboost.pkl",
        "le_file":     "label_encoder_catboost.pkl",
    },
    {
        "name":        "Gradient Boosting",
        "subdir":      "GradientBoosting",
        "model_file":  "gradient_boosting_model.pkl",
        "scaler_file": "scaler_gb.pkl",
        "le_file":     "label_encoder_gb.pkl",
    },
]


# ── Logger ────────────────────────────────────────────────────────────────────
class Logger:
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, "w", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ── Veri Yükleme ──────────────────────────────────────────────────────────────
def load_csv_data():
    """Test_CSV klasöründen önceden çıkarılmış IS10 özelliklerini yükler."""
    all_dfs = []
    for emotion in EMOTIONS:
        path = os.path.join(TEST_CSV_DIR, f"{emotion}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} bulunamadı. Önce extract_test_features.py çalıştırın.")
        df = pd.read_csv(path)
        df['label'] = emotion
        all_dfs.append(df)
    df = pd.concat(all_dfs, ignore_index=True)
    drop = ['label'] + [c for c in ['filename', 'name', 'path'] if c in df.columns]
    X = df.drop(columns=drop, errors='ignore').values.astype(np.float32)
    y = df['label'].values
    return X, y


def load_audio_data():
    """Test_Sounds WAV dosyalarından anlık IS10 özellik çıkarır."""
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.IS10,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    files = sorted([
        f for f in os.listdir(TEST_SOUNDS) if f.lower().endswith(".wav")
    ])
    X_list, y_list, err = [], [], 0
    for fname in tqdm(files, desc="  Ses → IS10", unit="ses"):
        emotion = fname.split("_")[0].lower()
        if emotion not in EMOTIONS:
            continue
        try:
            feat = smile.process_file(os.path.join(TEST_SOUNDS, fname))
            X_list.append(feat.iloc[0].values.astype(np.float32))
            y_list.append(emotion)
        except Exception as e:
            err += 1
    if err:
        print(f"  [UYARI] {err} dosya atlandı.")
    return np.array(X_list), np.array(y_list)


# ── Model Yükleme ─────────────────────────────────────────────────────────────
def load_models():
    loaded = {}
    for cfg in MODELS:
        d = os.path.join(MODELS_DIR, cfg["subdir"])
        try:
            loaded[cfg["name"]] = {
                "model":   joblib.load(os.path.join(d, cfg["model_file"])),
                "scaler":  joblib.load(os.path.join(d, cfg["scaler_file"])),
                "encoder": joblib.load(os.path.join(d, cfg["le_file"])),
                "cfg":     cfg,
            }
        except Exception as e:
            print(f"  [HATA] {cfg['name']} yüklenemedi: {e}")
    return loaded


# ── Değerlendirme ─────────────────────────────────────────────────────────────
def evaluate(tools, X_raw, y_str):
    model, scaler, encoder = tools["model"], tools["scaler"], tools["encoder"]

    classes = [c.lower() for c in encoder.classes_]
    y_idx   = np.array([EMOTIONS.index(e) for e in y_str])

    X = scaler.transform(X_raw)
    proba = model.predict_proba(X)

    # Encoder sırasını EMOTIONS sırasına hizala
    aligned = np.zeros((proba.shape[0], len(EMOTIONS)), dtype=np.float32)
    for i, emo in enumerate(EMOTIONS):
        if emo in classes:
            aligned[:, i] = proba[:, classes.index(emo)]

    preds = np.argmax(aligned, axis=1)

    # Confusion analizi
    confusion = {e: {t: 0 for t in EMOTIONS} for e in EMOTIONS}
    counts    = {e: 0 for e in EMOTIONS}
    for true_i, pred_i in zip(y_idx, preds):
        t, p = EMOTIONS[true_i], EMOTIONS[pred_i]
        confusion[t][p] += 1
        counts[t] += 1

    return {
        "probs":        aligned,
        "preds":        preds,
        "y_idx":        y_idx,
        "accuracy":     accuracy_score(y_idx, preds),
        "precision":    precision_score(y_idx, preds, average="macro", zero_division=0),
        "recall":       recall_score(y_idx, preds, average="macro", zero_division=0),
        "f1":           f1_score(y_idx, preds, average="macro", zero_division=0),
        "f1_per_class": f1_score(y_idx, preds, average=None,
                                 labels=list(range(len(EMOTIONS))), zero_division=0),
        "report":       classification_report(y_idx, preds,
                                              target_names=EMOTIONS, digits=4, zero_division=0),
        "confusion":    confusion,
        "counts":       counts,
    }


# ── Majority Voting ────────────────────────────────────────────────────────────
def majority_vote(all_results, weights, y_idx):
    N = list(all_results.values())[0]["probs"].shape[0]
    combined = np.zeros((N, len(EMOTIONS)), dtype=np.float64)
    for name, w in weights.items():
        if name in all_results:
            combined += w * all_results[name]["probs"]
    preds = np.argmax(combined, axis=1)

    confusion = {e: {t: 0 for t in EMOTIONS} for e in EMOTIONS}
    counts    = {e: 0 for e in EMOTIONS}
    for ti, pi in zip(y_idx, preds):
        t, p = EMOTIONS[ti], EMOTIONS[pi]
        confusion[t][p] += 1
        counts[t] += 1

    return {
        "preds":     preds,
        "accuracy":  accuracy_score(y_idx, preds),
        "precision": precision_score(y_idx, preds, average="macro", zero_division=0),
        "recall":    recall_score(y_idx, preds, average="macro", zero_division=0),
        "f1":        f1_score(y_idx, preds, average="macro", zero_division=0),
        "report":    classification_report(y_idx, preds,
                                           target_names=EMOTIONS, digits=4, zero_division=0),
        "confusion": confusion,
        "counts":    counts,
    }


# ── Yazdırma Yardımcıları ──────────────────────────────────────────────────────
W = 70

def sep(c="="):  print(c * W)
def title(t):    sep(); print(f"  {t}"); sep()


def print_model_table(results):
    print(f"  {'Model':<25} {'Accuracy':>9}  {'Precision':>9}  {'Recall':>9}  {'F1-Score':>9}")
    sep("-")
    for name, res in sorted(results.items(), key=lambda x: -x[1]["f1"]):
        print(f"  {name:<25} {res['accuracy']*100:8.2f}%  "
              f"{res['precision']*100:8.2f}%  "
              f"{res['recall']*100:8.2f}%  "
              f"{res['f1']*100:8.2f}%")


def print_confusion_detail(label, result):
    print(f"\n  [{label}] Duygu Bazlı Analiz:")
    sep("-")
    print(f"  {'Gerçek':<12} | {'Doğru':>12} | {'Yanlış':>12} | En Çok Karıştırılan")
    sep("-")
    for emo in EMOTIONS:
        total   = result["counts"][emo]
        correct = result["confusion"][emo][emo]
        wrong   = total - correct
        others  = {k: v for k, v in result["confusion"][emo].items() if k != emo}
        confused = "Yok"
        if wrong > 0:
            top = max(others, key=others.get)
            if others[top] > 0:
                confused = f"{top} ({others[top]})"
        pc = correct / total * 100 if total else 0
        pw = wrong   / total * 100 if total else 0
        print(f"  {emo.capitalize():<12} | "
              f"{correct:>3}/{total:<4} ({pc:5.1f}%) | "
              f"{wrong:>3}/{total:<4} ({pw:5.1f}%) | "
              f"{confused}")
    sep("-")


def print_f1_table(results):
    header = f"  {'Model':<25}" + "".join(f"  {e.capitalize():>8}" for e in EMOTIONS)
    print(header)
    sep("-")
    for name, res in sorted(results.items(), key=lambda x: -x[1]["f1"]):
        row = f"  {name:<25}"
        for i in range(len(EMOTIONS)):
            row += f"  {res['f1_per_class'][i]*100:7.2f}%"
        print(row)


def run_section(section_title, X, y, models):
    title(section_title)
    print(f"  Örnek sayısı: {len(X)}")
    for emo in EMOTIONS:
        print(f"    {emo.capitalize():<8}: {np.sum(y == emo)}")
    print()

    results = {}
    for name, tools in models.items():
        try:
            results[name] = evaluate(tools, X, y)
            print(f"  ✓ {name} tamamlandı — "
                  f"Accuracy: %{results[name]['accuracy']*100:.2f}")
        except Exception as e:
            print(f"  ✗ {name} HATA: {e}")

    # Bireysel performans tablosu
    sep()
    print(f"  {section_title} — Bireysel Model Performansı")
    sep()
    print_model_table(results)

    # Duygu bazlı confusion
    sep()
    print(f"  {section_title} — Duygu Bazlı Karışıklık Analizi")
    sep()
    for name, res in sorted(results.items(), key=lambda x: -x[1]["f1"]):
        print_confusion_detail(name, res)

    # Sınıf bazlı F1
    sep()
    print(f"  {section_title} — Sınıf Bazlı F1-Score")
    sep()
    print_f1_table(results)

    # Majority voting
    if results:
        accs = {n: r["accuracy"] for n, r in results.items()}
        y_idx = list(results.values())[0]["y_idx"]

        schemas = {
            "Eşit (1.0)":     {n: 1.0   for n in accs},
            "Doğrusal (acc)": {n: v     for n, v in accs.items()},
            "Kare (acc²)":    {n: v**2  for n, v in accs.items()},
            "Küp (acc³)":     {n: v**3  for n, v in accs.items()},
        }

        sep()
        print(f"  {section_title} — Majority Voting Karşılaştırması")
        sep()
        print(f"  {'Şema':<22} {'Accuracy':>9}  {'Precision':>9}  {'Recall':>9}  {'F1-Score':>9}")
        sep("-")

        best_name, best_acc, best_res = None, 0.0, None
        for sname, w in schemas.items():
            r = majority_vote(results, w, y_idx)
            marker = ""
            if r["accuracy"] > best_acc:
                best_acc, best_name, best_res = r["accuracy"], sname, r
                marker = "  ◄ EN İYİ"
            print(f"  {sname:<22} {r['accuracy']*100:8.2f}%  "
                  f"{r['precision']*100:8.2f}%  "
                  f"{r['recall']*100:8.2f}%  "
                  f"{r['f1']*100:8.2f}%{marker}")

        sep()
        print(f"  En İyi Şema: {best_name}")
        sep()
        print_confusion_detail(best_name, best_res)
        print()
        print(best_res["report"])

    return results


# ── Ana Akış ──────────────────────────────────────────────────────────────────
def main():
    log_path = os.path.join(RESULTS_DIR, "test_results.txt")
    sys.stdout = Logger(log_path)

    sep()
    print("  Models_2 Test Raporu — CSV ve Ham Ses Karşılaştırması")
    print("  Test verisi: All_Sounds/Test_Sounds")
    sep()

    # Modelleri yükle
    print("\nModeller yükleniyor...")
    models = load_models()
    if not models:
        print("Hiç model yüklenemedi.")
        return
    print(f"  {len(models)} model yüklendi: {', '.join(models.keys())}\n")

    # ── TEST 1: CSV ──────────────────────────────────────────────────────────
    try:
        print("CSV verileri yükleniyor...")
        X_csv, y_csv = load_csv_data()
        csv_results = run_section("TEST 1 — CSV (Önceden Çıkarılmış IS10)", X_csv, y_csv, models)
    except FileNotFoundError as e:
        print(f"\n[UYARI] CSV testi atlandı: {e}\n")
        csv_results = {}

    # ── TEST 2: Ham Ses ──────────────────────────────────────────────────────
    sep()
    print("TEST 2 — Ham Ses (Anlık IS10 Çıkarma)")
    sep()
    print("Test_Sounds WAV dosyalarından IS10 özellikleri çıkarılıyor...\n")
    X_audio, y_audio = load_audio_data()
    audio_results = run_section("TEST 2 — Ham Ses (Anlık IS10)", X_audio, y_audio, models)

    # ── KARŞILAŞTIRMA ────────────────────────────────────────────────────────
    if csv_results and audio_results:
        sep()
        print("  YÖNTEM KARŞILAŞTIRMASI — CSV vs Ham Ses")
        sep()
        print(f"  {'Model':<25} {'CSV Acc':>10}  {'Audio Acc':>10}  {'Fark':>8}")
        sep("-")
        for name in sorted(csv_results.keys(), key=lambda n: -csv_results[n]["accuracy"]):
            if name in audio_results:
                a_csv   = csv_results[name]["accuracy"] * 100
                a_audio = audio_results[name]["accuracy"] * 100
                diff    = a_audio - a_csv
                sign    = "+" if diff >= 0 else ""
                print(f"  {name:<25} {a_csv:9.2f}%  {a_audio:9.2f}%  {sign}{diff:6.2f}%")
        sep()

    print(f"\nSonuçlar kaydedildi: {log_path}")


if __name__ == "__main__":
    main()
