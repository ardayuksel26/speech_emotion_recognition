import os
import sys
import glob
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Path settings
BASE_DIR = r"c:\Users\ilhan\Desktop\SER_PROJECT"
BACKEND_DIR = os.path.join(BASE_DIR, "Backend")
MODELS_DIR = os.path.join(BASE_DIR, "Models")
DATASET_DIR = os.path.join(BASE_DIR, "Test", "sentencevoice_test")
OUTPUT_DIR = os.path.join(BASE_DIR, "Test", "Tests_Results_For_Train2-Mastermind", "RF_LightGBM_XGBoost_CatBoost_GradientB_Sentence-Test")

sys.path.append(BACKEND_DIR)
from preprocessing import extract_features

os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_CONFIG = {
    'CatBoost': {
        'model': os.path.join(MODELS_DIR, 'CatBoost', 'catboost_model.pkl'),
        'scaler': os.path.join(MODELS_DIR, 'CatBoost', 'scaler_cb.pkl'),
        'encoder': os.path.join(MODELS_DIR, 'CatBoost', 'label_encoder_cb.pkl')
    },
    'XGBoost': {
        'model': os.path.join(MODELS_DIR, 'XGBoost', 'xgboost_model.pkl'),
        'scaler': os.path.join(MODELS_DIR, 'XGBoost', 'scaler_xgb.pkl'),
        'encoder': os.path.join(MODELS_DIR, 'XGBoost', 'label_encoder_xgb.pkl')
    },
    'LightGBM': {
        'model': os.path.join(MODELS_DIR, 'LightGBM', 'lightgbm_model.pkl'),
        'scaler': os.path.join(MODELS_DIR, 'LightGBM', 'scaler_lgb.pkl'),
        'encoder': os.path.join(MODELS_DIR, 'LightGBM', 'label_encoder_lgb.pkl')
    },
    'RandomForest': {
        'model': os.path.join(MODELS_DIR, 'Random Forest', 'random_forest_model.pkl'),
        'scaler': os.path.join(MODELS_DIR, 'Random Forest', 'scaler_rf.pkl'),
        'encoder': os.path.join(MODELS_DIR, 'Random Forest', 'label_encoder_rf.pkl')
    },
    'GradientBoosting': {
        'model': os.path.join(MODELS_DIR, 'GradientBoosting', 'gradboost_model.pkl'),
        'scaler': os.path.join(MODELS_DIR, 'GradientBoosting', 'scaler_gb.pkl'),
        'encoder': os.path.join(MODELS_DIR, 'GradientBoosting', 'label_encoder_gb.pkl')
    }
}

print("=== TEMEL 5 MODEL DEĞERLENDİRME SİSTEMİ ===")

# Load Models
loaded_models = {}
for name, paths in MODEL_CONFIG.items():
    print(f"Loading {name}...")
    try:
        model = joblib.load(paths['model'])
        scaler = joblib.load(paths['scaler'])
        encoder = joblib.load(paths['encoder'])
        loaded_models[name] = {'model': model, 'scaler': scaler, 'encoder': encoder}
    except Exception as e:
        print(f"Failed to load {name}: {e}")

emotions = ['angry', 'calm', 'happy', 'sad']
true_labels = []
features_list = []

print(f"\nVeriseti Klasörü: {DATASET_DIR}")
print("Ses özellik çıkarımı başlatılıyor (extraction)... Lütfen bekleyin.")

# Özellik çıkarımı (extraction)
for emotion in emotions:
    folder = os.path.join(DATASET_DIR, emotion)
    if not os.path.isdir(folder): 
        print(f"Warning: {folder} bulunamadı.")
        continue
    
    wav_files = glob.glob(os.path.join(folder, "*.wav"))
    for wav_file in wav_files:
        features = extract_features(wav_file)
        if features is not None:
            features_list.append(features)
            true_labels.append(emotion)

features_np = np.array(features_list)
print(f"Toplam özellik çıkarılan dosya sayısı: {len(true_labels)}\n")

report_file = os.path.join(OUTPUT_DIR, "evaluation_report.txt")
with open(report_file, 'w', encoding='utf-8') as f:
    f.write("=== İLK NESİL 5 TEMEL MODEL: CÜMLE BAZLI TEST SONUÇLARI ===\n")
    f.write(f"Test Veriseti: {DATASET_DIR}\n")
    f.write(f"Toplam Test Örneği: {len(true_labels)}\n")
    f.write("="*60 + "\n\n")

for name, tools in loaded_models.items():
    print(f"[{name}] modeli test ediliyor...")
    model = tools['model']
    scaler = tools['scaler']
    encoder = tools['encoder']
    
    scaled_feats = scaler.transform(features_np)
    preds = model.predict(scaled_feats)
    
    # Boyut düzeltmeleri
    if isinstance(preds, list) or (isinstance(preds, np.ndarray) and len(preds.shape) > 1):
        preds = np.array(preds).flatten()
        
    try:
        # Int/Float ise inverse_transform yap
        if isinstance(preds[0], (int, np.integer, float, np.floating)):
            pred_labels = encoder.inverse_transform([int(p) for p in preds])
        else:
            # Zaten string geliyorsa
            pred_labels = preds
    except Exception as e:
        pred_labels = encoder.inverse_transform(preds)
        
    pred_labels = [str(p).lower() for p in pred_labels]
    
    # Neutral -> Calm eşlemesi (Gerekirse)
    pred_labels = ['calm' if p == 'neutral' else p for p in pred_labels]
    
    acc = accuracy_score(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels, labels=emotions, zero_division=0)
    cm = confusion_matrix(true_labels, pred_labels, labels=emotions)
    
    # Rapor Dosyasına Yaz
    with open(report_file, 'a', encoding='utf-8') as f:
        f.write(f"Model: {name}\n")
        f.write(f"Genel Doğruluk (Accuracy): %{acc * 100:.2f}\n")
        f.write("Detaylı Skorlar (Precision, Recall, F1-Score):\n")
        f.write(report + "\n")
        f.write("-" * 60 + "\n\n")
        
    # Karmaşıklık Matrisi (Confusion Matrix) Çizimi
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotions, yticklabels=emotions)
    plt.title(f'{name} Karmaşıklık Matrisi (Accuracy: %{acc*100:.2f})')
    plt.ylabel('Gerçek Duygu')
    plt.xlabel('Tahmin Edilen Duygu')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_confusion_matrix.png"))
    plt.close()

print(f"\nTest tamamlandı! Detaylı rapor ve grafikler şu klasöre kaydedildi:")
print(OUTPUT_DIR)
