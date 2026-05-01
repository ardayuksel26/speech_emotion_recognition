import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

# TensorFlow imports for DL models
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    print("⚠️ TensorFlow not available. DL models (CNN, DNN) will be skipped.")
    TF_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, '../Models'))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../TurEV-DB/Extracted CSV"))

EMOTION_FILES = {"angry.csv": "angry", "calm.csv": "calm", "happy.csv": "happy", "sad.csv": "sad"}

MODEL_CONFIG = {
    'catboost': {
        'model': os.path.join(MODELS_DIR, 'CatBoost/catboost_model.pkl'),
        'scaler': os.path.join(MODELS_DIR, 'CatBoost/scaler_cb.pkl'),
        'encoder': os.path.join(MODELS_DIR, 'CatBoost/label_encoder_cb.pkl')
    },
    'catboost_robust': {
        'model': os.path.join(MODELS_DIR, 'CatBoost_Robust/catboost_robust.pkl'),
        'scaler': os.path.join(MODELS_DIR, 'CatBoost_Robust/scaler_robust.pkl'),
        'encoder': os.path.join(MODELS_DIR, 'CatBoost_Robust/label_encoder_robust.pkl')
    },
    'xgboost': {
        'model': os.path.join(MODELS_DIR, 'XGBoost/xgboost_model.pkl'),
        'scaler': os.path.join(MODELS_DIR, 'XGBoost/scaler_xgb.pkl'),
        'encoder': os.path.join(MODELS_DIR, 'XGBoost/label_encoder_xgb.pkl')
    },
    'xgboost_robust': {
        'model': os.path.join(MODELS_DIR, 'XGBoost_Robust/xgboost_robust.pkl'),
        'scaler': os.path.join(MODELS_DIR, 'XGBoost_Robust/scaler_robust.pkl'),
        'encoder': os.path.join(MODELS_DIR, 'XGBoost_Robust/label_encoder_robust.pkl')
    },
    'lightgbm': {
        'model': os.path.join(MODELS_DIR, 'LightGBM/lightgbm_model.pkl'),
        'scaler': os.path.join(MODELS_DIR, 'LightGBM/scaler_lgb.pkl'),
        'encoder': os.path.join(MODELS_DIR, 'LightGBM/label_encoder_lgb.pkl')
    },
    'lightgbm_robust': {
        'model': os.path.join(MODELS_DIR, 'LightGBM_Robust/lightgbm_robust.pkl'),
        'scaler': os.path.join(MODELS_DIR, 'LightGBM_Robust/scaler_robust.pkl'),
        'encoder': os.path.join(MODELS_DIR, 'LightGBM_Robust/label_encoder_robust.pkl')
    },
    'rf': {
        'model': os.path.join(MODELS_DIR, 'Random Forest/random_forest_model.pkl'),
        'scaler': os.path.join(MODELS_DIR, 'Random Forest/scaler_rf.pkl'),
        'encoder': os.path.join(MODELS_DIR, 'Random Forest/label_encoder_rf.pkl')
    },
    'rf_robust': {
        'model': os.path.join(MODELS_DIR, 'Random Forest_Robust/random_forest_robust.pkl'),
        'scaler': os.path.join(MODELS_DIR, 'Random Forest_Robust/scaler_robust.pkl'),
        'encoder': os.path.join(MODELS_DIR, 'Random Forest_Robust/label_encoder_robust.pkl')
    },
    'knn': {
        'model': os.path.join(MODELS_DIR, 'KNN/knn_model.pkl'),
        'scaler': os.path.join(MODELS_DIR, 'KNN/scaler_knn.pkl'),
        'encoder': os.path.join(MODELS_DIR, 'KNN/label_encoder_knn.pkl')
    },
    'knn_robust': {
        'model': os.path.join(MODELS_DIR, 'KNN_Robust/knn_robust.pkl'),
        'scaler': os.path.join(MODELS_DIR, 'KNN_Robust/scaler_robust.pkl'),
        'encoder': os.path.join(MODELS_DIR, 'KNN_Robust/label_encoder_robust.pkl')
    },
    'svm': {
        'model': os.path.join(MODELS_DIR, 'SVM/svm_model.pkl'),
        'scaler': os.path.join(MODELS_DIR, 'SVM/scaler_svm.pkl'),
        'encoder': os.path.join(MODELS_DIR, 'SVM/label_encoder_svm.pkl')
    },
    'svm_robust': {
        'model': os.path.join(MODELS_DIR, 'SVM_Robust/svm_robust.pkl'),
        'scaler': os.path.join(MODELS_DIR, 'SVM_Robust/scaler_robust.pkl'),
        'encoder': os.path.join(MODELS_DIR, 'SVM_Robust/label_encoder_robust.pkl')
    },
    'mlp': {
        'model': os.path.join(MODELS_DIR, 'MLP/mlp_model.pkl'),
        'scaler': os.path.join(MODELS_DIR, 'MLP/scaler_mlp.pkl'),
        'encoder': os.path.join(MODELS_DIR, 'MLP/label_encoder_mlp.pkl')
    },
    'mlp_robust': {
        'model': os.path.join(MODELS_DIR, 'MLP_Robust/mlp_robust.pkl'),
        'scaler': os.path.join(MODELS_DIR, 'MLP_Robust/scaler_robust.pkl'),
        'encoder': os.path.join(MODELS_DIR, 'MLP_Robust/label_encoder_robust.pkl')
    },
    'gradient_boosting': {
        'model': os.path.join(MODELS_DIR, 'GradientBoosting/gradboost_model.pkl'),
        'scaler': os.path.join(MODELS_DIR, 'GradientBoosting/scaler_gb.pkl'),
        'encoder': os.path.join(MODELS_DIR, 'GradientBoosting/label_encoder_gb.pkl')
    },
    'gradient_boosting_robust': {
        'model': os.path.join(MODELS_DIR, 'GradientBoosting_Robust/gradboost_robust.pkl'),
        'scaler': os.path.join(MODELS_DIR, 'GradientBoosting_Robust/scaler_robust.pkl'),
        'encoder': os.path.join(MODELS_DIR, 'GradientBoosting_Robust/label_encoder_robust.pkl')
    },
    'dnn': {
        'model': os.path.join(MODELS_DIR, 'DNN/dnn_model.h5'),
        'scaler': os.path.join(MODELS_DIR, 'DNN/scaler_dnn.pkl'),
        'encoder': os.path.join(MODELS_DIR, 'DNN/label_encoder_dnn.pkl')
    },
    'dnn_robust': {
        'model': os.path.join(MODELS_DIR, 'DNN_Robust/dnn_robust.h5'),
        'scaler': os.path.join(MODELS_DIR, 'DNN_Robust/scaler_robust.pkl'),
        'encoder': os.path.join(MODELS_DIR, 'DNN_Robust/label_encoder_robust.pkl')
    },
    'cnn1d': {
        'model': os.path.join(MODELS_DIR, 'CNN1D/cnn1d_model.h5'),
        'scaler': os.path.join(MODELS_DIR, 'CNN1D/scaler_cnn1d.pkl'),
        'encoder': os.path.join(MODELS_DIR, 'CNN1D/label_encoder_cnn1d.pkl')
    },
    'cnn1d_robust': {
        'model': os.path.join(MODELS_DIR, 'CNN1D_Robust/cnn1d_robust.h5'),
        'scaler': os.path.join(MODELS_DIR, 'CNN1D_Robust/scaler_robust.pkl'),
        'encoder': os.path.join(MODELS_DIR, 'CNN1D_Robust/label_encoder_robust.pkl')
    }
}

def load_data():
    all_dfs = []
    if not os.path.exists(DATA_DIR):
        print(f"HATA: Veri klasörü bulunamadı: {DATA_DIR}")
        return None

    for fname, label in EMOTION_FILES.items():
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['true_label'] = label
            all_dfs.append(df)
            
    if not all_dfs: return None
    df = pd.concat(all_dfs, ignore_index=True)
    return df

def run_evaluation():
    print("="*80)
    print(" 📊 AKADEMİK VALIDATION ANALİZİ (GÖRÜLMEMİŞ GÜRÜLTÜLÜ VERİ DEĞERLENDİRMESİ) 📊")
    print("="*80)
    
    df = load_data()
    if df is None: return

    y_true_str = df['true_label'].to_numpy()
    drop_cols = ['true_label', 'filename', 'name', 'path', 'dosya_adi']
    X_raw = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore').to_numpy()
    
    # 1. ORİJİNAL VERİYİ EĞİTİM ESNASINDAKİ GİBİ BÖL (Aynı Random State 42 Kuralını Kullanıyoruz)
    # train_robust_models.py kısmındaki bölme mantığının aynısını taklit eder.
    # Bu sayede sadece modelin %20'lik "Görmediği" Test kısmını alırız
    from sklearn.model_selection import train_test_split
    _, X_raw_test_split, _, y_str_test_split = train_test_split(X_raw, y_true_str, test_size=0.2, random_state=42, stratify=None)
    
    # 2. TEST VERİSİNE İNANILMAZ SERT DIŞ GÜRÜLTÜ EKLE (Robust'u Zorlamak İçin)
    print("\n[+] Validation test verisine %20 seviyesinde Yapay Gürültü (Gaussian +0.15) ve Hiss ekleniyor...")
    X_test_noisy = X_raw_test_split + np.random.normal(0, 0.15, X_raw_test_split.shape)
    
    # Sadece Robust modelleri kullanıp grafiği oluşturacağız
    unique_emotions = sorted(list(set(y_str_test_split)))
    model_f1_scores = {emotion: [] for emotion in unique_emotions}
    model_names = []
    
    results_text = []
    results_text.append("="*80)
    results_text.append(" AKADEMİK ROBUST (GÜRÜLTÜLÜ) VALIDASYON RAPORU ")
    results_text.append("="*80)
    results_text.append("Not: Test işlemi modellerin daha önce *hiç görmediği* Validation (%20) verisi üzerine")
    results_text.append("ekstra sentetik +0.15 seviyesinde Gaussian gürültü bindirilerek yapılmıştır.\n")
    
    for key, paths in MODEL_CONFIG.items():
        if not os.path.exists(paths['model']):
            continue
            
        print(f"\nModel test ediliyor: {key.upper()}...")
        try:
            if paths['model'].endswith('.h5'):
                if not TF_AVAILABLE: continue
                model = load_model(paths['model'])
            else:
                model = joblib.load(paths['model'])
                
            scaler = joblib.load(paths['scaler'])
            encoder = joblib.load(paths['encoder'])
            
            # Gürültülü test verisini ölçeklendir
            X_scaled = scaler.transform(X_test_noisy)
            
            if key in ('cnn1d_robust', 'cnn1d'):
                X_scaled_3d = np.expand_dims(X_scaled, axis=2)
                probs = model.predict(X_scaled_3d, verbose=0)
                y_pred_idx = np.argmax(probs, axis=1)
            elif key in ('dnn_robust', 'dnn'):
                probs = model.predict(X_scaled, verbose=0)
                y_pred_idx = np.argmax(probs, axis=1)
            else:
                y_pred_idx = model.predict(X_scaled)
                if len(y_pred_idx.shape) > 1 and y_pred_idx.shape[1] == 1:
                    y_pred_idx = y_pred_idx.flatten()

            # Normal DL (DNN ve CNN1D) modellerinin encoder'i Binarizer (OHE) olabilir bu yüzden ters çevirmede shape hatası verebiliyor, 
            # try-except ile veya idx'ten direk array bulma ile çözelim:
            try:
                if (key in ('dnn', 'cnn1d')) and len(encoder.classes_.shape) > 1:
                    # Eger OneHotEncoder ise inverse transform shape ister
                    y_pred_str = encoder.inverse_transform(y_pred_idx.reshape(-1, 1))
                    y_pred_str = y_pred_str.flatten()
                else:
                    y_pred_str = encoder.inverse_transform(y_pred_idx.astype(int))
            except ValueError:
                # Olası bir [0, 1] OneHot array dönme sorununda
                categorical_preds = np.zeros((len(y_pred_idx), len(encoder.classes_)))
                categorical_preds[np.arange(len(y_pred_idx)), y_pred_idx] = 1
                y_pred_str = encoder.inverse_transform(categorical_preds)
                y_pred_str = y_pred_str.flatten()
            except BaseException:
                 y_pred_str = encoder.inverse_transform(y_pred_idx.astype(int))
                 
            if len(y_pred_str.shape) > 1:
                 y_pred_str = y_pred_str.flatten()
            y_pred_str = [yp.lower() if yp.lower() != 'neutral' else 'calm' for yp in y_pred_str]
            
            report = classification_report(y_str_test_split, y_pred_str, target_names=unique_emotions, output_dict=True, zero_division=0)
            
            model_names.append(key.upper().replace('_ROBUST', ''))
            
            results_text.append(f"\n{'='*40}")
            results_text.append(f" MODEL: {key.upper()}")
            results_text.append(f"{'='*40}")
            results_text.append(f"{'Duygu':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
            results_text.append("-" * 50)
            
            for emotion in unique_emotions:
                if emotion in report:
                    metrics = report[emotion]
                    p = metrics['precision']
                    r = metrics['recall']
                    f = metrics['f1-score']
                    model_f1_scores[emotion].append(f)
                    results_text.append(f"{emotion.capitalize():<10} | {p:.4f}     | {r:.4f}   | {f:.4f}")
                else:
                    model_f1_scores[emotion].append(0.0)
                    results_text.append(f"{emotion.capitalize():<10} | 0.0000     | 0.0000   | 0.0000")
            
            results_text.append("-" * 50)
            results_text.append(f"Accuracy : {report['accuracy']:.4f}")
            results_text.append(f"Macro F1 : {report['macro avg']['f1-score']:.4f}")
            results_text.append("")

        except Exception as e:
            print(f"HATA: {key.upper()} test edilirken sorun oluştu - {str(e)}")
            continue

    txt_path = os.path.join(BASE_DIR, "validation_robust_metrics.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(results_text))
    print(f"\n✅ Metin raporu kaydedildi: {txt_path}")

    if not model_names: return

    x = np.arange(len(model_names))
    width = 0.2
    
    plt.figure(figsize=(18, 8))
    
    colors = ['#ff595e', '#1982c4', '#8ac926', '#6a4c93']
    for i, (emotion, color) in enumerate(zip(unique_emotions, colors)):
        offset = (i - 1.5) * width
        plt.bar(x + offset, model_f1_scores[emotion], width, label=emotion.capitalize(), color=color, edgecolor='black', alpha=0.8)

    plt.xlabel('Modeller (Robust Sürümler)', fontweight='bold', fontsize=12)
    plt.ylabel('F1 Score', fontweight='bold', fontsize=12)
    plt.title('Görülmemiş Veri (Validation + Zorlu Gürültü) Üzerinde F1 Skorları', fontweight='bold', fontsize=16)
    plt.xticks(x, model_names, rotation=45, ha='right', fontsize=10)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.legend(title='Duygular', bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    img_path = os.path.join(BASE_DIR, "validation_robust_f1_scores.png")
    plt.savefig(img_path, dpi=300)
    print(f"✅ Görsel rapor kaydedildi: {img_path}")
    print("\nRobust Kelime Testi Tamamlandı!")

if __name__ == "__main__":
    run_evaluation()
