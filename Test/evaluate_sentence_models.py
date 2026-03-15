import os
import sys
import glob
import random
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import soundfile as sf
import librosa
import noisereduce as nr
import opensmile
from sklearn.metrics import classification_report

# DL Modelleri için TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    print("⚠️ TensorFlow bulunamadı. DL Modelleri (CNN, DNN) atlanacak.")
    TF_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

# --- YOL AYARLARI ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, '../Models'))
SOUND_DIR = os.path.abspath(os.path.join(BASE_DIR, "../TurEV-DB/Sound Source"))

# --- CONF AYARLARI ---
EMOTION_FOLDERS = {"angry": "Angry", "calm": "Calm", "happy": "Happy", "sad": "Sad"}
NUM_SENTENCES_PER_EMOTION = 100 # Toplam 400 test cümlesi
SAMPLE_RATE = 16000
MIN_LEN_SEC = 3.0
MAX_LEN_SEC = 5.0

print(f"\n🎧 OpenSMILE motoru başlatılıyor...")
try:
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.IS10,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    print("✅ OpenSMILE başarıyla yüklendi!")
except Exception as e:
    print(f"❌ OpenSMILE yüklenemedi: {e}")
    sys.exit(1)

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

def create_synthetic_sentence(files, sr=SAMPLE_RATE, min_len=MIN_LEN_SEC, max_len=MAX_LEN_SEC):
    """
    Kelimeleri rastgele birleştirip gerçekçi dünya şartlarına (seviye değişimi, dış ses, kesilme vb.)
    sokar ve rastgele bir cümle/konuşma sekansı (3-5sn) oluşturur.
    """
    num_words = random.randint(2, 5)
    chosen = random.sample(files, min(num_words, len(files)))
    
    sentence_audio = []
    
    # Başlangıçta sessizlik (0.1 - 0.4s)
    sentence_audio.append(np.zeros(int(random.uniform(0.1, 0.4) * sr)))
    
    for f in chosen:
        try:
            y, _ = librosa.load(f, sr=sr)
            y_trimmed, _ = librosa.effects.trim(y, top_db=25)
            sentence_audio.append(y_trimmed)
        except Exception:
            continue
            
        gap_len = random.uniform(0.1, 0.6)
        sentence_audio.append(np.zeros(int(gap_len * sr)))
        
    if not sentence_audio:
        return None
        
    final_audio = np.concatenate(sentence_audio)
    duration = len(final_audio) / sr
    
    # Minimum Süre Tamamlama
    if duration < min_len:
        pad_len = int((min_len - duration) * sr)
        final_audio = np.concatenate((final_audio, np.zeros(pad_len)))
        
    # Maksimum Süre Kesme
    if (len(final_audio) / sr) > max_len:
        final_audio = final_audio[:int(max_len * sr)]
        
    # --- RASTGELE ROBUST (DIŞ DÜNYA) EFEKTLERİ ---
    
    # 1. Volume Scaling (Mikrofondan uzaklaşma/yakınlaşma simülasyonu)
    vol_scale = random.uniform(0.3, 1.5)
    final_audio = final_audio * vol_scale
    
    # 2. Gaussian Gürültü (Arkaplan uğultusu/rüzgar) - %70 İhtimalle
    if random.random() < 0.70:
        noise_amp = random.uniform(0.005, 0.04)
        noise = np.random.normal(0, noise_amp, len(final_audio))
        final_audio = final_audio + noise
        
    # 3. Kısa Süreli Bağlantı Kesilmesi/Mikrofon Temassızlığı - %30 İhtimalle
    if random.random() < 0.30:
        drop_dur = int(sr * random.uniform(0.1, 0.4))
        drop_start = random.randint(0, max(0, len(final_audio) - drop_dur))
        final_audio[drop_start:drop_start+drop_dur] = 0.0

    # 4. Normalizasyon ve Clipping Önleme
    max_abs = np.max(np.abs(final_audio))
    if max_abs > 1.0:
        # Eğer pik noktası çok yüksekse (ses patlaması), clip uygula ve max_abs'ye böl
        final_audio = np.clip(final_audio, -1.0, 1.0)
    elif max_abs > 0:
        final_audio = final_audio / max_abs
        
    return final_audio

def extract_features(audio_data, sr):
    """ Backend sistemini (OpenSMILE) kullanarak 1584 boyutuna tamamlanmış matris döndürür. """
    try:
        reduced_audio = nr.reduce_noise(y=audio_data, sr=sr, prop_decrease=0.7)
        if reduced_audio.ndim == 1:
            reduced_audio = np.expand_dims(reduced_audio, axis=0) # (1, samples)
            
        df = smile.process_signal(reduced_audio, sr)
        if df is None or df.empty: return None
        
        features = df.to_numpy().flatten()
        if len(features) != 1582: return None
        
        features_padded = np.zeros(1584, dtype=np.float32)
        features_padded[1:1583] = features
        return features_padded
    except Exception:
        return None

def build_dataset():
    print(f"\n{'-'*60}")
    print(" 🛠️ ADIM 1: ZORLU DÜNYA ŞARTLARINDA SENTETİK CÜMLELER ÜRETİLİYOR")
    print(f"{'-'*60}")
    
    X_list, y_list = [], []
    
    for emo_label, folder_name in EMOTION_FOLDERS.items():
        folder_path = os.path.join(SOUND_DIR, folder_name)
        if not os.path.exists(folder_path):
            print(f"HATA: {folder_path} bulunamadı!")
            continue
            
        wav_files = glob.glob(os.path.join(folder_path, "*.wav"))
        if len(wav_files) < 2:
            print(f"Uyarı: {emo_label} için yeterli wav dosyası yok.")
            continue
            
        print(f"⏳ {folder_name} kategorisinden {NUM_SENTENCES_PER_EMOTION} rastgele gürültülü cümle sentezleniyor...")
        
        count = 0
        attempts = 0
        while count < NUM_SENTENCES_PER_EMOTION and attempts < NUM_SENTENCES_PER_EMOTION * 3:
            attempts += 1
            audio_array = create_synthetic_sentence(wav_files, sr=SAMPLE_RATE)
            if audio_array is None: continue
            
            feats = extract_features(audio_data=audio_array, sr=SAMPLE_RATE)
            if feats is not None:
                X_list.append(feats)
                y_list.append(emo_label) # angry, calm, happy, sad
                count += 1
                
        print(f"  -> Bitti ({count}/{NUM_SENTENCES_PER_EMOTION}).")
        
    X = np.array(X_list)
    y = np.array(y_list)
    print(f"\n✅ Veriseti tamamlandı! Toplam Örnek Sayısı: {len(X)}")
    return X, y

def map_predictions(y_pred_idx, encoder, key):
    # DL Modellerinde ve klasik modellerde OHE / Integer farklılıklarını yönet
    try:
        if (key in ('dnn', 'cnn1d', 'dnn_robust', 'cnn1d_robust')) and len(encoder.classes_.shape) > 1:
            y_pred_str = encoder.inverse_transform(y_pred_idx.reshape(-1, 1)).flatten()
        else:
            y_pred_str = encoder.inverse_transform(y_pred_idx.astype(int))
    except ValueError:
        try:
            categorical_preds = np.zeros((len(y_pred_idx), len(encoder.classes_)))
            categorical_preds[np.arange(len(y_pred_idx)), y_pred_idx] = 1
            y_pred_str = encoder.inverse_transform(categorical_preds).flatten()
        except:
             y_pred_str = ["unknown"] * len(y_pred_idx)
    except:
        y_pred_str = encoder.inverse_transform(y_pred_idx.astype(int))
        
    if len(np.array(y_pred_str).shape) > 1:
        y_pred_str = np.array(y_pred_str).flatten()
        
    # 'neutral' var ise 'calm' yap
    y_pred_str = [yp.lower() if yp.lower() != 'neutral' else 'calm' for yp in y_pred_str]
    return y_pred_str

def run_benchmarks(X_test, y_true_str):
    print(f"\n{'-'*60}")
    print(" 🚀 ADIM 2: MODELLER ZORLU GÜRÜLTÜLÜ CÜMLELER İÇİN TEST EDİLİYOR")
    print(f"{'-'*60}")
    
    unique_emotions = sorted(list(set(y_true_str)))
    model_metrics = [] # { model_name, f1_scores: dict, macro_p, macro_r, macro_f1 }
    
    results_text = []
    results_text.append("="*80)
    results_text.append(" PROFESYONEL AKADEMİK ZORLU CÜMLE BENCHMARK RAPORU ")
    results_text.append("="*80)
    results_text.append(f"Parametreler: {NUM_SENTENCES_PER_EMOTION} Cümle/Duygu, Gürültü/PacketLoss/Volume Değişimleri Aktif.\n")
    
    for key, paths in MODEL_CONFIG.items():
        if not os.path.exists(paths['model']):
            continue
            
        print(f"⏱️ Değerlendiriliyor: {key.upper()}...")
        try:
            if paths['model'].endswith('.h5'):
                if not TF_AVAILABLE: continue
                model = load_model(paths['model'])
            else:
                model = joblib.load(paths['model'])
                
            scaler = joblib.load(paths['scaler'])
            encoder = joblib.load(paths['encoder'])
            
            X_scaled = scaler.transform(X_test)
            
            if key in ('cnn1d', 'cnn1d_robust'):
                X_scaled_3d = np.expand_dims(X_scaled, axis=2)
                probs = model.predict(X_scaled_3d, verbose=0)
                y_pred_idx = np.argmax(probs, axis=1)
            elif key in ('dnn', 'dnn_robust'):
                probs = model.predict(X_scaled, verbose=0)
                y_pred_idx = np.argmax(probs, axis=1)
            else:
                y_pred_idx = model.predict(X_scaled)
                if len(y_pred_idx.shape) > 1 and y_pred_idx.shape[1] == 1:
                    y_pred_idx = y_pred_idx.flatten()

            y_pred_str = map_predictions(y_pred_idx, encoder, key)
            
            report = classification_report(y_true_str, y_pred_str, target_names=unique_emotions, output_dict=True, zero_division=0)
            
            m_data = {
                'name': key.upper(),
                'f1_scores': {},
                'macro_p': report['macro avg']['precision'],
                'macro_r': report['macro avg']['recall'],
                'macro_f1': report['macro avg']['f1-score'],
                'acc': report['accuracy']
            }
            
            results_text.append(f"\n{'='*55}")
            results_text.append(f" 🔹 MODEL: {m_data['name']}")
            results_text.append(f"{'='*55}")
            results_text.append(f"{'Duygu':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
            results_text.append("-" * 55)
            
            for emotion in unique_emotions:
                if emotion in report:
                    metrics = report[emotion]
                    p, r, f = metrics['precision'], metrics['recall'], metrics['f1-score']
                    m_data['f1_scores'][emotion] = f
                    results_text.append(f"{emotion.capitalize():<10} | {p:.4f}     | {r:.4f}   | {f:.4f}")
                else:
                    m_data['f1_scores'][emotion] = 0.0
                    results_text.append(f"{emotion.capitalize():<10} | 0.0000     | 0.0000   | 0.0000")
                    
            results_text.append("-" * 55)
            results_text.append(f"Accuracy  : {m_data['acc']:.4f}")
            results_text.append(f"Macro P   : {m_data['macro_p']:.4f}")
            results_text.append(f"Macro R   : {m_data['macro_r']:.4f}")
            results_text.append(f"Macro F1  : {m_data['macro_f1']:.4f}")
            results_text.append(f"=======================================================\n")
            
            model_metrics.append(m_data)

        except Exception as e:
            print(f"HATA: {key.upper()} test edilirken sorun oluştu - {str(e)}")
            continue

    # 1. Raporu Kaydet
    txt_path = os.path.join(BASE_DIR, "sentence_robust_metrics_all.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
         f.write("\n".join(results_text))
    print(f"\n✅ Akademik Metin Raporu Kaydedildi: {txt_path}")

    # 2. Makro-Ortalamalar Grafiği (Modellerin Genel Zeka Karşılaştırması)
    if model_metrics:
        plot_macro_metrics(model_metrics)
        plot_grouped_emotions(model_metrics, unique_emotions)
        print("\n✅ Tüm Görseller ve Veriler Başarıyla Hazırlandı!")

def plot_macro_metrics(model_metrics):
    names = [m['name'] for m in model_metrics]
    precisions = [m['macro_p'] for m in model_metrics]
    recalls = [m['macro_r'] for m in model_metrics]
    f1s = [m['macro_f1'] for m in model_metrics]

    x = np.arange(len(names))
    width = 0.25

    plt.figure(figsize=(18, 9))
    plt.bar(x - width, precisions, width, label='Macro Precision', color='#8ac926', edgecolor='black', alpha=0.9)
    plt.bar(x, recalls, width, label='Macro Recall', color='#1982c4', edgecolor='black', alpha=0.9)
    plt.bar(x + width, f1s, width, label='Macro F1-Score', color='#ff595e', edgecolor='black', alpha=0.9)

    plt.xlabel('Tüm Modeller (Standart ve Robust Sürümler)', fontweight='bold', fontsize=12)
    plt.ylabel('Puan (0-1.0 Arası Değer)', fontweight='bold', fontsize=12)
    plt.title('Zorlu Cümle/Dış Ses Senaryosunda Genel Model Başarısı Karşılaştırması', fontweight='bold', fontsize=16)
    plt.xticks(x, names, rotation=45, ha='right', fontsize=10)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    img_path = os.path.join(BASE_DIR, "sentence_robust_macro_metrics.png")
    plt.savefig(img_path, dpi=300)
    print(f"📸 Genel (Macro) Kıyas Grafik Kaydedildi: {img_path}")

def plot_grouped_emotions(model_metrics, unique_emotions):
    # Duygu bazlı F1-skoru karşılaştırması
    colors = ['#ff595e', '#1982c4', '#8ac926', '#6a4c93']
    names = [m['name'].replace('_ROBUST', ' (R)') for m in model_metrics]
    
    plt.figure(figsize=(18, 10))
    x = np.arange(len(names))
    width = 0.2
    
    for i, (emotion, color) in enumerate(zip(unique_emotions, colors)):
        vals = [m['f1_scores'][emotion] for m in model_metrics]
        offset = (i - 1.5) * width
        plt.bar(x + offset, vals, width, label=emotion.capitalize(), color=color, edgecolor='black', alpha=0.85)
        
    plt.xlabel('Tüm Modeller', fontweight='bold', fontsize=12)
    plt.ylabel('F1 Score', fontweight='bold', fontsize=12)
    plt.title('Duygu Kategorilerine Göre Sentetik Cümle Performansları (F1-Score)', fontweight='bold', fontsize=16)
    plt.xticks(x, names, rotation=45, ha='right', fontsize=10)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.legend(title='Duygular', bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    img_path = os.path.join(BASE_DIR, "sentence_robust_emotion_f1.png")
    plt.savefig(img_path, dpi=300)
    print(f"📸 Duygu Kıyaslama Grafiği Kaydedildi: {img_path}")

if __name__ == "__main__":
    X, y = build_dataset()
    if X is not None and len(X) > 0:
        run_benchmarks(X, y)
    else:
        print("❌ HATA: Veriseti oluşturulamadı, test başlatılamıyor.")
