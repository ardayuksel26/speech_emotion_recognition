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
import soundfile as sf
import librosa
import noisereduce as nr
import opensmile
from sklearn.metrics import classification_report
import tempfile
import time

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
BACKEND_DIR = os.path.abspath(os.path.join(BASE_DIR, '../Backend'))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))

# Backend ve Sentence modüllerini path'e ekle
sys.path.append(BACKEND_DIR)
sys.path.append(ROOT_DIR)

from Sentence.sentence_processing import SentenceProcessor
try:
    from stt_service import transcribe
    STT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ STT Service bulunamadı veya yüklenemedi: {e}")
    STT_AVAILABLE = False

# --- CONF AYARLARI ---
EMOTION_FOLDERS = {"angry": "Angry", "calm": "Calm", "happy": "Happy", "sad": "Sad"}

# WhisperX performansı düştüğü / hata verdiği için örneklem sayısını 10'da tutuyoruz.
NUM_SENTENCES_PER_EMOTION = 10 
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

# Initialize SentenceProcessor for VAD and Voting logic
processor = SentenceProcessor(target_sr=SAMPLE_RATE, vad_db_threshold=40)

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

# Modelleri RAM'e Yükle
loaded_models = {}
for key, paths in MODEL_CONFIG.items():
    if not os.path.exists(paths['model']):
        continue
    try:
        if paths['model'].endswith('.h5'):
            if not TF_AVAILABLE: continue
            model = load_model(paths['model'])
        else:
            model = joblib.load(paths['model'])
        scaler = joblib.load(paths['scaler'])
        encoder = joblib.load(paths['encoder'])
        loaded_models[key] = {'model': model, 'scaler': scaler, 'encoder': encoder}
    except Exception as e:
        print(f"Hata yükleme: {key} - {e}")


def create_synthetic_sentence(files, sr=SAMPLE_RATE, min_len=MIN_LEN_SEC, max_len=MAX_LEN_SEC):
    """
    Orta-Üst Segment Mikrofon Simülasyonu.
    Daha yüksek kelime yoğunluğu (3-5 sn içinde 5 ila 9 kelime).
    Çok kısa sessizlikler (0.03sn - 0.2sn).
    Çok hafif dip gürültüsü ve temiz aktarım.
    """
    num_words = random.randint(5, 9)
    chosen = random.sample(files, min(num_words, len(files)))
    
    # Çok kısa bir yutkunma/nefes payı (0.05sn - 0.1sn)
    sentence_audio = [np.zeros(int(random.uniform(0.05, 0.1) * sr))]
    
    for f in chosen:
        try:
            y, _ = librosa.load(f, sr=sr)
            # Daha hassas trim (daha çok sesi korur)
            y_trimmed, _ = librosa.effects.trim(y, top_db=30) 
            sentence_audio.append(y_trimmed)
        except Exception:
            continue
        # Kelime arası sessizlik çok daha kısa (akıcı konuşma simülasyonu)
        gap_len = random.uniform(0.03, 0.2)
        sentence_audio.append(np.zeros(int(gap_len * sr)))
        
    if not sentence_audio: return None
        
    final_audio = np.concatenate(sentence_audio)
    duration = len(final_audio) / sr
    if duration < min_len:
        final_audio = np.concatenate((final_audio, np.zeros(int((min_len - duration) * sr))))
    if (len(final_audio) / sr) > max_len:
        final_audio = final_audio[:int(max_len * sr)]
        
    # --- YÜKSEK KALİTE (HQ) MİKROFON EFEKTLERİ ---
    # İstikrarlı ses çıkışı
    final_audio = final_audio * random.uniform(0.85, 1.1) 
    
    # %20 ihtimalle çok hafif, ortamın doğal beyaz gürültüsü
    if random.random() < 0.20: 
        final_audio += np.random.normal(0, random.uniform(0.001, 0.003), len(final_audio))
        
    # %5 ihtimalle minik USB veya Bluetooth gecikme cızırtısı
    if random.random() < 0.05: 
        drop_dur = int(sr * random.uniform(0.02, 0.05))
        drop_start = random.randint(0, max(0, len(final_audio) - drop_dur))
        final_audio[drop_start:drop_start+drop_dur] = 0.0

    max_abs = np.max(np.abs(final_audio))
    if max_abs > 1.0: final_audio = np.clip(final_audio, -1.0, 1.0)
    elif max_abs > 0: final_audio = final_audio / max_abs
        
    return final_audio

def extract_features(audio_data, sr):
    try:
        reduced_audio = nr.reduce_noise(y=audio_data, sr=sr, prop_decrease=0.7)
        if reduced_audio.ndim == 1: reduced_audio = np.expand_dims(reduced_audio, axis=0)
        df = smile.process_signal(reduced_audio, sr)
        if df is None or df.empty: return None
        features = df.to_numpy().flatten()
        if len(features) != 1582: return None
        features_padded = np.zeros(1584, dtype=np.float32)
        features_padded[1:1583] = features
        return features_padded
    except Exception:
        return None

def build_datasets_for_methods():
    print(f"\n{'-'*60}")
    print(" 🛠️ ADIM 1: YÜKSEK KALİTE CÜMLELER ÜRETİLİYOR (HQ MIC)")
    print("   Metotlar: 1) Energy-Based VAD  2) Vosk STT")
    print(f"{'-'*60}")
    
    methods = ["VAD", "VOSK"] if STT_AVAILABLE else ["VAD"]
    
    # Her metot içindeki test sonuçlarını tutacak ana veri yapısı
    test_data = {m: [] for m in methods}
    
    temp_wav = os.path.join(BASE_DIR, "temp_bench_sentence_hq.wav")
    
    for emo_label, folder_name in EMOTION_FOLDERS.items():
        folder_path = os.path.join(SOUND_DIR, folder_name)
        if not os.path.exists(folder_path): continue
        wav_files = glob.glob(os.path.join(folder_path, "*.wav"))
        
        print(f"⏳ {emo_label.upper()} kategorisinden {NUM_SENTENCES_PER_EMOTION} yüksek kalite cümle ayrıştırılıyor...")
        count = 0
        attempts = 0
        while count < NUM_SENTENCES_PER_EMOTION and attempts < NUM_SENTENCES_PER_EMOTION * 3:
            attempts += 1
            audio_array = create_synthetic_sentence(wav_files, sr=SAMPLE_RATE)
            if audio_array is None: continue
            
            sf.write(temp_wav, audio_array, SAMPLE_RATE)
            
            method_features = {m: [] for m in methods}
            valid_sentence_for_all_methods = True
            
            for m in methods:
                try:
                    if m == "VAD":
                        segments_info = processor.extract_segments_info(temp_wav)
                    elif m == "VOSK":
                        segments_info = transcribe(temp_wav, engine="vosk")
                        
                    if not segments_info:
                        segments_info = [{'start': 0.0, 'end': len(audio_array)/SAMPLE_RATE}]
                        
                    feats_list = []
                    for seg in segments_info:
                        start_idx = int(seg['start'] * SAMPLE_RATE)
                        end_idx = int((seg['end'] + 0.1) * SAMPLE_RATE)
                        seg_audio = audio_array[start_idx:end_idx]
                        
                        if len(seg_audio) > (0.1 * SAMPLE_RATE):
                            f = extract_features(seg_audio, SAMPLE_RATE)
                            if f is not None:
                                feats_list.append(f)
                                
                    if not feats_list:
                        f = extract_features(audio_array, SAMPLE_RATE)
                        if f is not None: feats_list.append(f)
                        else: valid_sentence_for_all_methods = False; break
                        
                    method_features[m] = feats_list
                except Exception as e:
                    print(f"Uyarı: {m} segmentation hatası: {e}")
                    valid_sentence_for_all_methods = False
                    break
                    
            if valid_sentence_for_all_methods:
                for m in methods:
                    test_data[m].append({
                        'ground_truth': emo_label,
                        'segments_features': method_features[m]
                    })
                count += 1
                
        print(f"  -> Bitti ({count}/{NUM_SENTENCES_PER_EMOTION}).")
        
    if os.path.exists(temp_wav):
        os.remove(temp_wav)
        
    return test_data, methods

def predict_segments_and_vote(model_data, model_key, segments_features, ground_truth):
    model = model_data['model']
    scaler = model_data['scaler']
    encoder = model_data['encoder']
    
    results = []
    
    for f in segments_features:
        f_reshaped = f.reshape(1, -1)
        f_scaled = scaler.transform(f_reshaped)
        
        all_scores_dict = {}
        if model_key in ['cnn1d', 'cnn1d_robust']:
            f_3d = np.expand_dims(f_scaled, axis=2)
            probs = model.predict(f_3d, verbose=0)[0]
            pred_idx = np.argmax(probs)
            for i, c_name in enumerate(encoder.classes_):
                if c_name.lower() != 'neutral':
                    all_scores_dict[c_name.lower()] = float(probs[i] * 100)
        elif model_key in ['dnn', 'dnn_robust']:
            probs = model.predict(f_scaled, verbose=0)[0]
            pred_idx = np.argmax(probs)
            for i, c_name in enumerate(encoder.classes_):
                if c_name.lower() != 'neutral':
                    all_scores_dict[c_name.lower()] = float(probs[i] * 100)
        else:
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(f_scaled)[0]
                pred_idx = np.argmax(probs)
                for i, c_name in enumerate(encoder.classes_):
                    if c_name.lower() != 'neutral':
                        all_scores_dict[c_name.lower()] = float(probs[i] * 100)
            else:
                pred = model.predict(f_scaled)[0]
                pred_idx = int(pred.item()) if isinstance(pred, (list, np.ndarray)) else int(pred)
                label = encoder.inverse_transform([pred_idx])[0].lower()
                all_scores_dict = {label: 100.0}

        total = sum(all_scores_dict.values())
        if total > 0:
            all_scores_dict = {k: (v/total)*100 for k,v in all_scores_dict.items()}
            
        label = max(all_scores_dict.items(), key=lambda x: x[1])[0]
        conf = all_scores_dict[label]
        
        results.append({
            'emotion': label,
            'confidence': conf,
            'all_scores': all_scores_dict
        })
        
    if not results:
        return 'unknown'
        
    voted_res = processor.weighted_voting(results)
    if voted_res:
        return voted_res['final_emotion']
    return 'unknown'

def run_benchmarks(test_data, methods):
    print(f"\n{'-'*60}")
    print(" 🚀 ADIM 2: MODELLER YÜKSEK KALİTE CÜMLELERDE DEĞERLENDİRİLİYOR")
    print(f"{'-'*60}")
    
    unique_emotions = sorted(list(EMOTION_FOLDERS.keys()))
    
    results_text = []
    results_text.append("================================================================================")
    results_text.append(" PROFESYONEL HQ CÜMLE METODOLOJİLERİ KARŞILAŞTIRMA RAPORU ")
    results_text.append(" Orta/Üst Segment Mikrofon Simülasyonu (Az gürültü, yüksek kelime yoğunluğu)")
    results_text.append("================================================================================")
    results_text.append(f"Parametreler: {NUM_SENTENCES_PER_EMOTION} Cümle/Duygu, 5-9 Kelime/Cümle, 3-5s uzunluk.")
    results_text.append("Metotlar: Energy-Based VAD, Vosk STT, WhisperX STT\n\n")
    
    full_metrics = {m: [] for m in methods}
    
    for method in methods:
        data_method = test_data[method]
        
        y_true_str = [d['ground_truth'] for d in data_method]
        
        results_text.append(f"\n{chr(9608)*80}")
        results_text.append(f" 🟩 METOT: {method.upper()} SEGMENTATION")
        results_text.append(f"{chr(9608)*80}\n")
        
        for model_key, model_data in loaded_models.items():
            print(f"[{method}] ⏱️ Değerlendiriliyor: {model_key.upper()}...")
            
            y_pred_str = []
            for item in data_method:
                pred_final = predict_segments_and_vote(
                    model_data, model_key, 
                    item['segments_features'], 
                    item['ground_truth']
                )
                y_pred_str.append(pred_final)
                
            report = classification_report(y_true_str, y_pred_str, target_names=unique_emotions, output_dict=True, zero_division=0)
            
            m_data = {
                'name': model_key.upper(),
                'f1_scores': {},
                'macro_p': report.get('macro avg', {}).get('precision', 0),
                'macro_r': report.get('macro avg', {}).get('recall', 0),
                'macro_f1': report.get('macro avg', {}).get('f1-score', 0),
                'acc': report.get('accuracy', 0)
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
            results_text.append(f"Macro F1  : {m_data['macro_f1']:.4f}")
            results_text.append(f"=======================================================\n")
            
            full_metrics[method].append(m_data)

    txt_path = os.path.join(BASE_DIR, "sentence_hq_methods_benchmark_results.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
         f.write("\n".join(results_text))
    print(f"\n✅ Akademik HQ Metin Raporu Kaydedildi: {txt_path}")

    plot_methods_comparison(full_metrics, methods)

def plot_methods_comparison(full_metrics, methods):
    if not full_metrics or not list(full_metrics.values())[0]:
        return
        
    model_names = [m['name'].replace('_ROBUST', '(R)') for m in full_metrics[methods[0]]]
    x = np.arange(len(model_names))
    width = 0.25
    
    plt.figure(figsize=(20, 10))
    colors = ['#1982c4', '#8ac926', '#ff595e']
    
    for i, method in enumerate(methods):
        accs = [m['acc'] for m in full_metrics[method]]
        offset = (i - 1) * width
        label = f"{method} Yöntemi (HQ)"
        plt.bar(x + offset, accs, width, label=label, color=colors[i % len(colors)], edgecolor='black', alpha=0.9)
        
    plt.xlabel('Seri (Modeller)', fontweight='bold', fontsize=12)
    plt.ylabel('Cümle Bütünü Duygu Tahmini Doğruluğu (Accuracy)', fontweight='bold', fontsize=12)
    plt.title('HQ Mikrofon Simülasyonunda VAD, Vosk ve WhisperX Parçalama Yöntemlerinin Karşılaştırılması', fontweight='bold', fontsize=16)
    plt.xticks(x, model_names, rotation=45, ha='right', fontsize=10)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    img_path = os.path.join(BASE_DIR, "sentence_hq_methods_comparison.png")
    plt.savefig(img_path, dpi=300)
    print(f"📸 Grafikler Kaydedildi: {img_path}")

if __name__ == "__main__":
    t_data, test_methods = build_datasets_for_methods()
    if t_data and any(t_data.values()):
        run_benchmarks(t_data, test_methods)
    else:
        print("❌ HATA: Veriseti oluşturulamadı veya parçalanacak metot çalışmadı.")
