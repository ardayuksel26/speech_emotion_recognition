# Force Auto-reload trigger for Perfect Confidence & Distribution UI Sync
import os
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from preprocessing import extract_features
import sys
# Cümleleştirme mantığı artık root/Sentence klasöründe olduğu için sys.path ekliyoruz
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Sentence.sentence_processing import SentenceProcessor
from Voting.majority_voting import calculate_majority_vote, MODEL_WEIGHTS
from stt_service import transcribe

# TensorFlow imports for DL models
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    print("⚠️ TensorFlow not available. DL models (CNN, DNN) will be disabled.")
    TF_AVAILABLE = False

import warnings
import logging

# --- PRODUCTION LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('SER_API')

app = Flask(__name__)
# Enable CORS securely for all origins
CORS(app, resources={r"/*": {"origins": "*"}})

# ==============================================================================
# VOCAL TONE CALIBRATION (Turkish Emotion Priorities)
# ==============================================================================
# Microphones often dynamically compress high frequencies, making energetic 
# Turkish "Happy" (Mutlu) expressions acoustically misclassify as "Sad" (Üzgün).
# Tweak these global multipliers if the models are continuously biasing toward one.
VOCAL_CALIBRATION = {
    'happy': 1.30,  # +30% Boost: En zor tespit edilen duygu, mikrofon bass'ı yutuyor.
    'sad': 0.80,    # -20% Penalty: Boğuk mikrofon sesi genelde "üzgün" sanılır, bunu bastırıyoruz.
    'angry': 1.10,  # +10% Boost: Yüksek enerji gerektiren kelimeleri daha net yakalasın.
    'calm': 0.90    # -10% Penalty: Düz/cansız seslerin hemen 'sakin' sanılmasını önler.
}

# --- AYARLAR / DIRECTORIES ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '../Models')

# Hangi modelleri destekliyoruz? İsimleri ve dosya yolları burada tanımlı.
# DİKKAT: Dosya adlarının senin kaydettiklerinle aynı olduğundan emin ol.
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
    # Deep Learning Models
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

# --- TÜM MODELLERİ YÜKLE (LOADER) ---
loaded_models = {}

logger.info("Modeller başlatılıyor. Lütfen bekleyin...")

for key, paths in MODEL_CONFIG.items():
    try:
        # Check if the primary model file exists
        if os.path.exists(paths['model']):
            
            # Load according to extension
            if paths['model'].endswith('.h5'):
                if TF_AVAILABLE:
                    model_obj = load_model(paths['model'])
                else:
                    logger.warning(f"[{key.upper()}] Atlanıyor (TensorFlow bağımlılığı bulunamadı).")
                    continue
            else:
                model_obj = joblib.load(paths['model'])

            loaded_models[key] = {
                'model': model_obj,
                'scaler': joblib.load(paths['scaler']),
                'encoder': joblib.load(paths['encoder'])
            }
            logger.info(f"[{key.upper()}] Model ve preprocessors başarıyla yüklendi.")
        else:
            logger.warning(f"[{key.upper()}] Dosyaları bulunamadı, atlanıyor. ({paths['model']})")
    except Exception as e:
        logger.error(f"[{key.upper()}] Yüklenirken kritik hata: {e}")

logger.info(f"Sistem hazır. Toplam {len(loaded_models)} model aktifleştirildi!")


@app.route('/', methods=['GET'])
def home():
    # Hangi modellerin aktif olduğunu gösterelim
    active_models = list(loaded_models.keys())
    return jsonify({
        "message": "Speech Emotion Recognition API Çalışıyor!",
        "available_models": active_models
    })

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Frontend'den gelen 'model_type' verisini al (Varsayılan: catboost)
    selected_model_key = request.form.get('model_type', 'catboost')

    # 2. Seçilen model yüklü mü diye kontrol et
    if selected_model_key not in loaded_models:
        return jsonify({
            'error': f"Seçilen model ('{selected_model_key}') bulunamadı veya yüklenemedi.",
            'available_models': list(loaded_models.keys())
        }), 400

    # İlgili modelin araçlarını al
    tools = loaded_models[selected_model_key]
    model = tools['model']
    scaler = tools['scaler']
    encoder = tools['encoder']

    # 3. Dosya Kontrolü
    if 'file' not in request.files:
        return jsonify({'error': 'Ses dosyası yok'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'}), 400

    temp_path = os.path.join(BASE_DIR, 'temp_audio.wav')
    file.save(temp_path)

    try:
        # 4. Özellik Çıkarımı
        features = extract_features(temp_path)
        
        if features is None:
            return jsonify({'error': 'Ses özellikleri çıkarılamadı'}), 500

        # 5. Tahmin (Seçilen Modele Göre)
        features = features.reshape(1, -1)
        features_scaled = scaler.transform(features)

        probabilities = None
        prediction_index = None
        confidence = 0.0

        if selected_model_key in ['cnn1d', 'dnn']:
            # DL Modelleri (Keras - .h5)
            # CNN için 3D input gerekebilir: (batch, steps, features) veya (batch, features, channels)
            # Burada (1, 1584, 1) yapıyoruz
            if 'cnn' in selected_model_key:
                features_scaled = np.expand_dims(features_scaled, axis=2)
            
            # Keras predict
            probabilities = model.predict(features_scaled, verbose=0)[0]
            prediction_index = np.argmax(probabilities)
            confidence = np.max(probabilities) * 100
            
        else:
            # Klasik ML Modelleri (Sklearn/CatBoost etc)
            prediction_index = model.predict(features_scaled)[0]
            
            # Bazı modeller (örn. CatBoost) sonucu liste içinde verir, bazıları (RF) direkt sayı verir.
            if isinstance(prediction_index, list) or isinstance(prediction_index, np.ndarray):
                prediction_index = prediction_index.item()
                
            # Olasılıkları al (varsa)
            try:
                probabilities = model.predict_proba(features_scaled)[0]
                confidence = np.max(probabilities) * 100
            except:
                probabilities = None
                confidence = 0.0

        # Olasılık (Confidence) ve Tüm Skorlar
        all_scores = {}
        if probabilities is not None:
            # Sınıf isimlerini al
            class_names = encoder.classes_
            
            # --- APPLY VOCAL TONE CALIBRATION ---
            raw_scores = {}
            for i, class_name in enumerate(class_names):
                cat = class_name.lower()
                if cat != 'neutral': # Ignore any legacy neutral output
                    # Apply global calibration multiplier
                    multiplier = VOCAL_CALIBRATION.get(cat, 1.0)
                    raw_scores[cat] = float(probabilities[i] * 100) * multiplier
                    
            # Re-normalize out of 100%
            total = sum(raw_scores.values())
            target_scores = {}
            if total > 0:
                for k, v in raw_scores.items():
                    target_scores[k] = (v / total) * 100
            else:
                target_scores = raw_scores
                
            all_scores = target_scores
            
            # Update final prediction based on calibrated scores
            predicted_label = max(all_scores.items(), key=lambda x: x[1])[0]
            confidence = all_scores[predicted_label]
            
        else:
             # Probability yoksa sadece tahmin edilene (önceden hesaplanan) 100 ver
             predicted_label = encoder.inverse_transform([int(prediction_index)])[0].lower()
             if predicted_label == 'neutral':
                 predicted_label = 'calm' # Force fallback
             all_scores = {predicted_label: 100.0}
             confidence = 100.0

        # --- STT (Speech-to-Text) Integration ---
        stt_engine = request.form.get('stt_engine', None)
        word_timestamps = []
        if stt_engine and stt_engine in ('vosk', 'whisperx'):
            try:
                # STT işlemi için orijinal dosyayı tekrar kaydet (predict zaten temp_path'e kaydetti)
                # Ancak dosya silinmiş olabilir, kontrol et
                stt_temp = os.path.join(BASE_DIR, f'temp_stt_{selected_model_key}.wav')
                # Orijinal dosyayı tekrar kaydet
                request.files['file'].seek(0)
                request.files['file'].save(stt_temp)
                
                stt_words = transcribe(stt_temp, engine=stt_engine)
                
                # Frontend'in beklediği formata dönüştür
                for w in stt_words:
                    word_timestamps.append({
                        'word': w['word'],
                        'start': w['start'],
                        'end': w['end'],
                        'emotion': predicted_label,
                        'confidence': float(confidence) / 100.0
                    })
                
                if os.path.exists(stt_temp):
                    os.remove(stt_temp)
                    
            except Exception as stt_err:
                logger.warning(f"STT hatası ({stt_engine}): {stt_err}")

        # Temizlik
        if os.path.exists(temp_path):
            os.remove(temp_path)

        response_data = {
            'emotion': predicted_label,
            'confidence': f"%{confidence:.2f}",
            'all_scores': all_scores,
            'model_used': selected_model_key.upper()
        }
        if word_timestamps:
            response_data['word_timestamps'] = word_timestamps

        return jsonify(response_data)

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': f'Tahmin hatası: {str(e)}'}), 500

def _predict_emotions_for_segments(audio_path, segments, start_key='start', end_key='end', model_key='catboost'):
    """
    Kelimelerin veya cümle parçalarının başlangıç/bitiş zamanlarını kullanarak hızlı bir model 
    üzerinden (varsayılan: catboost) her kelimenin duygusunu tahmin eder ve segment içine 'emotion' ekler.
    """
    if not segments:
        return segments
    try:
        import librosa
        import soundfile as sf
        import tempfile
        import numpy as np

        tools = loaded_models.get(model_key)
        if not tools:
            for s in segments: s['emotion'] = 'neutral'
            return segments

        model, scaler, encoder = tools['model'], tools['scaler'], tools['encoder']
        y, sr = librosa.load(audio_path, sr=22050)

        for seg in segments:
            start_s = float(seg.get(start_key, 0))
            end_s = float(seg.get(end_key, 0))
            if end_s - start_s < 0.1:
                seg['emotion'] = 'neutral'
                continue

            start_idx = int(start_s * sr)
            end_idx = int((end_s + 0.1) * sr) # Add a small buffer for very short words
            y_seg = y[start_idx:end_idx]

            fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            
            try:
                sf.write(tmp_path, y_seg, sr)
                feats = extract_features(tmp_path)
                
                if feats is None:
                    seg['emotion'] = 'neutral'
                else:
                    feats = feats.reshape(1, -1)
                    feats_scaled = scaler.transform(feats)
                    
                    if hasattr(model, 'predict_proba'):
                        probs = model.predict_proba(feats_scaled)[0]
                        pred_idx = np.argmax(probs)
                    else:
                        pred = model.predict(feats_scaled)[0]
                        pred_idx = int(pred.item()) if isinstance(pred, np.ndarray) else int(pred)
                        
                    label = encoder.inverse_transform([pred_idx])[0].lower()
                    seg['emotion'] = label
            except Exception as e:
                seg['emotion'] = 'error'
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    
        return segments
    except Exception as e:
        logger.error(f"[ENHANCE] Emotion prediction failed for segments: {e}")
        for s in segments: s.setdefault('emotion', '?')
        return segments


@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """
    Standalone Speech-to-Text endpoint.
    Ses dosyasını Vosk veya WhisperX ile kelimelere ayırır ve zaman damgalarını döndürür.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'Ses dosyası yok'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'}), 400

    stt_engine = request.form.get('stt_engine', 'vosk')
    if stt_engine not in ('vosk', 'whisperx'):
        return jsonify({'error': f"Geçersiz STT motoru: '{stt_engine}'. 'vosk' veya 'whisperx' kullanın."}), 400

    import uuid
    temp_filename = f"temp_transcribe_{uuid.uuid4().hex}.wav"
    temp_path = os.path.join(BASE_DIR, temp_filename)
    file.save(temp_path)

    try:
        import time as _time
        _start = _time.time()
        words = transcribe(temp_path, engine=stt_engine)
        
        # Kelimelere duygu ekle
        fast_model = request.form.get('model_type', 'xgboost') # Varsayılan olarak xgboost ile hızlı test et
        words = _predict_emotions_for_segments(temp_path, words, start_key='start', end_key='end', model_key=fast_model)
        
        _elapsed = round(_time.time() - _start, 2)
        return jsonify({
            'words': words,
            'stt_engine': stt_engine,
            'word_count': len(words),
            'elapsed_seconds': _elapsed
        })
    except Exception as e:
        logger.error(f"[TRANSCRIBE] Hata ({stt_engine}): {e}")
        return jsonify({'error': f'Transkripsiyon hatası: {str(e)}'}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/analyze_voting', methods=['POST'])
def analyze_voting():
    """
    Majority Voting endpoint: runs ALL models for the selected quality mode,
    collects predictions, applies weighted voting, and returns the final result.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'Ses dosyası yok'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'}), 400

    quality = request.form.get('quality', 'robust')  # 'studio' or 'robust'

    import uuid
    temp_filename = f"temp_voting_{uuid.uuid4().hex}.wav"
    temp_path = os.path.join(BASE_DIR, temp_filename)
    file.save(temp_path)

    try:
        # 1. Extract features ONCE
        features = extract_features(temp_path)
        if features is None:
            return jsonify({'error': 'Ses özellikleri çıkarılamadı'}), 500
        features = features.reshape(1, -1)

        # 2. Determine which model keys to use based on quality
        base_keys = ['catboost', 'xgboost', 'lightgbm', 'rf', 'knn', 'svm',
                      'mlp', 'gradient_boosting', 'dnn', 'cnn1d']
        if quality == 'robust':
            target_keys = [f"{k}_robust" for k in base_keys]
        else:
            target_keys = base_keys

        # 3. Run predictions on all available models
        model_predictions = {}
        for model_key in target_keys:
            if model_key not in loaded_models:
                continue
            try:
                tools = loaded_models[model_key]
                model = tools['model']
                scaler = tools['scaler']
                encoder = tools['encoder']

                features_scaled = scaler.transform(features)

                probabilities = None
                if model_key.replace('_robust', '') in ['cnn1d']:
                    features_3d = np.expand_dims(features_scaled, axis=2)
                    probabilities = model.predict(features_3d, verbose=0)[0]
                elif model_key.replace('_robust', '') in ['dnn']:
                    probabilities = model.predict(features_scaled, verbose=0)[0]
                else:
                    try:
                        probabilities = model.predict_proba(features_scaled)[0]
                    except Exception:
                        # Model doesn't support predict_proba
                        pred_idx = model.predict(features_scaled)[0]
                        if isinstance(pred_idx, (list, np.ndarray)):
                            pred_idx = pred_idx.item()
                        pred_label = encoder.inverse_transform([int(pred_idx)])[0].lower()
                        scores = {e: 0.0 for e in ['happy', 'sad', 'angry', 'calm']}
                        if pred_label in scores:
                            scores[pred_label] = 100.0
                        model_predictions[model_key] = scores
                        continue

                if probabilities is not None:
                    class_names = encoder.classes_
                    raw_scores = {}
                    for i, class_name in enumerate(class_names):
                        cat = class_name.lower()
                        if cat != 'neutral':
                            multiplier = VOCAL_CALIBRATION.get(cat, 1.0)
                            raw_scores[cat] = float(probabilities[i] * 100) * multiplier
                    # Re-normalize to 100%
                    total = sum(raw_scores.values())
                    if total > 0:
                        for k in raw_scores:
                            raw_scores[k] = (raw_scores[k] / total) * 100
                    model_predictions[model_key] = raw_scores

            except Exception as e:
                logger.warning(f"[VOTING] {model_key} tahmin hatası: {e}")
                continue

        # 4. Calculate majority vote
        if not model_predictions:
            return jsonify({'error': 'Hiçbir model tahmin yapamadı'}), 500

        result = calculate_majority_vote(model_predictions)

        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return jsonify({
            'emotion': result['final_emotion'],
            'confidence': f"%{result['confidence']:.2f}",
            'all_scores': result['all_scores'],
            'model_used': 'MAJORITY_VOTING',
            'model_details': result['model_details']
        })

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        logger.error(f"[VOTING] Genel hata: {e}")
        return jsonify({'error': f'Voting hatası: {str(e)}'}), 500

@app.route('/segment-sentence', methods=['POST'])
def segment_sentence():
    """
    Returns only the segment timestamps (start, end) without doing model predictions.
    Used for the frontend segment viewer in sentence mode.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    # Unique temp filename
    import uuid
    temp_filename = f"temp_segmentation_{uuid.uuid4().hex}.wav"
    temp_path = os.path.join(BASE_DIR, temp_filename)
    file.save(temp_path)
    
    try:
        import time as _time
        _start = _time.time()
        processor = SentenceProcessor(target_sr=22050, vad_db_threshold=30)
        segments = processor.extract_segments_info(temp_path)
        
        # Segmentlere duygu ekle
        fast_model = request.form.get('model_type', 'xgboost')
        segments = _predict_emotions_for_segments(temp_path, segments, start_key='start', end_key='end', model_key=fast_model)
        
        _elapsed = round(_time.time() - _start, 2)
        
        return jsonify({
            'segments': segments,
            'elapsed_seconds': _elapsed
        })
        
    except Exception as e:
        return jsonify({'error': f"Segmentation error: {str(e)}"}), 500
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/predict-sentence', methods=['POST'])
def predict_sentence():
    """
    Sentence-level prediction endpoint.
    Splits the input audio into segments (words/phrases), predicts emotion for each,
    and aggregates the results.
    """
    # 1. Frontend parameters
    selected_model_key = request.form.get('model_type', 'catboost') # Default user preference
    
    # 2. Model check
    if selected_model_key not in loaded_models:
        return jsonify({
            'error': f"Model '{selected_model_key}' not found.",
            'available_models': list(loaded_models.keys())
        }), 400
        
    tools = loaded_models[selected_model_key]
    model = tools['model']
    scaler = tools['scaler']
    encoder = tools['encoder']

    # 3. File check
    if 'file' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    # Unique temp filename
    import uuid
    temp_filename = f"temp_sent_{uuid.uuid4().hex}.wav"
    temp_path = os.path.join(BASE_DIR, temp_filename)
    file.save(temp_path)
    
    segment_paths = []
    
    try:
        # 4. Segmentation (The core "preprocessing step")
        # Split sentence into word-like chunks using new Class
        processor = SentenceProcessor(target_sr=22050, vad_db_threshold=40)
        segment_paths = processor.process_audio(temp_path, output_dir=os.path.join(BASE_DIR, "temp_segments"))
        
        results = []
        
        # 5. Predict for each segment
        for seg_path in segment_paths:
            feats = extract_features(seg_path)
            
            if feats is None:
                continue
                
            # Prepare features
            feats = feats.reshape(1, -1)
            feats_scaled = scaler.transform(feats)
            
            # Predict
            if selected_model_key in ['cnn1d', 'dnn'] and 'cnn' in selected_model_key:
                 feats_scaled = np.expand_dims(feats_scaled, axis=2)
                 
            # Logic similar to /predict but simplified for loop
            all_scores_dict = {}
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(feats_scaled)[0]
                pred_idx = np.argmax(probs)
                
                # Fetch dictionary of probabilities for all classes
                for i, class_name in enumerate(encoder.classes_):
                    cat = class_name.lower()
                    if cat != 'neutral':
                        all_scores_dict[cat] = float(probs[i]) * 100.0
                        
                # Normalize out of 100
                total_prob = sum(all_scores_dict.values())
                if total_prob > 0:
                    all_scores_dict = {k: (v / total_prob) * 100.0 for k,v in all_scores_dict.items()}
                else:
                    all_scores_dict = {k: 0.0 for k in all_scores_dict.keys()}
                
                label = max(all_scores_dict.items(), key=lambda x: x[1])[0]
                conf = all_scores_dict[label]
            else:
                # Fallback for models without probability
                pred = model.predict(feats_scaled)[0]
                if isinstance(pred, (list, np.ndarray)):
                    pred_idx = int(pred.item())
                else:
                    pred_idx = int(pred)
                conf = 100.0 # Mock confidence
                label = encoder.inverse_transform([pred_idx])[0].lower()
                all_scores_dict = {label: 100.0}

            results.append({
                'segment': os.path.basename(seg_path),
                'emotion': label,
                'confidence': float(conf),
                'all_scores': all_scores_dict
            })
            
        # 6. Aggregation
        if not results:
             return jsonify({'error': 'No valid segments extracted or feature extraction failed.'}), 500
             
        # Majority Vote (Legacy/Reference)
        from collections import Counter
        emotions = [r['emotion'] for r in results]
        most_common = Counter(emotions).most_common(1)[0]
        final_emotion_vote = most_common[0]
        
        # --- NEW: Weighted Voting Strategy via SentenceProcessor ---
        weighted_result = processor.weighted_voting(results)
        
        if weighted_result:
            final_emotion = weighted_result['final_emotion']
            final_conf = weighted_result['confidence']
            details = weighted_result['weighted_details']
        else:
            # Fallback
            final_emotion = final_emotion_vote
            final_conf = 0.0
            details = {}

        response = {
            'final_emotion': final_emotion,
            'confidence': f"%{final_conf:.2f}",
            'segments_count': len(results),
            'breakdown': results,
            'voting_result': final_emotion_vote,
            'weighted_details': details,
            'model_used': selected_model_key.upper()
        }
        
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f"Sentence processing error: {str(e)}"}), 500
        
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
        for p in segment_paths:
            if os.path.exists(p):
                os.remove(p)

if __name__ == '__main__':
    # use_reloader=False ekleyerek klasördeki dosya değişikliklerinin (veya yeni kütüphane yüklemelerinin)
    # sunucuyu aniden resetlemesini engelliyoruz.
    app.run(debug=True, use_reloader=False, port=5000)