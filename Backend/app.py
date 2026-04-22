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
import librosa

# Huggingface model integration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from Huggingface.hubert_model import HubertEmotionPredictor
    HUBERT_AVAILABLE = True
except ImportError:
    HUBERT_AVAILABLE = False

_hubert_predictor = None
_wav2vec2_turkish_predictor = None
_sensevoice_predictor = None

def get_hubert_predictor():
    global _hubert_predictor
    if _hubert_predictor is None:
        try:
            from Huggingface.hubert_model import HubertEmotionPredictor
            _hubert_predictor = HubertEmotionPredictor()
        except Exception as e:
            logger.error(f"Error loading HuBERT: {e}")
            return None
    return _hubert_predictor

def get_wav2vec2_turkish_predictor():
    global _wav2vec2_turkish_predictor
    if _wav2vec2_turkish_predictor is None:
        try:
            from Huggingface.wav2vec2_model import Wav2Vec2TurkishPredictor
            _wav2vec2_turkish_predictor = Wav2Vec2TurkishPredictor()
        except Exception as e:
            logger.error(f"Error loading Wav2Vec2 Turkish: {e}")
            return None
    return _wav2vec2_turkish_predictor

_sensevoice_load_error = None
_exhubert_predictor = None
_wavlm_predictor = None
_xlsr_predictor = None
_qwen2_audio_predictor = None
_wavlm_base_plus_predictor = None
_wav2vec2_english_predictor = None

def get_qwen2_audio_predictor():
    global _qwen2_audio_predictor
    if _qwen2_audio_predictor is None:
        try:
            from Huggingface.qwen2_audio_model import Qwen2AudioEmotionPredictor
            _qwen2_audio_predictor = Qwen2AudioEmotionPredictor()
        except Exception as e:
            logger.error(f"Error loading Qwen2-Audio: {e}")
            return None
    return _qwen2_audio_predictor

def get_wavlm_base_plus_predictor():
    global _wavlm_base_plus_predictor
    if _wavlm_base_plus_predictor is None:
        try:
            from Huggingface.wavlm_model import WavLMBasePlusEmotionPredictor
            _wavlm_base_plus_predictor = WavLMBasePlusEmotionPredictor()
        except Exception as e:
            logger.error(f"Error loading WavLM Base Plus: {e}")
            return None
    return _wavlm_base_plus_predictor

def get_wav2vec2_english_predictor():
    global _wav2vec2_english_predictor
    if _wav2vec2_english_predictor is None:
        try:
            from Huggingface.xlsr_model import Wav2Vec2EnglishPredictor
            _wav2vec2_english_predictor = Wav2Vec2EnglishPredictor()
        except Exception as e:
            logger.error(f"Error loading Wav2Vec2 English: {e}")
            return None
    return _wav2vec2_english_predictor

def get_exhubert_predictor():
    global _exhubert_predictor
    if _exhubert_predictor is None:
        try:
            from Huggingface.exhubert_model import ExHuBERTEmotionPredictor
            _exhubert_predictor = ExHuBERTEmotionPredictor()
        except Exception as e:
            logger.error(f"Error loading ExHuBERT: {e}")
            return None
    return _exhubert_predictor

def get_wavlm_predictor():
    global _wavlm_predictor
    if _wavlm_predictor is None:
        try:
            from Huggingface.wavlm_model import WavLMEmotionPredictor
            _wavlm_predictor = WavLMEmotionPredictor()
        except Exception as e:
            logger.error(f"Error loading WavLM: {e}")
            return None
    return _wavlm_predictor

def get_xlsr_predictor():
    global _xlsr_predictor
    if _xlsr_predictor is None:
        try:
            from Huggingface.xlsr_model import XLSREmotionPredictor
            _xlsr_predictor = XLSREmotionPredictor()
        except Exception as e:
            logger.error(f"Error loading XLSR: {e}")
            return None
    return _xlsr_predictor

def get_sensevoice_predictor():
    global _sensevoice_predictor, _sensevoice_load_error
    if _sensevoice_predictor is None and _sensevoice_load_error is None:
        try:
            from Huggingface.sensevoice_model import SenseVoiceEmotionPredictor
            _sensevoice_predictor = SenseVoiceEmotionPredictor()
        except Exception as e:
            import traceback
            _sensevoice_load_error = traceback.format_exc()
            logger.error(f"Error loading SenseVoice: {e}\n{_sensevoice_load_error}")
    return _sensevoice_predictor

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
    'happy': 1.00,
    'sad': 1.00,
    'angry': 1.00,
    'calm': 1.00
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

# ==============================================================================
# EXPERIMENTAL: WHOLE SENTENCE MODELS (High Accuracy Calibration)
# ==============================================================================
SENTENCE_MODELS_DIR = os.path.join(BASE_DIR, '../Models/Sentence_Models')
SENTENCE_MODEL_CONFIG = {
    'catboost_sentence': {
        'model': os.path.join(SENTENCE_MODELS_DIR, 'CatBoost/catboost_model.pkl'),
        'scaler': os.path.join(SENTENCE_MODELS_DIR, 'CatBoost/scaler_cb.pkl'),
        'encoder': os.path.join(SENTENCE_MODELS_DIR, 'CatBoost/label_encoder_cb.pkl'),
        'weight': 3.0
    },
    'gradboost_sentence': {
        'model': os.path.join(SENTENCE_MODELS_DIR, 'GradientBoosting/gradboost_model.pkl'),
        'scaler': os.path.join(SENTENCE_MODELS_DIR, 'GradientBoosting/scaler_gb.pkl'),
        'encoder': os.path.join(SENTENCE_MODELS_DIR, 'GradientBoosting/label_encoder_gb.pkl'),
        'weight': 3.0
    },
    'xgboost_sentence': {
        'model': os.path.join(SENTENCE_MODELS_DIR, 'XGBoost/xgboost_model.pkl'),
        'scaler': os.path.join(SENTENCE_MODELS_DIR, 'XGBoost/scaler_xgb.pkl'),
        'encoder': os.path.join(SENTENCE_MODELS_DIR, 'XGBoost/label_encoder_xgb.pkl'),
        'weight': 1.0
    },
    'lightgbm_sentence': {
        'model': os.path.join(SENTENCE_MODELS_DIR, 'LightGBM/lightgbm_model.pkl'),
        'scaler': os.path.join(SENTENCE_MODELS_DIR, 'LightGBM/scaler_lgb.pkl'),
        'encoder': os.path.join(SENTENCE_MODELS_DIR, 'LightGBM/label_encoder_lgb.pkl'),
        'weight': 1.0
    },
    'svm_sentence': {
        'model': os.path.join(SENTENCE_MODELS_DIR, 'SVM/svm_model.pkl'),
        'scaler': os.path.join(SENTENCE_MODELS_DIR, 'SVM/scaler_svm.pkl'),
        'encoder': os.path.join(SENTENCE_MODELS_DIR, 'SVM/label_encoder_svm.pkl'),
        'weight': 1.0
    },
    'rf_sentence': {
        'model': os.path.join(SENTENCE_MODELS_DIR, 'random_forest_model.pkl'),
        'scaler': os.path.join(SENTENCE_MODELS_DIR, 'scaler_rf.pkl'),
        'encoder': os.path.join(SENTENCE_MODELS_DIR, 'label_encoder_rf.pkl'),
        'weight': 1.0
    },
    'mlp_sentence': {
        'model': os.path.join(SENTENCE_MODELS_DIR, 'MLP/mlp_model.pkl'),
        'scaler': os.path.join(SENTENCE_MODELS_DIR, 'MLP/scaler_mlp.pkl'),
        'encoder': os.path.join(SENTENCE_MODELS_DIR, 'MLP/label_encoder_mlp.pkl'),
        'weight': 1.0
    },
    'knn_sentence': {
        'model': os.path.join(SENTENCE_MODELS_DIR, 'KNN/knn_model.pkl'),
        'scaler': os.path.join(SENTENCE_MODELS_DIR, 'KNN/scaler_knn.pkl'),
        'encoder': os.path.join(SENTENCE_MODELS_DIR, 'KNN/label_encoder_knn.pkl'),
        'weight': 0.2
    },
    'dnn_sentence': {
        'model': os.path.join(SENTENCE_MODELS_DIR, 'DNN/dnn_model.h5'),
        'scaler': os.path.join(SENTENCE_MODELS_DIR, 'DNN/scaler_dnn.pkl'),
        'encoder': os.path.join(SENTENCE_MODELS_DIR, 'DNN/label_encoder_dnn.pkl'),
        'weight': 0.1
    },
    'cnn1d_sentence': {
        'model': os.path.join(SENTENCE_MODELS_DIR, 'CNN1D/cnn1d_model.h5'),
        'scaler': os.path.join(SENTENCE_MODELS_DIR, 'CNN1D/scaler_cnn1d.pkl'),
        'encoder': os.path.join(SENTENCE_MODELS_DIR, 'CNN1D/label_encoder_cnn1d.pkl'),
        'weight': 0.1
    }
}
CALIBRATION_FILE = os.path.join(BASE_DIR, 'calibration_user.pkl')

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

# ==============================================================================
# MODELS_2 — Noise-Augmented Training (plain IS10, 1582 features)
# ==============================================================================
MODELS_2_DIR = os.path.join(BASE_DIR, '../Models_2')
MODELS_2_CONFIG = {
    'rf_v2':   ('RandomForest/random_forest_model.pkl',          'RandomForest/scaler_rf.pkl',            'RandomForest/label_encoder_rf.pkl'),
    'lgbm_v2': ('LightGBM/lightgbm_model.pkl',                   'LightGBM/scaler_lgbm.pkl',              'LightGBM/label_encoder_lgbm.pkl'),
    'xgb_v2':  ('XGBoost/xgboost_model.pkl',                     'XGBoost/scaler_xgb.pkl',                'XGBoost/label_encoder_xgb.pkl'),
    'cb_v2':   ('CatBoost/catboost_model.pkl',                    'CatBoost/scaler_catboost.pkl',          'CatBoost/label_encoder_catboost.pkl'),
    'gb_v2':   ('GradientBoosting/gradient_boosting_model.pkl',   'GradientBoosting/scaler_gb.pkl',        'GradientBoosting/label_encoder_gb.pkl'),
}
loaded_models_v2 = {}
for _key, (_mp, _sp, _ep) in MODELS_2_CONFIG.items():
    try:
        _m_path = os.path.join(MODELS_2_DIR, _mp)
        if os.path.exists(_m_path):
            loaded_models_v2[_key] = {
                'model':   joblib.load(_m_path),
                'scaler':  joblib.load(os.path.join(MODELS_2_DIR, _sp)),
                'encoder': joblib.load(os.path.join(MODELS_2_DIR, _ep)),
            }
            logger.info(f"[V2] {_key.upper()} yüklendi.")
        else:
            logger.warning(f"[V2] {_key} bulunamadı: {_m_path}")
    except Exception as _e:
        logger.error(f"[V2] {_key} yüklenemedi: {_e}")
logger.info(f"Models_2 hazır. Toplam {len(loaded_models_v2)} model aktifleştirildi.")

def extract_features_plain(file_path):
    """Plain IS10 extraction without denoising — matches Models_2 training features (1582 dims)."""
    try:
        import opensmile as _os2
        _smile2 = _os2.Smile(
            feature_set=_os2.FeatureSet.IS10,
            feature_level=_os2.FeatureLevel.Functionals,
        )
        df = _smile2.process_file(file_path)
        feats = df.to_numpy().flatten().astype(np.float32)
        return feats if len(feats) == 1582 else None
    except Exception as _e2:
        logger.warning(f"extract_features_plain hata: {_e2}")
        return None

# --- SENTENCE MODELS LOADER ---
loaded_sentence_models = {}
for key, paths in SENTENCE_MODEL_CONFIG.items():
    try:
        if os.path.exists(paths['model']):
            if paths['model'].endswith('.h5'):
                if TF_AVAILABLE:
                    model_obj = load_model(paths['model'])
                else:
                    continue
            else:
                model_obj = joblib.load(paths['model'])
                
            loaded_sentence_models[key] = {
                'model': model_obj,
                'scaler': joblib.load(paths['scaler']),
                'encoder': joblib.load(paths['encoder']),
                'weight': paths['weight']
            }
            logger.info(f"[EXPERIMENTAL] {key.upper()} yüklendi.")
    except Exception as e:
        logger.error(f"[EXPERIMENTAL] {key} yüklenemedi: {e}")


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
                    seg['confidence'] = 0.5
                else:
                    feats = feats.reshape(1, -1)
                    feats_scaled = scaler.transform(feats)
                    
                    if hasattr(model, 'predict_proba'):
                        probs = model.predict_proba(feats_scaled)[0]
                        pred_idx = np.argmax(probs)
                        seg['confidence'] = float(np.max(probs))
                    else:
                        pred = model.predict(feats_scaled)[0]
                        pred_idx = int(pred.item()) if isinstance(pred, np.ndarray) else int(pred)
                        seg['confidence'] = 1.0
                        
                    label = encoder.inverse_transform([pred_idx])[0].lower()
                    seg['emotion'] = label
            except Exception as e:
                seg['emotion'] = 'error'
                seg['confidence'] = 0.0
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
        base_keys = ['rf', 'lightgbm', 'xgboost', 'catboost', 'gradient_boosting']
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
        global_feats = extract_features(temp_path)
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

            import librosa
            dur = librosa.get_duration(path=seg_path)

            results.append({
                'segment': os.path.basename(seg_path),
                'emotion': label,
                'confidence': float(conf),
                'all_scores': all_scores_dict,
                'duration': dur
            })
            
        # 6. Aggregation
        if not results:
             return jsonify({'error': 'No valid segments extracted or feature extraction failed.'}), 500
             
        # Majority Vote (Legacy/Reference)
        from collections import Counter
        emotions = [r['emotion'] for r in results]
        most_common = Counter(emotions).most_common(1)[0]
        final_emotion_vote = most_common[0]
        
        # --- NEW: Ensemble Global & Segment Strategies ---
        voted_scores = {}
        weighted_result = processor.weighted_voting(results)
        if weighted_result:
            voted_scores = weighted_result.get('weighted_details', {})
            
        # Global Prediction
        global_scores = {}
        if global_feats is not None:
            global_f_reshaped = global_feats.reshape(1, -1)
            global_f_scaled = scaler.transform(global_f_reshaped)
            
            if selected_model_key in ['cnn1d', 'dnn'] and 'cnn' in selected_model_key:
                global_f_scaled = np.expand_dims(global_f_scaled, axis=2)
                
            probs = None
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(global_f_scaled)[0]
            elif selected_model_key in ['cnn1d', 'dnn', 'cnn1d_robust', 'dnn_robust']:
                probs = model.predict(global_f_scaled, verbose=0)[0]
                
            if probs is not None:
                for i, class_name in enumerate(encoder.classes_):
                    cat = class_name.lower()
                    if cat != 'neutral':
                        global_scores[cat] = float(probs[i]) * 100.0
                t = sum(global_scores.values())
                if t > 0: global_scores = {k: (v/t)*100 for k,v in global_scores.items()}
            else:
                 pred = model.predict(global_f_scaled)[0]
                 pred_idx = int(pred.item()) if isinstance(pred, (list, np.ndarray)) else int(pred)
                 lbl = encoder.inverse_transform([pred_idx])[0].lower()
                 global_scores = {lbl: 100.0}

        # Blending 
        final_combined = {}
        GLOBAL_WEIGHT = 0.60
        SEGMENT_WEIGHT = 0.40
        
        valid_classes = [c.lower() for c in encoder.classes_ if c.lower() != 'neutral']
        if voted_scores and global_scores:
            for emo in valid_classes:
                val_g = global_scores.get(emo, 0.0) * GLOBAL_WEIGHT
                val_s = voted_scores.get(emo, 0.0) * SEGMENT_WEIGHT
                final_combined[emo] = val_g + val_s
        elif voted_scores:
            final_combined = voted_scores
        elif global_scores:
            final_combined = global_scores
            
        if final_combined:
            final_emotion = max(final_combined.items(), key=lambda x: x[1])[0]
            final_conf = final_combined[final_emotion]
            details = final_combined
        else:
            final_emotion = 'unknown'
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

# ==============================================================================
# 🧠 THE MASTERMIND (ÜST AKIL) ENDPOINT
# ==============================================================================
@app.route('/api/predict_mastermind', methods=['POST'])
def predict_mastermind():
    """
    Sıfır parametre. Yalnızca ses dosyası bağla.
    Tüm kalabalıkları ve hatalı modelleri reddeden, F1 skoru en yüksek VOSK 
    ve CatBoost/XGBoost (%60/%40) birleşimini kullanıp, 'Sad' için RF_Robust vetosuna
    başvuran nihai üretim (production) API uç noktasıdır.
    """
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file part'}), 400
        
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    required_models = ['catboost', 'xgboost', 'rf_robust']
    missing = [m for m in required_models if m not in loaded_models]
    if missing:
        return jsonify({'error': f"Gerekli 'Üst Akıl' modelleri eksik: {missing}"}), 500
         
    import tempfile
    import soundfile as sf
    import librosa
    
    temp_path = ""
    segment_paths = []
    
    try:
        temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
        os.close(temp_fd)
        file.save(temp_path)
        
        # 1. GLOBAL ÖZELLİKLERİ ÇIKAR
        global_f = extract_features(temp_path)
        if global_f is None:
            return jsonify({'error': 'Global ses vektörü çıkarılamadı (Aşırı gürültü veya sessiz).'}), 400
            
        y_global, sr = librosa.load(temp_path, sr=22050)
            
        # 2. VOSK İLE KELİMELERİ (SEGMENTLERİ) AYIKLA
        # WhisperX ve VAD testlerde F1 skalasında VOSK'un gerisinde kaldığı için elendi.
        segments_info = transcribe(temp_path, engine="vosk")
        if not segments_info:
             segments_info = [{'start': 0.0, 'end': len(y_global)/sr}]
             
        feats_list = []
        for i, seg in enumerate(segments_info):
            start_s = float(seg.get('start', 0.0))
            end_s = float(seg.get('end', 0.0))
            start_idx = int(start_s * sr)
            end_idx = int((end_s + 0.1) * sr) # +100ms güvenli bant
            
            seg_audio = y_global[start_idx:end_idx]
            
            if len(seg_audio) > int(0.1 * sr):
                seg_path = os.path.join(tempfile.gettempdir(), f"master_seg_{i}_{os.urandom(4).hex()}.wav")
                sf.write(seg_path, seg_audio, sr)
                segment_paths.append(seg_path)
                
                f = extract_features(seg_path)
                if f is not None:
                     feats_list.append(f)
                     
        # 3. 'ÜST AKIL' HESAPLAMA MOTORU (Model Skorlayıcı Fonksiyon)
        def score_model(m_key):
             m_data = loaded_models[m_key]
             m = m_data['model']
             s = m_data['scaler']
             e = m_data['encoder']
             
             # Global Melodi tahmini (%60 gücünde olacak)
             global_f_scaled = s.transform(global_f.reshape(1, -1))
             if hasattr(m, 'predict_proba'):
                 g_probs = m.predict_proba(global_f_scaled)[0]
             else:
                 g_probs = np.zeros(len(e.classes_))
                 pred = int(m.predict(global_f_scaled)[0])
                 g_probs[list(e.transform(e.classes_)).index(pred)] = 1.0
             
             # Kelime ortalama tahmini (%40 gücünde olacak)
             seg_probs_sum = np.zeros(len(e.classes_))
             if feats_list:
                 for sf_vec in feats_list:
                     sf_scaled = s.transform(sf_vec.reshape(1, -1))
                     if hasattr(m, 'predict_proba'):
                         p = m.predict_proba(sf_scaled)[0]
                     else:
                         p = np.zeros(len(e.classes_))
                         pred = int(m.predict(sf_scaled)[0])
                         p[list(e.transform(e.classes_)).index(pred)] = 1.0
                     seg_probs_sum += np.array(p)
                 seg_probs_avg = seg_probs_sum / len(feats_list)
             else:
                 seg_probs_avg = np.array(g_probs) # Eğer kelime bulamadıysa (sadece sessizlikse)
                 
             # 60/40 Altın Oran Harmanlaması (Global Melodi + Saf Kelime Duygusu)
             # Not: Testlerde en yüksek skoru bu sabit oran vermiştir. Dinamik eşikler elenmiştir.
             combined = 0.60 * np.array(g_probs) + 0.40 * seg_probs_avg
             
             # Sınıf isimlerine göre eşleştirme yapıp 100 üzerinden normalize et
             scores = {}
             valid_classes = [c.lower() for c in e.classes_ if c.lower() != 'neutral']
             for i, class_name in enumerate(e.classes_):
                 cat = class_name.lower()
                 if cat in valid_classes:
                      scores[cat] = float(combined[i]) * 100.0
             
             # Toplamı %100'e kilitle
             total_s = sum(scores.values())
             if total_s > 0: 
                 scores = {k: (v/total_s)*100.0 for k,v in scores.items()}
             return scores
             
        # CatBoost ve XGBoost'u yarıştırıp ortak bir kanaat (ortalama) oluşturuyoruz.
        # Neden ikisi? Çünkü Angry ve Happy testlerinde zirveyi onlar paylaşıyor.
        cb_scores = score_model('catboost')
        xgb_scores = score_model('xgboost')
        
        main_scores = {}
        for k in cb_scores.keys():
            main_scores[k] = (cb_scores[k] + xgb_scores[k]) / 2.0
            
        primary_emotion = max(main_scores.items(), key=lambda x: x[1])[0]
        primary_conf = main_scores[primary_emotion]
        
        # 4. RF_ROBUST Sad Detector (VETO SİSTEMİ)
        # Testlerde SAD'ı bozmadan %55 orana ulaştıran tek kahraman buydu.
        rf_scores = score_model('rf_robust')
        sad_val_from_rf = rf_scores.get('sad', 0.0)
        rf_primary_emotion = max(rf_scores.items(), key=lambda x: x[1])[0] if rf_scores else 'unknown'
        
        final_emotion = primary_emotion
        final_conf = primary_conf
        is_vetoed = False
        
        # KURAL 1: EĞER ANA MODELLER "SAD" DİYORSA, AMA RF ONAYLAMIYORSA (<30), BUNU REDDET!
        # Çünkü ana modeller (CatBoost/XGB) Calm veya nefes sesinde yanılıp Sad diyebiliyor. 
        # Sad kelimesi RF_Robust'un tekelindedir. RF onaylamıyorsa, ana modelin ikinci duygusuna geçilir.
        if primary_emotion == 'sad' and sad_val_from_rf < 30.0:
            scores_without_sad = {k: v for k, v in main_scores.items() if k != 'sad'}
            second_emotion = max(scores_without_sad.items(), key=lambda x: x[1])[0]
            final_emotion = second_emotion
            final_conf = scores_without_sad[second_emotion]
            # is_vetoed = False kalır, burası Veto değil, de-hallucination işlemidir.
            
        # KURAL 2: MASTERMIND VETO (RF_Robust "Sad" barajını aştıysa, ana model ne derse desin ezer)
        elif primary_emotion != 'sad' and sad_val_from_rf >= 30.0:
            final_emotion = 'sad'
            final_conf = sad_val_from_rf
            is_vetoed = True
            
        # HIZLI SEGMENT (KELİME) DUYGU İNCELEMESİ
        # Mastermind'ın timeline (zaman çizelgesi) için kelimelerin duyguları gerekiyor.
        word_timestamps = _predict_emotions_for_segments(temp_path, segments_info, start_key='start', end_key='end', model_key='catboost')

        # Frontend için görsel sonuç formatı
        return jsonify({
             'final_emotion': final_emotion,
             'confidence': f"%{final_conf:.2f}",
             'veto_applied': is_vetoed,
             'rf_sad_score': f"%{sad_val_from_rf:.2f}",
             'main_scores': main_scores,
             'segments_analyzed': len(feats_list),
             'word_timestamps': word_timestamps
        })
        
    except Exception as e:
        logger.error(f"Mastermind Error: {e}")
        return jsonify({'error': str(e)}), 500
        
    finally:
        if temp_path and os.path.exists(temp_path):
             os.remove(temp_path)
        for p in segment_paths:
             if os.path.exists(p): os.remove(p)

# ==============================================================================
# EXPERIMENTAL ENDPOINT: WHOLE SENTENCE ANALYSIS
# ==============================================================================
@app.route('/api/predict_sentence_whole', methods=['POST'])
def predict_sentence_whole():
    """
    Experimental endpoint for high-accuracy whole sentence analysis.
    Applies per-speaker calibration (Z-Normalization) before inference.
    """
    temp_path = None
    try:
        # 1. Ensemble Initalization
        if not loaded_sentence_models:
             return jsonify({'error': 'Cümle modelleri yüklü değil'}), 500

        # 2. File Handling
        if 'file' not in request.files:
            return jsonify({'error': 'Ses dosyası yok'}), 400
        
        import uuid
        temp_path = os.path.join(BASE_DIR, f"temp_ensemble_{uuid.uuid4().hex[:8]}.wav")
        request.files['file'].save(temp_path)

        # 3. Feature Extraction & Denoising
        import opensmile
        import noisereduce as nr
        import soundfile as sf
        
        audio_data, rate = sf.read(temp_path)
        audio_cleaned = nr.reduce_noise(y=audio_data, sr=rate, prop_decrease=0.7)
        
        smile_local = opensmile.Smile(
            feature_set=opensmile.FeatureSet.IS10,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        df_feats = smile_local.process_signal(audio_cleaned, rate)
        features_raw = df_feats.to_numpy().flatten() # 1582 features

        # 4. SPEAKER NORMALIZATION (Calibration)
        features = features_raw.copy()
        if os.path.exists(CALIBRATION_FILE):
            profile = joblib.load(CALIBRATION_FILE)
            mean_vals = profile['mean']
            std_vals = profile['std']
            if len(features) == len(mean_vals):
                features = (features - mean_vals) / (std_vals + 1e-6)
                logger.info("Experimental Ensemble Normalization applied.")

        # 5. ENSEMBLE PREDICTION (Weighted Logic)
        weighted_probabilities = np.zeros(4) # angry, calm, happy, sad
        total_weight = 0
        model_details = []
        
        # Hardcoded classes order for consistency in blending if encoders differ
        # But we assume all encoders use ['angry', 'calm', 'happy', 'sad'] alphabetical
        EMOTION_ORDER = ['angry', 'calm', 'happy', 'sad']
        
        for m_key, tools in loaded_sentence_models.items():
            try:
                m_obj = tools['model']
                m_scaler = tools['scaler']
                m_encoder = tools['encoder']
                m_weight = tools['weight']
                
                # Scale & Predict
                f_scaled = m_scaler.transform(features.reshape(1, -1))
                p_raw = m_obj.predict_proba(f_scaled)[0]
                
                # Map probabilities to standard EMOTION_ORDER
                p_mapped = np.zeros(4)
                m_scores = {}
                for i, cls in enumerate(m_encoder.classes_):
                    if cls.lower() in EMOTION_ORDER:
                         idx = EMOTION_ORDER.index(cls.lower())
                         p_mapped[idx] = p_raw[i]

                # Normalize p_mapped to 1.0 (Ignore 'neutral' or other classes for the ensemble)
                m_sum = np.sum(p_mapped)
                if m_sum > 0:
                    p_mapped = p_mapped / m_sum
                
                for i, emo in enumerate(EMOTION_ORDER):
                    m_scores[emo] = float(p_mapped[i] * 100)
                
                # Apply weight
                weighted_probabilities += (p_mapped * m_weight)
                total_weight += m_weight
                
                # Save for UI
                p_max_idx = np.argmax(p_mapped)
                model_details.append({
                    'model': m_key.replace('_sentence', '').upper(),
                    'prediction': EMOTION_ORDER[p_max_idx],
                    'confidence': float(np.max(p_mapped) * 100),
                    'weight': m_weight,
                    'scores': m_scores # NEW: Full breakdown
                })
            except Exception as me:
                logger.error(f"Error in model {m_key}: {me}")

        # Final Blending
        if total_weight > 0:
            final_probs = weighted_probabilities / np.sum(weighted_probabilities) if np.sum(weighted_probabilities) > 0 else weighted_probabilities / total_weight
        else:
            final_probs = weighted_probabilities # Fallback

        # Final Normalization to ensure 100% total
        f_sum = np.sum(final_probs)
        if f_sum > 0:
            final_probs = final_probs / f_sum
            
        prediction_index = np.argmax(final_probs)
        final_emotion = EMOTION_ORDER[prediction_index]
        final_confidence = np.max(final_probs) * 100
        
        all_scores = {}
        for i, emo in enumerate(EMOTION_ORDER):
            all_scores[emo] = float(final_probs[i] * 100)

        # 6. Frequency/Pitch Extraction (Visual Feedback)
        pitch_data = []
        try:
            # Extract pitch using librosa
            # Using piptrack for speed or spectral_centroid for envelope
            S = np.abs(librosa.stft(audio_cleaned))
            pitches, magnitudes = librosa.piptrack(S=S, sr=rate)
            
            # Select most dominant pitch per frame
            pitch_vals = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_vals.append(float(pitch))
                else:
                    # Keep zeros as silence or previous value to prevent spikes
                    pitch_vals.append(0.0)
            
            # Resample to 100 points for frontend performance
            total_points = len(pitch_vals)
            if total_points > 100:
                indices = np.linspace(0, total_points - 1, 100).astype(int)
                pitch_vals = [pitch_vals[i] for i in indices]
            
            for i, p in enumerate(pitch_vals):
                pitch_data.append({"time": i, "freq": p})
                
        except Exception as p_err:
            logger.warning(f"Pitch extraction error: {p_err}")

        # 7. STT Integration
        stt_engine = request.form.get('stt_engine', 'vosk')
        word_timestamps = []
        try:
            stt_words = transcribe(temp_path, engine=stt_engine)
            for w in stt_words:
                word_timestamps.append({
                    'word': w['word'],
                    'start': w['start'],
                    'end': w['end'],
                    'emotion': final_emotion,
                    'confidence': float(final_confidence) / 100.0
                })
        except: pass

        return jsonify({
            'emotion': final_emotion,
            'confidence': f"%{final_confidence:.2f}",
            'all_scores': all_scores,
            'model_used': "WHOLE_SENTENCE_ENSEMBLE (10-Models)",
            'word_timestamps': word_timestamps,
            'frequency_data': pitch_data, # NEW
            'model_details': model_details # For VotingDetails component
        })

    except Exception as e:
        logger.error(f"Experimental Predict Error: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

# ==============================================================================
# ADVANCED SENTENCE ANALYSIS — Models_2 (noise-augmented, plain IS10)
# ==============================================================================
@app.route('/api/predict_advanced_sentence', methods=['POST'])
def predict_advanced_sentence():
    """
    VOSK kelime ayrıştırma + Models_2'nin 5 modeli ile her kelime için duygu tahmini.
    Global ses (%60) + kelime ortalaması (%40) harmanlaması, tüm 5 modelin eşit oylaması.
    """
    if not loaded_models_v2:
        return jsonify({'error': 'Models_2 modelleri yüklü değil'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'Ses dosyası yok'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'}), 400

    import tempfile, soundfile as sf
    temp_path = ""
    segment_paths = []

    try:
        temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
        os.close(temp_fd)
        file.save(temp_path)

        # 1. Global IS10 özellik çıkar (1582 boyut, temiz)
        global_f = extract_features_plain(temp_path)
        if global_f is None:
            return jsonify({'error': 'Global ses vektörü çıkarılamadı'}), 400

        y_global, sr = librosa.load(temp_path, sr=22050)

        # 2. VOSK ile kelimeleri ayıkla
        segments_info = transcribe(temp_path, engine="vosk")
        if not segments_info:
            segments_info = [{'start': 0.0, 'end': len(y_global) / sr}]

        # 3. Her kelime segmenti için IS10 özellik çıkar
        feats_list = []
        for i, seg in enumerate(segments_info):
            start_s = float(seg.get('start', 0.0))
            end_s = float(seg.get('end', 0.0))
            start_idx = int(start_s * sr)
            end_idx = int((end_s + 0.1) * sr)
            seg_audio = y_global[start_idx:end_idx]
            if len(seg_audio) > int(0.1 * sr):
                seg_path = os.path.join(tempfile.gettempdir(), f"adv_seg_{i}_{os.urandom(4).hex()}.wav")
                sf.write(seg_path, seg_audio, sr)
                segment_paths.append(seg_path)
                f = extract_features_plain(seg_path)
                if f is not None:
                    feats_list.append(f)

        # 4. Tüm 5 Models_2 modelini eşit oylamayla harmanlayan skorlayıcı
        EMOTION_ORDER = ['angry', 'calm', 'happy', 'sad']

        MODEL_DISPLAY_NAMES = {
            'rf_v2':   'Random Forest (V2)',
            'lgbm_v2': 'LightGBM (V2)',
            'xgb_v2':  'XGBoost (V2)',
            'cb_v2':   'CatBoost (V2)',
            'gb_v2':   'Gradient Boost (V2)',
        }

        def score_v2_ensemble(global_feat, seg_feats):
            """
            Models_2 (5 model) harmanlaması. 
            Test sonuçlarına göre en iyi blend oranı: LightGBM %35, diğer 4 model %16.25.
            """
            ensemble_probs = np.zeros(4)
            model_count = 0
            per_model_details = []

            # En iyi test sonuçlarından gelen ağırlıklar (Test3/Results/blend_ratio_results.txt)
            MODEL_WEIGHTS = {
                'lgbm_v2': 0.35,
                'rf_v2':   0.1625,
                'xgb_v2':  0.1625,
                'cb_v2':   0.1625,
                'gb_v2':   0.1625,
            }

            for m_key, tools in loaded_models_v2.items():
                m = tools['model']
                s = tools['scaler']
                e = tools['encoder']
                w = MODEL_WEIGHTS.get(m_key, 0.20) # Varsayılan 0.20 (eğer model bulunamazsa)

                # Global tahmin
                gf_scaled = s.transform(global_feat.reshape(1, -1))
                g_probs_raw = m.predict_proba(gf_scaled)[0]

                g_mapped = np.zeros(4)
                for idx, cls in enumerate(e.classes_):
                    if cls.lower() in EMOTION_ORDER:
                        g_mapped[EMOTION_ORDER.index(cls.lower())] = g_probs_raw[idx]

                # Segment ortalaması
                if seg_feats:
                    seg_sum = np.zeros(4)
                    for sf_vec in seg_feats:
                        sf_scaled = s.transform(sf_vec.reshape(1, -1))
                        p_raw = m.predict_proba(sf_scaled)[0]
                        p_mapped = np.zeros(4)
                        for idx, cls in enumerate(e.classes_):
                            if cls.lower() in EMOTION_ORDER:
                                p_mapped[EMOTION_ORDER.index(cls.lower())] = p_raw[idx]
                        seg_sum += p_mapped
                    seg_avg = seg_sum / len(seg_feats)
                else:
                    seg_avg = g_mapped.copy()

                # 10/90 Global/Kelime harmanlaması (Test3/Results/global_segment_ratio_results.txt - En İyi Oran)
                combined = 0.10 * g_mapped + 0.90 * seg_avg
                c_sum = np.sum(combined)
                if c_sum > 0:
                    combined = combined / c_sum

                m_pred_idx = int(np.argmax(combined))
                per_model_details.append({
                    'model': MODEL_DISPLAY_NAMES.get(m_key, m_key.upper()),
                    'prediction': EMOTION_ORDER[m_pred_idx],
                    'confidence': float(np.max(combined) * 100),
                    'weight': float(w),
                    'scores': {emo: float(combined[i] * 100) for i, emo in enumerate(EMOTION_ORDER)},
                })

                # Ağırlıklı toplama
                ensemble_probs += (combined * w)
                model_count += 1

            # Ağırlıklar toplamı zaten ~1.0 olmalı ama garantiye alalım
            total = np.sum(ensemble_probs)
            if total > 0:
                ensemble_probs /= total
            
            final = {emo: float(ensemble_probs[i]) * 100.0 for i, emo in enumerate(EMOTION_ORDER)}
            return final, per_model_details

        final_scores, model_details = score_v2_ensemble(global_f, feats_list)
        final_emotion = max(final_scores.items(), key=lambda x: x[1])[0]
        final_conf = final_scores[final_emotion]

        # 5. Kelime bazlı duygu zaman damgaları (her kelime için tek model tahmininde hız öncelikli: lgbm_v2)
        word_timestamps = []
        word_feat_idx = 0
        for i, seg in enumerate(segments_info):
            start_s = float(seg.get('start', 0.0))
            end_s = float(seg.get('end', 0.0))
            word_text = seg.get('word', f'kelime_{i+1}')
            word_emotion = final_emotion
            word_conf = final_conf / 100.0

            if word_feat_idx < len(feats_list):
                wf = feats_list[word_feat_idx]
                word_feat_idx += 1
                # Hızlı: LightGBM_v2 ile kelime tahmini
                if 'lgbm_v2' in loaded_models_v2:
                    tools = loaded_models_v2['lgbm_v2']
                    wf_scaled = tools['scaler'].transform(wf.reshape(1, -1))
                    wp_raw = tools['model'].predict_proba(wf_scaled)[0]
                    w_mapped = np.zeros(4)
                    for idx, cls in enumerate(tools['encoder'].classes_):
                        if cls.lower() in EMOTION_ORDER:
                            w_mapped[EMOTION_ORDER.index(cls.lower())] = wp_raw[idx]
                    word_emotion = EMOTION_ORDER[int(np.argmax(w_mapped))]
                    word_conf = float(np.max(w_mapped))

            word_timestamps.append({
                'word': word_text,
                'start': start_s,
                'end': end_s,
                'emotion': word_emotion,
                'confidence': word_conf
            })

        return jsonify({
            'emotion': final_emotion,
            'confidence': f"%{final_conf:.2f}",
            'all_scores': final_scores,
            'word_timestamps': word_timestamps,
            'model_details': model_details,
            'model_used': 'ADVANCED_SENTENCE (Models_2, 5-Model Ensemble)',
            'segments_analyzed': len(feats_list),
        })

    except Exception as e:
        logger.error(f"Advanced Sentence Error: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        for p in segment_paths:
            if os.path.exists(p):
                os.remove(p)
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/analyze_hubert', methods=['POST'])
def analyze_hubert():
    """
    Huggingface SeaBenSea/HuBERT model endpoint.
    Directly predicts emotion from audio waveform using HubertForSequenceClassification.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'Ses dosyası yok'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'}), 400

    predictor = get_hubert_predictor()
    if predictor is None:
        return jsonify({'error': 'HuBERT modeli yüklenemedi. Bağımlılıkları kontrol edin.'}), 500

    import uuid
    temp_path = os.path.join(BASE_DIR, f"temp_hubert_{uuid.uuid4().hex[:8]}.wav")
    file.save(temp_path)

    try:
        import time
        start_time = time.time()
        
        result = predictor.predict(temp_path)
        
        # Standart skor formatına dönüştür (% bazlı)
        all_scores = {res['label']: res['score'] * 100 for res in result['scores']}
        
        elapsed = round(time.time() - start_time, 2)
        
        return jsonify({
            'emotion': result['emotion'],
            'confidence': f"%{all_scores.get(result['emotion'], 0):.2f}",
            'all_scores': all_scores,
            'model_used': 'SeaBenSea / HuBERT (HuggingFace)',
            'elapsed_seconds': elapsed
        })
    except Exception as e:
        logger.error(f"HuBERT Inference Error: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/analyze_wav2vec2_turkish', methods=['POST'])
def analyze_wav2vec2_turkish():
    """
    Sefa-Alper/wav2vec2-xlsr-turkish-speech-emotion-recognition-v3 endpoint.
    Turkish speech sentiment/emotion: NEGATIVE, NEUTRAL, POSITIVE.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'Ses dosyası yok'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'}), 400

    predictor = get_wav2vec2_turkish_predictor()
    if predictor is None:
        return jsonify({'error': 'Wav2Vec2 Turkish modeli yüklenemedi. Bağımlılıkları kontrol edin.'}), 500

    import uuid
    temp_path = os.path.join(BASE_DIR, f"temp_wav2vec2_{uuid.uuid4().hex[:8]}.wav")
    file.save(temp_path)

    try:
        import time
        start_time = time.time()
        result = predictor.predict(temp_path)
        all_scores = {res['label']: res['score'] * 100 for res in result['scores']}
        elapsed = round(time.time() - start_time, 2)
        return jsonify({
            'emotion': result['emotion'],
            'confidence': f"%{all_scores.get(result['emotion'], 0):.2f}",
            'all_scores': all_scores,
            'model_used': 'Sefa-Alper / Wav2Vec2 Turkish (HuggingFace)',
            'elapsed_seconds': elapsed
        })
    except Exception as e:
        logger.error(f"Wav2Vec2 Turkish Inference Error: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/analyze_exhubert', methods=['POST'])
def analyze_exhubert():
    """
    amiriparian/ExHuBERT endpoint.
    6-class arousal-valence emotion: disgust, neutral, kind, angry, surprised, happy.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'Ses dosyası yok'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'}), 400

    predictor = get_exhubert_predictor()
    if predictor is None:
        return jsonify({'error': 'ExHuBERT modeli yüklenemedi.'}), 500

    import uuid
    temp_path = os.path.join(BASE_DIR, f"temp_exhubert_{uuid.uuid4().hex[:8]}.wav")
    file.save(temp_path)

    try:
        import time
        start_time = time.time()
        result = predictor.predict(temp_path)
        all_scores = {res['label']: res['score'] * 100 for res in result['scores']}
        elapsed = round(time.time() - start_time, 2)
        return jsonify({
            'emotion': result['emotion'],
            'confidence': f"%{all_scores.get(result['emotion'], 0):.2f}",
            'all_scores': all_scores,
            'model_used': 'amiriparian / ExHuBERT (HuggingFace)',
            'elapsed_seconds': elapsed
        })
    except Exception as e:
        logger.error(f"ExHuBERT Inference Error: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/analyze_sensevoice', methods=['POST'])
def analyze_sensevoice():
    """
    FunAudioLLM/SenseVoiceSmall endpoint (Alibaba).
    Multilingual speech emotion recognition with 7 emotion classes.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'Ses dosyası yok'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'}), 400

    predictor = get_sensevoice_predictor()
    if predictor is None:
        detail = _sensevoice_load_error or 'funasr paketi kurulu mu kontrol edin.'
        return jsonify({'error': f'SenseVoice yüklenemedi: {detail}'}), 500

    import uuid
    temp_path = os.path.join(BASE_DIR, f"temp_sensevoice_{uuid.uuid4().hex[:8]}.wav")
    file.save(temp_path)

    try:
        import time
        start_time = time.time()
        result = predictor.predict(temp_path)
        all_scores = {res['label']: res['score'] * 100 for res in result['scores']}
        elapsed = round(time.time() - start_time, 2)
        return jsonify({
            'emotion': result['emotion'],
            'confidence': f"%{all_scores.get(result['emotion'], 0):.2f}",
            'all_scores': all_scores,
            'model_used': 'FunAudioLLM / SenseVoiceSmall (Alibaba)',
            'elapsed_seconds': elapsed,
            'raw_text': result.get('raw_text', '')
        })
    except Exception as e:
        logger.error(f"SenseVoice Inference Error: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/analyze_wavlm', methods=['POST'])
def analyze_wavlm():
    """
    3loi/SER-Odyssey-Baseline-WavLM-Categorical endpoint.
    8-class emotion: Angry, Sad, Happy, Surprise, Fear, Disgust, Contempt, Neutral.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'Ses dosyası yok'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'}), 400

    predictor = get_wavlm_predictor()
    if predictor is None:
        return jsonify({'error': 'WavLM modeli yüklenemedi.'}), 500

    import uuid
    temp_path = os.path.join(BASE_DIR, f"temp_wavlm_{uuid.uuid4().hex[:8]}.wav")
    file.save(temp_path)

    try:
        import time
        start_time = time.time()
        result = predictor.predict(temp_path)
        all_scores = {res['label']: res['score'] * 100 for res in result['scores']}
        elapsed = round(time.time() - start_time, 2)
        return jsonify({
            'emotion': result['emotion'],
            'confidence': f"%{all_scores.get(result['emotion'], 0):.2f}",
            'all_scores': all_scores,
            'model_used': '3loi / WavLM-Categorical (HuggingFace)',
            'elapsed_seconds': elapsed
        })
    except Exception as e:
        logger.error(f"WavLM Inference Error: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/analyze_xlsr', methods=['POST'])
def analyze_xlsr():
    """
    ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition endpoint.
    8-class multilingual emotion: angry, calm, disgust, fearful, happy, neutral, sad, surprised.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'Ses dosyası yok'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'}), 400

    predictor = get_xlsr_predictor()
    if predictor is None:
        return jsonify({'error': 'XLSR modeli yüklenemedi.'}), 500

    import uuid
    temp_path = os.path.join(BASE_DIR, f"temp_xlsr_{uuid.uuid4().hex[:8]}.wav")
    file.save(temp_path)

    try:
        import time
        start_time = time.time()
        result = predictor.predict(temp_path)
        all_scores = {res['label']: res['score'] * 100 for res in result['scores']}
        elapsed = round(time.time() - start_time, 2)
        return jsonify({
            'emotion': result['emotion'],
            'confidence': f"%{all_scores.get(result['emotion'], 0):.2f}",
            'all_scores': all_scores,
            'model_used': 'ehcalabres / XLSR (HuggingFace)',
            'elapsed_seconds': elapsed
        })
    except Exception as e:
        logger.error(f"XLSR Inference Error: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/analyze_qwen2_audio', methods=['POST'])
def analyze_qwen2_audio():
    """
    Qwen/Qwen2-Audio-7B-Instruct endpoint.
    LLM-based speech emotion recognition.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'Ses dosyası yok'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'}), 400

    predictor = get_qwen2_audio_predictor()
    if predictor is None:
        return jsonify({'error': 'Qwen2-Audio modeli yüklenemedi. VRAM yetersiz olabilir.'}), 500

    import uuid
    temp_path = os.path.join(BASE_DIR, f"temp_qwen_{uuid.uuid4().hex[:8]}.wav")
    file.save(temp_path)

    try:
        import time
        start_time = time.time()
        result = predictor.predict(temp_path)
        all_scores = {res['label']: res['score'] * 100 for res in result['scores']}
        elapsed = round(time.time() - start_time, 2)
        return jsonify({
            'emotion': result['emotion'],
            'confidence': f"%{all_scores.get(result['emotion'], 0):.2f}",
            'all_scores': all_scores,
            'model_used': 'Qwen / Qwen2-Audio-7B (HuggingFace)',
            'elapsed_seconds': elapsed,
            'raw_text': result.get('raw_text', '')
        })
    except Exception as e:
        logger.error(f"Qwen2-Audio Inference Error: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/analyze_wavlm_base_plus', methods=['POST'])
def analyze_wavlm_base_plus():
    """
    harritaylor/wavlm-base-plus-speech-emotion-recognition endpoint.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'Ses dosyası yok'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'}), 400

    predictor = get_wavlm_base_plus_predictor()
    if predictor is None:
        return jsonify({'error': 'WavLM Base Plus modeli yüklenemedi.'}), 500

    import uuid
    temp_path = os.path.join(BASE_DIR, f"temp_wavlm_bp_{uuid.uuid4().hex[:8]}.wav")
    file.save(temp_path)

    try:
        import time
        start_time = time.time()
        result = predictor.predict(temp_path)
        all_scores = {res['label']: res['score'] * 100 for res in result['scores']}
        elapsed = round(time.time() - start_time, 2)
        return jsonify({
            'emotion': result['emotion'],
            'confidence': f"%{all_scores.get(result['emotion'], 0):.2f}",
            'all_scores': all_scores,
            'model_used': 'harritaylor / WavLM Base Plus (HuggingFace)',
            'elapsed_seconds': elapsed
        })
    except Exception as e:
        logger.error(f"WavLM Base Plus Inference Error: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/analyze_wav2vec2_english', methods=['POST'])
def analyze_wav2vec2_english():
    """
    r-f/wav2vec-english-speech-emotion-recognition endpoint.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'Ses dosyası yok'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'}), 400

    predictor = get_wav2vec2_english_predictor()
    if predictor is None:
        return jsonify({'error': 'Wav2Vec2 English modeli yüklenemedi.'}), 500

    import uuid
    temp_path = os.path.join(BASE_DIR, f"temp_w2v2_en_{uuid.uuid4().hex[:8]}.wav")
    file.save(temp_path)

    try:
        import time
        start_time = time.time()
        result = predictor.predict(temp_path)
        all_scores = {res['label']: res['score'] * 100 for res in result['scores']}
        elapsed = round(time.time() - start_time, 2)
        return jsonify({
            'emotion': result['emotion'],
            'confidence': f"%{all_scores.get(result['emotion'], 0):.2f}",
            'all_scores': all_scores,
            'model_used': 'r-f / Wav2Vec2 English (HuggingFace)',
            'elapsed_seconds': elapsed
        })
    except Exception as e:
        logger.error(f"Wav2Vec2 English Inference Error: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == '__main__':
    # use_reloader=False ekleyerek klasördeki dosya değişikliklerinin (veya yeni kütüphane yüklemelerinin)
    # sunucuyu aniden resetlemesini engelliyoruz.
    app.run(debug=True, use_reloader=False, port=5000)