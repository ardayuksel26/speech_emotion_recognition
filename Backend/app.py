import os
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from preprocessing import extract_features 

# TensorFlow imports for DL models
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    print("⚠️ TensorFlow not available. DL models (CNN, DNN) will be disabled.")
    TF_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# --- AYARLAR ---
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
    'xgboost': {
        'model': os.path.join(MODELS_DIR, 'XGBoost/xgboost_model.pkl'),
        'scaler': os.path.join(MODELS_DIR, 'XGBoost/scaler_xgb.pkl'),
        'encoder': os.path.join(MODELS_DIR, 'XGBoost/label_encoder_xgb.pkl')
    },
    'lightgbm': {
        'model': os.path.join(MODELS_DIR, 'LightGBM/lightgbm_model.pkl'),
        'scaler': os.path.join(MODELS_DIR, 'LightGBM/scaler_lgb.pkl'),
        'encoder': os.path.join(MODELS_DIR, 'LightGBM/label_encoder_lgb.pkl')
    },
    'rf': {
        'model': os.path.join(MODELS_DIR, 'Random Forest/random_forest_model.pkl'),
        'scaler': os.path.join(MODELS_DIR, 'Random Forest/scaler_rf.pkl'),
        'encoder': os.path.join(MODELS_DIR, 'Random Forest/label_encoder_rf.pkl')
    },
    'knn': {
        'model': os.path.join(MODELS_DIR, 'KNN/knn_model.pkl'),
        'scaler': os.path.join(MODELS_DIR, 'KNN/scaler_knn.pkl'),
        'encoder': os.path.join(MODELS_DIR, 'KNN/label_encoder_knn.pkl')
    },
    'svm': {
        'model': os.path.join(MODELS_DIR, 'SVM/svm_model.pkl'),
        'scaler': os.path.join(MODELS_DIR, 'SVM/scaler_svm.pkl'),
        'encoder': os.path.join(MODELS_DIR, 'SVM/label_encoder_svm.pkl')
    },
    'mlp': {
        'model': os.path.join(MODELS_DIR, 'MLP/mlp_model.pkl'),
        'scaler': os.path.join(MODELS_DIR, 'MLP/scaler_mlp.pkl'),
        'encoder': os.path.join(MODELS_DIR, 'MLP/label_encoder_mlp.pkl')
    },
    'gradient_boosting': {
        'model': os.path.join(MODELS_DIR, 'GradientBoosting/gradboost_model.pkl'),
        'scaler': os.path.join(MODELS_DIR, 'GradientBoosting/scaler_gb.pkl'),
        'encoder': os.path.join(MODELS_DIR, 'GradientBoosting/label_encoder_gb.pkl')
    },
    # Deep Learning Models
    'dnn': {
        'model': os.path.join(MODELS_DIR, 'DNN/dnn_model.h5'),
        'scaler': os.path.join(MODELS_DIR, 'DNN/scaler_dnn.pkl'),
        'encoder': os.path.join(MODELS_DIR, 'DNN/label_encoder_dnn.pkl')
    },
    'cnn1d': {
        'model': os.path.join(MODELS_DIR, 'CNN1D/cnn1d_model.h5'),
        'scaler': os.path.join(MODELS_DIR, 'CNN1D/scaler_cnn1d.pkl'),
        'encoder': os.path.join(MODELS_DIR, 'CNN1D/label_encoder_cnn1d.pkl')
    }
}

# --- TÜM MODELLERİ YÜKLE (LOADER) ---
loaded_models = {}

print("⏳ Modeller yükleniyor, lütfen bekleyin...")

for key, paths in MODEL_CONFIG.items():
    try:
        # Dosyalar var mı kontrol et
        # .h5 dosyası için özel kontrol yok, load_model halleder ama dosya yoksa hata verir.
        if os.path.exists(paths['model']):
            
            # Model yükleme (PKL vs H5)
            if paths['model'].endswith('.h5'):
                if TF_AVAILABLE:
                    model_obj = load_model(paths['model'])
                else:
                    print(f"   ⚠️ {key.upper()} atlanıyor (TensorFlow yok).")
                    continue
            else:
                model_obj = joblib.load(paths['model'])

            loaded_models[key] = {
                'model': model_obj,
                'scaler': joblib.load(paths['scaler']),
                'encoder': joblib.load(paths['encoder'])
            }
            print(f"   ✅ {key.upper()} yüklendi.")
        else:
            print(f"   ⚠️ {key.upper()} dosyaları bulunamadı, atlanıyor. ({paths['model']})")
    except Exception as e:
        print(f"   ❌ {key.upper()} yüklenirken hata: {e}")

print(f"🚀 Toplam {len(loaded_models)} model kullanıma hazır!")


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

        predicted_label = encoder.inverse_transform([int(prediction_index)])[0]

        # Olasılık (Confidence) ve Tüm Skorlar
        all_scores = {}
        if probabilities is not None:
            # Sınıf isimlerini al
            class_names = encoder.classes_
            
            # Her sınıf için skoru kaydet
            for i, class_name in enumerate(class_names):
                all_scores[class_name] = float(probabilities[i] * 100)
        else:
             # Probability yoksa sadece tahmin edilene 100 ver
             all_scores = {predicted_label: 100.0}

        # Temizlik
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return jsonify({
            'emotion': predicted_label,
            'confidence': f"%{confidence:.2f}",
            'all_scores': all_scores,
            'model_used': selected_model_key.upper()
        })

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': f'Tahmin hatası: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)