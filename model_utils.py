#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model yükleme ve kaydetme yardımcı fonksiyonları
"""

import pickle
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

def load_trained_model(model_path='models/best_emotion_model.pkl'):
    """Eğitilmiş modeli yükle"""
    if not os.path.exists(model_path):
        print(f"HATA: Model dosyası bulunamadı: {model_path}")
        return None
    
    try:
        with open(model_path, 'rb') as f:
            model_package = pickle.load(f)
        
        print(f"Model başarıyla yüklendi: {model_package['best_model_name']}")
        print(f"Model doğruluğu: {model_package['accuracy']:.4f}")
        
        return model_package
    except Exception as e:
        print(f"HATA: Model yüklenemedi: {str(e)}")
        return None

def predict_with_loaded_model(model_package, features):
    """Yüklenen model ile tahmin yap"""
    if model_package is None:
        return None
    
    try:
        # Özellikleri normalize et
        features_scaled = model_package['scaler'].transform([features])
        
        # Tahmin yap
        model = model_package['model']
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Etiket çevir
        emotion = model_package['label_encoder'].inverse_transform([prediction])[0]
        
        return {
            'emotion': emotion,
            'confidence': max(probabilities),
            'all_probabilities': dict(zip(model_package['label_encoder'].classes_, probabilities)),
            'model_name': model_package['best_model_name'],
            'model_accuracy': model_package['accuracy']
        }
    except Exception as e:
        print(f"HATA: Tahmin yapılamadı: {str(e)}")
        return None

def get_model_info(model_path='models/best_emotion_model.pkl'):
    """Model bilgilerini al"""
    model_package = load_trained_model(model_path)
    if model_package is None:
        return None
    
    return {
        'model_name': model_package['best_model_name'],
        'accuracy': model_package['accuracy'],
        'cv_mean': model_package['cv_mean'],
        'cv_std': model_package['cv_std'],
        'all_results': model_package.get('all_results', {})
    }

def check_model_exists(model_path='models/best_emotion_model.pkl'):
    """Model dosyasının var olup olmadığını kontrol et"""
    return os.path.exists(model_path)