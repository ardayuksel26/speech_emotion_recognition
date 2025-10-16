#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Analizi - Hangi özelliklerin ne anlama geldiğini gösterir
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def analyze_features():
    """Feature'ları analiz et ve en önemlilerini göster"""
    print("FEATURE ANALIZI - HANGI OZELLIKLER NE ANLAMA GELIYOR")
    print("="*60)
    
    # Veri yukleme
    emotions = ['angry', 'calm', 'happy', 'sad']
    dataframes = []
    
    for emotion in emotions:
        df = pd.read_csv(f'Extracted_CSV/{emotion}.csv')
        df['emotion'] = emotion
        dataframes.append(df)
    
    data = pd.concat(dataframes, ignore_index=True)
    
    # Veri on isleme
    X = data.drop('emotion', axis=1)
    y = data['emotion']
    
    # String sutunlari kaldir
    string_columns = X.select_dtypes(include=['object']).columns
    X = X.drop(columns=string_columns)
    
    # ID sutununu kaldir
    if 'id' in X.columns:
        X = X.drop('id', axis=1)
    
    # Label encoding
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    # Veri bolumleme
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalizasyon
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Random Forest modeli egit
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Feature onemlerini al
    feature_importance = rf_model.feature_importances_
    feature_names = X.columns
    
    # En onemli 20 feature'i bul
    top_20_indices = np.argsort(feature_importance)[-20:]
    top_20_features = feature_names[top_20_indices]
    top_20_importance = feature_importance[top_20_indices]
    
    print("\nEN ONEMLI 20 OZELLIK:")
    print("-" * 60)
    
    for i, (feature, importance) in enumerate(zip(top_20_features, top_20_importance)):
        print(f"{i+1:2d}. {feature:<50} | Onem: {importance:.4f}")
    
    print("\n" + "="*60)
    print("OZELLIK TURLERI VE ANLAMLARI:")
    print("="*60)
    
    # Feature kategorilerini analiz et
    categories = {
        'PCM Loudness (Ses Yuksekligi)': [f for f in top_20_features if 'pcm_loudness' in f],
        'MFCC (Ses Tini)': [f for f in top_20_features if 'mfcc' in f],
        'Spektral Ozellikler': [f for f in top_20_features if any(x in f for x in ['spectral', 'fftMag', 'logMel'])],
        'Prosodik Ozellikler': [f for f in top_20_features if any(x in f for x in ['F0', 'pitch', 'voicing'])],
        'Zaman Domeni': [f for f in top_20_features if any(x in f for x in ['zcr', 'rms', 'energy'])],
        'Diger': [f for f in top_20_features if not any(x in f for x in ['pcm_loudness', 'mfcc', 'spectral', 'fftMag', 'logMel', 'F0', 'pitch', 'voicing', 'zcr', 'rms', 'energy'])]
    }
    
    for category, features in categories.items():
        if features:
            print(f"\n{category}:")
            for feature in features:
                idx = np.where(feature_names == feature)[0][0]
                importance = feature_importance[idx]
                print(f"  - {feature} (Onem: {importance:.4f})")
    
    print("\n" + "="*60)
    print("OZELLIK ACIKLAMALARI:")
    print("="*60)
    
    # Detayli aciklamalar
    explanations = {
        'pcm_loudness': 'Ses yuksekligi - konusmanin ne kadar yuksek oldugu',
        'mfcc': 'Mel-frekans cepstral katsayilari - sesin tinisi ve spektral zarf',
        'spectral_centroid': 'Spektral merkez - sesin parlakligi',
        'spectral_rolloff': 'Spektral rolloff - spektrumun yogunlugunun %85\'i',
        'spectral_flatness': 'Spektral duzluk - sesin gurultu benzeri olup olmadigi',
        'F0': 'Temel frekans - sesin perdesi (pitch)',
        'voicing': 'Sesli/sessiz orani - konusmanin ne kadar sesli oldugu',
        'zcr': 'Sifir gecis orani - sesin ne kadar hizli degistigi',
        'rms': 'Kok ortalama kare - sesin enerjisi',
        'energy': 'Enerji - sesin gucu'
    }
    
    for key, explanation in explanations.items():
        matching_features = [f for f in top_20_features if key in f.lower()]
        if matching_features:
            print(f"\n{key.upper()}: {explanation}")
            for feature in matching_features:
                idx = np.where(feature_names == feature)[0][0]
                importance = feature_importance[idx]
                print(f"  - {feature} (Onem: {importance:.4f})")
    
    return top_20_features, top_20_importance

if __name__ == "__main__":
    analyze_features()
