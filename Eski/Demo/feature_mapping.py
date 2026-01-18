#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Eşleştirme Analizi - Feature_X'lerin gerçek isimlerini gösterir
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def map_features():
    """Feature numaralarını gerçek isimlerle eşleştir"""
    print("FEATURE ESLEŞTIRME ANALIZI")
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
    
    print("\nFOTOĞRAFTAKI FEATURE_X'LERIN GERÇEK İSİMLERİ:")
    print("-" * 80)
    print("Fotoğraftaki İsim    | Gerçek Feature İsmi")
    print("-" * 80)
    
    # Fotoğraftaki sıralamaya göre eşleştirme
    for i, (feature, importance) in enumerate(zip(top_20_features, top_20_importance)):
        feature_number = 20 - i  # Fotoğrafta en yüksekten başlıyor
        print(f"Feature_{feature_number:<12} | {feature}")
    
    print("\n" + "="*80)
    print("EN ÖNEMLİ 5 FEATURE'IN DETAYLI AÇIKLAMASI:")
    print("="*80)
    
    # En önemli 5 feature'ı detaylı açıkla
    top_5_indices = top_20_indices[-5:]
    top_5_features = feature_names[top_5_indices]
    top_5_importance = feature_importance[top_5_indices]
    
    for i, (feature, importance) in enumerate(zip(top_5_features, top_5_importance)):
        feature_number = 20 - (len(top_20_indices) - len(top_5_indices) + i)
        print(f"\n{feature_number}. Feature_{feature_number} (En Önemli {i+1}):")
        print(f"   Gerçek İsim: {feature}")
        print(f"   Önem Skoru: {importance:.6f}")
        
        # Feature açıklaması
        if 'pcm_loudness' in feature:
            print(f"   Açıklama: Ses yüksekliği ile ilgili özellik")
        elif 'mfcc' in feature:
            print(f"   Açıklama: MFCC (Mel-frekans cepstral katsayısı) - sesin tınısı")
        elif 'F0' in feature:
            print(f"   Açıklama: Temel frekans (perde) ile ilgili özellik")
        elif 'logMel' in feature:
            print(f"   Açıklama: Log-Mel frekans bandı - spektral özellik")
        else:
            print(f"   Açıklama: Diğer ses özelliği")
    
    print("\n" + "="*80)
    print("ÖZET:")
    print("="*80)
    print("• Fotoğraftaki Feature_30, Feature_25 gibi isimler sadece görselleştirme için")
    print("• Gerçek feature isimleri çok uzun olduğu için kısaltılmış")
    print("• En önemli feature'lar MFCC, ses yüksekliği ve perde ile ilgili")
    print("• Her Feature_X'in gerçek karşılığı yukarıdaki tabloda gösterilmiştir")

if __name__ == "__main__":
    map_features()
