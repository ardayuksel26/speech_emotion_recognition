#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Turkce Sesten Duygu Tanima - Hizli Demo Versiyonu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

def quick_demo():
    """Hizli demo analizi"""
    print("TURKCE SESTEN DUYGU TANIMA - HIZLI DEMO")
    print("="*50)
    
    # Veri yukleme
    print("Veri yukleniyor...")
    emotions = ['angry', 'calm', 'happy', 'sad']
    dataframes = []
    
    for emotion in emotions:
        try:
            df = pd.read_csv(f'Extracted_CSV/{emotion}.csv')
            df['emotion'] = emotion
            dataframes.append(df)
            print(f"OK {emotion}.csv: {len(df)} ornek")
        except FileNotFoundError:
            print(f"HATA: {emotion}.csv bulunamadi!")
    
    if not dataframes:
        print("HATA: Veri yuklenemedi!")
        return
    
    # Veriyi birlestir
    data = pd.concat(dataframes, ignore_index=True)
    print(f"\nToplam veri: {len(data)} ornek")
    
    # Veri on isleme
    print("\nVeri on isleme...")
    
    # Ozellikler ve hedef degisken
    X = data.drop('emotion', axis=1)
    y = data['emotion']
    
    # String sutunlari kaldir (emotion hariç)
    string_columns = X.select_dtypes(include=['object']).columns
    if len(string_columns) > 0:
        print(f"String sutunlar kaldiriliyor: {list(string_columns)}")
        X = X.drop(columns=string_columns)
    
    # ID sutununu kaldir
    if 'id' in X.columns:
        X = X.drop('id', axis=1)
    
    # Duygu etiketlerini sayisal degerlere cevir
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    print(f"Ozellik sayisi: {X.shape[1]}")
    print(f"Sinif sayisi: {len(np.unique(y))}")
    print(f"Siniflar: {label_encoder.classes_}")
    
    # Veriyi bolumle
    print("\nVeri bolumleniyor...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Ozellikleri normalize et
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Egitim seti: {X_train.shape[0]} ornek")
    print(f"Test seti: {X_test.shape[0]} ornek")
    
    # Model egitimi
    print("\nModel egitiliyor...")
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Tahmin
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nSONUCLAR:")
    print(f"Dogruluk: {accuracy:.4f}")
    
    # Detayli rapor
    print(f"\nDetayli Performans Raporu:")
    print(classification_report(y_test, y_pred, 
                              target_names=label_encoder.classes_))
    
    # Gorsellestirme
    print("\nGorsellestirme olusturuluyor...")
    
    plt.figure(figsize=(12, 8))
    
    # 1. Confusion Matrix
    plt.subplot(2, 2, 1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=label_encoder.classes_,
               yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gercek')
    
    # 2. Duygu dagilimi
    plt.subplot(2, 2, 2)
    emotion_counts = pd.Series(y).value_counts()
    emotion_labels = [label_encoder.classes_[i] for i in emotion_counts.index]
    plt.pie(emotion_counts.values, labels=emotion_labels, autopct='%1.1f%%')
    plt.title('Duygu Dagilimi')
    
    # 3. Ozellik onemleri
    plt.subplot(2, 2, 3)
    feature_importance = model.feature_importances_
    top_features = np.argsort(feature_importance)[-15:]
    plt.barh(range(len(top_features)), feature_importance[top_features])
    plt.xlabel('Ozellik Onem Skoru')
    plt.title('En Onemli 15 Ozellik')
    plt.yticks(range(len(top_features)), [f'F{i}' for i in top_features])
    
    # 4. Model performansi
    plt.subplot(2, 2, 4)
    plt.bar(['Dogruluk'], [accuracy], color='green', alpha=0.7)
    plt.ylabel('Skor')
    plt.title('Model Performansi')
    plt.ylim(0, 1)
    plt.text(0, accuracy + 0.02, f'{accuracy:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('quick_demo_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nDemo tamamlandi!")
    print("Sonuclar 'quick_demo_results.png' dosyasinda kaydedildi.")
    
    return model, scaler, label_encoder

if __name__ == "__main__":
    quick_demo()
