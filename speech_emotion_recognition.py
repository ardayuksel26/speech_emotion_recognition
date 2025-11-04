#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Turkce Sesten Duygu Tanima (Speech Emotion Recognition) Projesi
Bu proje Turkce ses dosyalarindan duygu tanima yapmak icin gelistirilmistir.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Turkce karakter desteği için matplotlib ayarları
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class SpeechEmotionRecognition:
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """CSV dosyalarından veriyi yükle"""
        print("Veri yukleniyor...")
        
        # Her duygu için CSV dosyalarını yükle
        emotions = ['angry', 'calm', 'happy', 'sad']
        dataframes = []
        
        for emotion in emotions:
            try:
                df = pd.read_csv(f'Extracted_CSV/{emotion}.csv')
                df['emotion'] = emotion
                dataframes.append(df)
                print(f"OK {emotion}.csv yuklendi: {len(df)} ornek")
            except FileNotFoundError:
                print(f"HATA: {emotion}.csv dosyasi bulunamadi!")
                
        if dataframes:
            self.data = pd.concat(dataframes, ignore_index=True)
            print(f"\nToplam veri: {len(self.data)} ornek")
            print(f"Ozellik sayisi: {self.data.shape[1]-2}")  # id ve emotion hariç
            return True
        else:
            print("HATA: Hicbir veri dosyasi yuklenemedi!")
            return False
    
    def explore_data(self):
        """Veri setini keşfet"""
        print("\nVERI KESFI")
        print("="*50)
        
        # Temel bilgiler
        print(f"Veri boyutu: {self.data.shape}")
        print(f"Duygu dagilimi:")
        emotion_counts = self.data['emotion'].value_counts()
        print(emotion_counts)
        
        # Eksik değerler
        missing_values = self.data.isnull().sum()
        if missing_values.sum() > 0:
            print(f"\nEksik degerler:")
            print(missing_values[missing_values > 0])
        else:
            print("\nEksik deger yok!")
        
        # Özellik türleri
        print(f"\nOzellik turleri:")
        print(self.data.dtypes.value_counts())
        
        # İstatistiksel özet
        print(f"\nIstatistiksel ozet:")
        print(self.data.describe())
        
        return emotion_counts
    
    def preprocess_data(self):
        """Veriyi ön işle"""
        print("\nVERI ON ISLEME")
        print("="*50)
        
        # ID sütununu kaldır
        if 'id' in self.data.columns:
            self.data = self.data.drop('id', axis=1)
            print("ID sutunu kaldirildi")
        
        # Eksik değerleri doldur
        if self.data.isnull().sum().sum() > 0:
            self.data = self.data.fillna(self.data.mean())
            print("Eksik degerler ortalama ile dolduruldu")
        
        # Özellikler ve hedef değişkeni ayır
        self.X = self.data.drop('emotion', axis=1)
        self.y = self.data['emotion']
        
        # String sütunları kaldır
        string_columns = self.X.select_dtypes(include=['object']).columns
        if len(string_columns) > 0:
            print(f"String sutunlar kaldiriliyor: {list(string_columns)}")
            self.X = self.X.drop(columns=string_columns)
        
        # Duygu etiketlerini sayısal değerlere çevir
        self.y = self.label_encoder.fit_transform(self.y)
        
        print(f"Ozellik matrisi: {self.X.shape}")
        print(f"Hedef degisken: {len(np.unique(self.y))} sinif")
        print(f"Sinif etiketleri: {self.label_encoder.classes_}")
    
    def feature_engineering(self):
        """Özellik mühendisliği"""
        print("\nOZELLIK MUHENDISLIGI")
        print("="*50)
        
        # Korelasyon analizi
        correlation_matrix = self.X.corr()
        high_corr_features = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i, j]) > 0.95:
                    high_corr_features.append(correlation_matrix.columns[j])
        
        # Yüksek korelasyonlu özellikleri kaldır
        if high_corr_features:
            self.X = self.X.drop(columns=high_corr_features)
            print(f"{len(high_corr_features)} yuksek korelasyonlu ozellik kaldirildi")
        
        # Özellikleri normalize et
        self.X = self.scaler.fit_transform(self.X)
        print(f"Ozellikler normalize edildi")
        print(f"Son ozellik sayisi: {self.X.shape[1]}")
    
    def split_data(self, test_size=0.2, random_state=42):
        """Veriyi eğitim ve test setlerine ayır"""
        print(f"\nVERI BOLUMLEME")
        print("="*50)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        print(f"Egitim seti: {self.X_train.shape[0]} ornek")
        print(f"Test seti: {self.X_test.shape[0]} ornek")
        print(f"Ozellik sayisi: {self.X_train.shape[1]}")
    
    def train_models(self):
        """Farklı modelleri eğit"""
        print("\nMODEL EGITIMI")
        print("="*50)
        
        # Model tanımları
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(random_state=42),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
        }
        
        # Her modeli eğit ve değerlendir
        for name, model in models.items():
            print(f"\n{name} egitiliyor...")
            
            # Modeli eğit
            model.fit(self.X_train, self.y_train)
            
            # Tahmin yap
            y_pred = model.predict(self.X_test)
            
            # Doğruluk hesapla
            accuracy = accuracy_score(self.y_test, y_pred)
            
            # Çapraz doğrulama
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
            
            # Sonuçları kaydet
            self.models[name] = model
            self.results[name] = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred
            }
            
            print(f"{name} - Dogruluk: {accuracy:.4f}")
            print(f"{name} - CV Ortalama: {cv_scores.mean():.4f} (+/-{cv_scores.std():.4f})")
    
    def evaluate_models(self):
        """Modelleri değerlendir"""
        print("\nMODEL DEGERLENDIRME")
        print("="*50)
        
        # En iyi modeli bul
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        best_model = self.models[best_model_name]
        
        print(f"En iyi model: {best_model_name}")
        print(f"En iyi dogruluk: {self.results[best_model_name]['accuracy']:.4f}")
        
        # Detaylı değerlendirme
        y_pred = self.results[best_model_name]['predictions']
        
        print(f"\nDetayli Performans Raporu:")
        print(classification_report(self.y_test, y_pred, 
                                  target_names=self.label_encoder.classes_))
        
        return best_model_name, best_model
    
    def save_best_model(self):
        """En iyi modeli ve gerekli bileşenleri kaydet"""
        print("\nMODEL KAYDETME")
        print("="*50)
        
        if not self.models or not self.results:
            print("HATA: Kaydedilecek model bulunamadi!")
            return False
        
        # En iyi modeli bul
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        best_model = self.models[best_model_name]
        
        # Kaydetmek için model paketini hazırla
        model_package = {
            'model': best_model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'best_model_name': best_model_name,
            'accuracy': self.results[best_model_name]['accuracy'],
            'cv_mean': self.results[best_model_name]['cv_mean'],
            'cv_std': self.results[best_model_name]['cv_std'],
            'feature_columns': list(self.X.columns) if hasattr(self.X, 'columns') else None,
            'all_results': self.results
        }
        
        # Modeli kaydet
        os.makedirs('models', exist_ok=True)
        model_path = 'models/best_emotion_model.pkl'
        
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model_package, f)
            print(f"Model basariyla kaydedildi: {model_path}")
            print(f"Kaydedilen model: {best_model_name}")
            print(f"Model dogrulugu: {self.results[best_model_name]['accuracy']:.4f}")
            return True
        except Exception as e:
            print(f"HATA: Model kaydedilemedi: {str(e)}")
            return False
    
    def visualize_results(self):
        """Sonuçları görselleştir"""
        print("\nGORSELLESTIRME")
        print("="*50)
        
        # Model performanslarını karşılaştır
        plt.figure(figsize=(15, 10))
        
        # 1. Model doğrulukları
        plt.subplot(2, 3, 1)
        model_names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        cv_means = [self.results[name]['cv_mean'] for name in model_names]
        cv_stds = [self.results[name]['cv_std'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        plt.bar(x - width/2, accuracies, width, label='Test Dogrulugu', alpha=0.8)
        plt.bar(x + width/2, cv_means, width, label='CV Ortalama', alpha=0.8, yerr=cv_stds, capsize=5)
        
        plt.xlabel('Modeller')
        plt.ylabel('Dogruluk')
        plt.title('Model Performans Karsilastirmasi')
        plt.xticks(x, model_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. En iyi model için confusion matrix
        plt.subplot(2, 3, 2)
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        y_pred = self.results[best_model_name]['predictions']
        
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.xlabel('Tahmin Edilen')
        plt.ylabel('Gercek')
        
        # 3. Duygu dağılımı
        plt.subplot(2, 3, 3)
        emotion_counts = pd.Series(self.y).value_counts()
        emotion_labels = [self.label_encoder.classes_[i] for i in emotion_counts.index]
        
        plt.pie(emotion_counts.values, labels=emotion_labels, autopct='%1.1f%%', startangle=90)
        plt.title('Duygu Dagilimi')
        
        # 4. Özellik önemleri (Random Forest için)
        if 'Random Forest' in self.models:
            plt.subplot(2, 3, 4)
            rf_model = self.models['Random Forest']
            feature_importance = rf_model.feature_importances_
            
            # En önemli 20 özelliği göster
            top_features = np.argsort(feature_importance)[-20:]
            plt.barh(range(len(top_features)), feature_importance[top_features])
            plt.xlabel('Ozellik Onem Skoru')
            plt.title('En Onemli 20 Ozellik (Random Forest)')
            plt.yticks(range(len(top_features)), [f'Feature_{i}' for i in top_features])
        
        # 5. PCA ile boyut azaltma görselleştirmesi
        plt.subplot(2, 3, 5)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X)
        
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=self.y, cmap='viridis', alpha=0.6)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('PCA Gorsellestirmesi')
        plt.colorbar(scatter, label='Duygu')
        
        # 6. Model karşılaştırma tablosu
        plt.subplot(2, 3, 6)
        plt.axis('off')
        
        # Performans tablosu
        table_data = []
        for name in model_names:
            table_data.append([
                name,
                f"{self.results[name]['accuracy']:.4f}",
                f"{self.results[name]['cv_mean']:.4f}",
                f"{self.results[name]['cv_std']:.4f}"
            ])
        
        table = plt.table(cellText=table_data,
                         colLabels=['Model', 'Test Acc.', 'CV Mean', 'CV Std'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        plt.title('Model Performans Tablosu')
        
        plt.tight_layout()
        plt.savefig('speech_emotion_recognition_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Gorsellestirmeler kaydedildi: speech_emotion_recognition_results.png")
    
    def predict_emotion(self, audio_features):
        """Yeni ses özelliklerinden duygu tahmin et"""
        if not self.models:
            print("HATA: Once modelleri egitmeniz gerekiyor!")
            return None
        
        # En iyi modeli kullan
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        best_model = self.models[best_model_name]
        
        # Özellikleri normalize et
        audio_features_scaled = self.scaler.transform([audio_features])
        
        # Tahmin yap
        prediction = best_model.predict(audio_features_scaled)[0]
        emotion = self.label_encoder.inverse_transform([prediction])[0]
        
        # Olasılık skorları
        probabilities = best_model.predict_proba(audio_features_scaled)[0]
        
        return {
            'emotion': emotion,
            'confidence': max(probabilities),
            'all_probabilities': dict(zip(self.label_encoder.classes_, probabilities))
        }
    
    def run_complete_analysis(self):
        """Tam analiz sürecini çalıştır"""
        print("TURKCE SESTEN DUYGU TANIMA PROJESI")
        print("="*60)
        
        # 1. Veri yükleme
        if not self.load_data():
            return False
        
        # 2. Veri keşfi
        self.explore_data()
        
        # 3. Veri ön işleme
        self.preprocess_data()
        
        # 4. Özellik mühendisliği
        self.feature_engineering()
        
        # 5. Veri bölümleme
        self.split_data()
        
        # 6. Model eğitimi
        self.train_models()
        
        # 7. Model değerlendirme
        best_model_name, best_model = self.evaluate_models()
        
        # 8. En iyi modeli kaydet
        self.save_best_model()
        
        # 9. Görselleştirme
        self.visualize_results()
        
        print(f"\nAnaliz tamamlandi!")
        print(f"En iyi model: {best_model_name}")
        print(f"En iyi dogruluk: {self.results[best_model_name]['accuracy']:.4f}")
        print(f"Model 'models/best_emotion_model.pkl' dosyasina kaydedildi.")
        
        return True

def main():
    """Ana fonksiyon"""
    # SER nesnesini oluştur
    ser = SpeechEmotionRecognition()
    
    # Tam analizi çalıştır
    success = ser.run_complete_analysis()
    
    if success:
        print("\nProje basariyla tamamlandi!")
        print("Sonuclar 'speech_emotion_recognition_results.png' dosyasinda kaydedildi.")
    else:
        print("\nProje tamamlanamadi!")

if __name__ == "__main__":
    main()