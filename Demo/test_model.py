#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model test scripti - Kaydedilen modelin çalışıp çalışmadığını test eder
"""

from model_utils import load_trained_model, get_model_info, check_model_exists
import numpy as np

def test_model():
    """Kaydedilen modeli test et"""
    print("🧪 Model Test Scripti")
    print("="*50)
    
    # Model var mı kontrol et
    if not check_model_exists():
        print("❌ Model dosyası bulunamadı!")
        print("Önce 'python speech_emotion_recognition.py' komutunu çalıştırın.")
        return False
    
    print("✅ Model dosyası bulundu!")
    
    # Model bilgilerini al
    model_info = get_model_info()
    if model_info is None:
        print("❌ Model bilgileri alınamadı!")
        return False
    
    print(f"📊 Model Bilgileri:")
    print(f"   - Model Türü: {model_info['model_name']}")
    print(f"   - Test Doğruluğu: {model_info['accuracy']:.2%}")
    print(f"   - CV Ortalama: {model_info['cv_mean']:.2%}")
    print(f"   - CV Std: {model_info['cv_std']:.4f}")
    
    # Modeli yükle
    model_package = load_trained_model()
    if model_package is None:
        print("❌ Model yüklenemedi!")
        return False
    
    print("✅ Model başarıyla yüklendi!")
    
    # Test tahmini yap
    print("\n🔮 Test Tahmini:")
    
    # Rastgele özellik vektörü oluştur (gerçek özellik sayısı kadar)
    n_features = model_package['scaler'].n_features_in_
    test_features = np.random.randn(n_features)
    
    try:
        # Tahmin yap
        features_scaled = model_package['scaler'].transform([test_features])
        prediction = model_package['model'].predict(features_scaled)[0]
        probabilities = model_package['model'].predict_proba(features_scaled)[0]
        
        # Sonuçları göster
        emotion = model_package['label_encoder'].inverse_transform([prediction])[0]
        confidence = max(probabilities)
        
        print(f"   - Tahmin Edilen Duygu: {emotion}")
        print(f"   - Güven Skoru: {confidence:.2%}")
        
        print(f"\n📊 Tüm Duygu Olasılıkları:")
        for i, (label, prob) in enumerate(zip(model_package['label_encoder'].classes_, probabilities)):
            print(f"   - {label}: {prob:.2%}")
        
        print("\n✅ Model testi başarılı!")
        return True
        
    except Exception as e:
        print(f"❌ Tahmin sırasında hata: {str(e)}")
        return False

def main():
    """Ana fonksiyon"""
    success = test_model()
    
    if success:
        print("\n🎉 Tüm testler başarılı!")
        print("Demo uygulamasını çalıştırmak için:")
        print("   streamlit run demo_app.py")
    else:
        print("\n❌ Test başarısız!")
        print("Modeli yeniden eğitmek için:")
        print("   python speech_emotion_recognition.py")

if __name__ == "__main__":
    main()