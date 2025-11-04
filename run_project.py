#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Proje çalıştırma scripti
Önce modeli eğitir, sonra demo uygulamasını başlatır
"""

import os
import sys
import subprocess
from model_utils import check_model_exists

def run_training():
    """Model eğitimini çalıştır"""
    print("🚀 Model eğitimi başlatılıyor...")
    print("="*50)
    
    try:
        # speech_emotion_recognition.py dosyasını çalıştır
        result = subprocess.run([sys.executable, 'speech_emotion_recognition.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Model eğitimi başarıyla tamamlandı!")
            print(result.stdout)
            return True
        else:
            print("❌ Model eğitiminde hata oluştu:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Eğitim sırasında hata: {str(e)}")
        return False

def run_demo():
    """Demo uygulamasını başlat"""
    print("\n🎵 Demo uygulaması başlatılıyor...")
    print("="*50)
    
    try:
        # Streamlit uygulamasını başlat
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'demo_app.py'])
    except KeyboardInterrupt:
        print("\n👋 Demo uygulaması kapatıldı.")
    except Exception as e:
        print(f"❌ Demo başlatılırken hata: {str(e)}")

def main():
    """Ana fonksiyon"""
    print("🎵 Türkçe Sesten Duygu Tanıma Projesi")
    print("="*60)
    
    # Model var mı kontrol et
    if check_model_exists():
        print("✅ Eğitilmiş model bulundu!")
        choice = input("Modeli yeniden eğitmek istiyor musunuz? (y/N): ").lower().strip()
        
        if choice in ['y', 'yes', 'evet', 'e']:
            if not run_training():
                print("❌ Model eğitimi başarısız. Demo çalıştırılamıyor.")
                return
        else:
            print("📊 Mevcut model kullanılacak.")
    else:
        print("⚠️ Eğitilmiş model bulunamadı. Önce model eğitilecek...")
        if not run_training():
            print("❌ Model eğitimi başarısız. Demo çalıştırılamıyor.")
            return
    
    # Demo'yu başlat
    run_demo()

if __name__ == "__main__":
    main()