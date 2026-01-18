"""
Hızlı Backend Test Scripti

Bu script backend'in temel fonksiyonlarını hızlıca test eder.
"""

import requests
import sys
import time

def test_health_check(base_url="http://localhost:8000"):
    """Health check endpoint'ini test et"""
    print("1. Health Check Test...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   [OK] Server saglikli")
            print(f"   Queue size: {data.get('queue_size', 0)}")
            return True
        else:
            print(f"   [FAIL] Server sagliksiz: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"   [FAIL] Server'a baglanilamadi: {e}")
        print(f"   Lutfen server'i baslatin: python main_sentence_analysis.py")
        return False

def test_invalid_format(base_url="http://localhost:8000"):
    """Geçersiz format testi"""
    print("\n2. Geçersiz Format Testi...")
    try:
        files = {"file": ("test.mp3", b"fake audio data", "audio/mpeg")}
        response = requests.post(f"{base_url}/api/analyze-sentence", files=files, timeout=5)
        
        if response.status_code == 400:
            data = response.json()
            if "detail" in data and data["detail"].get("code") == "INVALID_AUDIO_FORMAT":
                print("   [OK] Gecersiz format dogru sekilde reddedildi")
                print(f"   Mesaj: {data['detail'].get('message_tr', data['detail'].get('message'))}")
                return True
            else:
                print(f"   [FAIL] Beklenmeyen hata formati: {data}")
                return False
        else:
            print(f"   [FAIL] Beklenen 400, alinan: {response.status_code}")
            return False
    except Exception as e:
        print(f"   [FAIL] Test hatasi: {e}")
        return False

def test_invalid_strategy(base_url="http://localhost:8000"):
    """Geçersiz aggregation strategy testi"""
    print("\n3. Geçersiz Aggregation Strategy Testi...")
    try:
        # Geçerli bir WAV dosyası oluştur (minimal)
        import numpy as np
        from scipy.io import wavfile
        import tempfile
        import os
        
        # Basit bir test audio oluştur
        duration = 0.5
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.3
        audio_int16 = (audio * 32767).astype(np.int16)
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        wavfile.write(temp_file.name, sample_rate, audio_int16)
        temp_file.close()
        
        try:
            with open(temp_file.name, 'rb') as f:
                files = {"file": ("test.wav", f, "audio/wav")}
                data = {"aggregation_strategy": "invalid_strategy_123"}
                response = requests.post(
                    f"{base_url}/api/analyze-sentence",
                    files=files,
                    data=data,
                    timeout=5
                )
            
            if response.status_code == 400:
                response_data = response.json()
                if "detail" in response_data and response_data["detail"].get("code") == "INVALID_AGGREGATION_STRATEGY":
                    print("   [OK] Gecersiz strategy dogru sekilde reddedildi")
                    return True
                else:
                    print(f"   [FAIL] Beklenmeyen hata: {response_data}")
                    return False
            else:
                print(f"   [FAIL] Beklenen 400, alinan: {response.status_code}")
                return False
        finally:
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)
                
    except ImportError:
        print("   [SKIP] scipy bulunamadi, test atlandi")
        return None
    except Exception as e:
        print(f"   [FAIL] Test hatasi: {e}")
        return False

def test_job_not_found(base_url="http://localhost:8000"):
    """Geçersiz job ID testi"""
    print("\n4. Geçersiz Job ID Testi...")
    try:
        response = requests.get(f"{base_url}/api/status/invalid-job-id-12345", timeout=5)
        
        if response.status_code == 404:
            data = response.json()
            if "detail" in data and data["detail"].get("code") == "JOB_NOT_FOUND":
                print("   [OK] Gecersiz job ID dogru sekilde handle edildi")
                return True
            else:
                print(f"   [FAIL] Beklenmeyen hata formati: {data}")
                return False
        else:
            print(f"   [FAIL] Beklenen 404, alinan: {response.status_code}")
            return False
    except Exception as e:
        print(f"   [FAIL] Test hatasi: {e}")
        return False

def main():
    """Ana test fonksiyonu"""
    print("=" * 60)
    print("Backend Hızlı Test")
    print("=" * 60)
    print()
    
    base_url = "http://localhost:8000"
    
    results = []
    
    # Test 1: Health check
    results.append(("Health Check", test_health_check(base_url)))
    
    # Eğer server çalışmıyorsa diğer testleri atla
    if not results[0][1]:
        print("\n" + "=" * 60)
        print("[WARNING] Server calismiyor. Lutfen once server'i baslatin:")
        print("   python main_sentence_analysis.py")
        print("=" * 60)
        sys.exit(1)
    
    # Test 2: Invalid format
    results.append(("Geçersiz Format", test_invalid_format(base_url)))
    
    # Test 3: Invalid strategy
    strategy_result = test_invalid_strategy(base_url)
    if strategy_result is not None:
        results.append(("Geçersiz Strategy", strategy_result))
    
    # Test 4: Job not found
    results.append(("Geçersiz Job ID", test_job_not_found(base_url)))
    
    # Sonuçları özetle
    print("\n" + "=" * 60)
    print("Test Sonuçları")
    print("=" * 60)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_name, result in results:
        if result is True:
            print(f"[OK] {test_name}: PASSED")
            passed += 1
        elif result is False:
            print(f"[FAIL] {test_name}: FAILED")
            failed += 1
        else:
            print(f"[SKIP] {test_name}: SKIPPED")
            skipped += 1
    
    print("=" * 60)
    print(f"Toplam: {len(results)} test")
    print(f"  [OK] Basarili: {passed}")
    print(f"  [FAIL] Basarisiz: {failed}")
    print(f"  [SKIP] Atlandi: {skipped}")
    print("=" * 60)
    
    if failed > 0:
        sys.exit(1)
    else:
        print("\n[SUCCESS] Tum testler basarili!")
        print("\nDaha detaylı test için:")
        print("  - Integration test: python test_api_integration.py")
        print("  - Property testler: pytest tests/ -v")
        print("  - Test rehberi: TESTING_GUIDE.md")

if __name__ == "__main__":
    main()

