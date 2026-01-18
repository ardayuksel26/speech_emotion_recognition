import os
import librosa
import numpy as np
from tqdm import tqdm  # İlerleme çubuğu için

# --- AYARLAR ---
SOURCE_PATH = "TurEV-DB/Sound Source"
TARGET_SR = 22050

def check_durations():
    audio_files = [] # (süre, dosya_yolu) ikililerini tutacak liste
    
    print("Dosyalar taranıyor...")
    
    # Tüm wav dosyalarını bul
    paths = []
    for root, _, files in os.walk(SOURCE_PATH):
        for f in files:
            if f.endswith('.wav'):
                full_path = os.path.join(root, f)
                paths.append(full_path)

    print(f"Toplam {len(paths)} dosya bulundu. Süreler hesaplanıyor...")

    durations = []
    
    # İlerleme çubuğu ile dosyaları tek tek analiz et
    for p in tqdm(paths):
        try:
            # Sadece süreyi öğrenmek için yüklüyoruz
            y, sr = librosa.load(p, sr=TARGET_SR)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Listeye ekle: (Süre, Dosya Adı)
            durations.append(duration)
            audio_files.append((duration, p))
            
        except Exception as e:
            print(f"HATA: {p} dosyası okunamadı. Sebebi: {e}")

    # --- SONUÇLARI ANALİZ ET ---
    if not audio_files:
        print("Hiç .wav dosyası bulunamadı!")
        return

    # En kısa ve en uzun dosyayı bul
    shortest = min(audio_files, key=lambda x: x[0]) # Süreye (0. indeks) göre en küçük
    longest = max(audio_files, key=lambda x: x[0])  # Süreye göre en büyük
    avg_duration = np.mean(durations)

    print("\n" + "="*40)
    print("      SES SÜRESİ ANALİZ SONUÇLARI")
    print("="*40)
    
    print(f"📂 Toplam Dosya: {len(durations)}")
    print(f"⏱️  Ortalama Süre: {avg_duration:.2f} saniye")
    print("-" * 40)
    
    print(f"🟢 EN KISA SES:")
    print(f"   Süre : {shortest[0]:.4f} saniye")
    print(f"   Dosya: {os.path.basename(shortest[1])}")
    print(f"   Konum: {shortest[1]}")
    print("-" * 40)
    
    print(f"🔴 EN UZUN SES:")
    print(f"   Süre : {longest[0]:.4f} saniye")
    print(f"   Dosya: {os.path.basename(longest[1])}")
    print(f"   Konum: {longest[1]}")
    print("="*40)

    # Tavsiye Bölümü
    print("\n💡 TAVSİYE:")
    recommended_duration = int(np.ceil(longest[0]))
    if recommended_duration > 3:
        print(f"Dikkat! En uzun ses {longest[0]:.2f} saniye.")
        print(f"Mevcut DURATION = 3 ayarı ile verinin bir kısmı KESİLİYOR olabilir.")
        print(f"DURATION = {recommended_duration} yapman daha güvenli olabilir.")
    else:
        print(f"En uzun ses {longest[0]:.2f} saniye.")
        print("DURATION = 3 ayarı GÜVENLİ. Hiçbir veri kaybı yaşanmayacak.")

if __name__ == "__main__":
    check_durations()