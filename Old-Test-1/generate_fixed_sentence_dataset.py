import os
import glob
import random
import numpy as np
import soundfile as sf
import librosa

# --- YOLLAR ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOUND_DIR = os.path.abspath(os.path.join(BASE_DIR, "../TurEV-DB/Sound Source"))
OUTPUT_DIR = os.path.join(BASE_DIR, "sentencevoice")

EMOTIONS = ["Angry", "Calm", "Happy", "Sad"]
NUM_SENTENCES = 20
SAMPLE_RATE = 16000
MIN_LEN_SEC = 3.0
MAX_LEN_SEC = 10.0

def create_synthetic_sentence(files, sr=SAMPLE_RATE):
    """
    Kullanıcının isteğine göre yüksek kalite mikrofona uygun, 
    min 3, max 10 saniye aralığında, normal ve gerçekçi kelime sayılarına
    ve mantıklı nefes boşluklarına sahip cümleler oluşturur.
    """
    # Kelime sayısı: İnsanlar ortalama 3-10 saniyede rahatça 6 ila 15 kelime edebilir.
    num_words = random.randint(6, 15)
    chosen = random.sample(files, min(num_words, len(files)))
    
    # Doğal başlangıç sessizliği (0.05 - 0.2sn saniye)
    sentence_audio = [np.zeros(int(random.uniform(0.05, 0.2) * sr))]
    
    for idx, f in enumerate(chosen):
        try:
            y, _ = librosa.load(f, sr=sr)
            # Daha yumuşak trim, kelimenin boğuk kalmaması için
            y_trimmed, _ = librosa.effects.trim(y, top_db=35) 
            sentence_audio.append(y_trimmed)
        except Exception:
            continue
            
        # Son kelime değilse araya boşluk ekle
        if idx != len(chosen) - 1:
            # Nadiren uzun nefes arası (virgül noktası vb) %10 ihtimalle 0.4-0.6sn
            # Çoğunlukla akıcı konuşma boşluğu 0.05 - 0.2sn
            if random.random() < 0.10:
                gap_len = random.uniform(0.4, 0.6)
            else:
                gap_len = random.uniform(0.05, 0.2)
            sentence_audio.append(np.zeros(int(gap_len * sr)))
        
    if not sentence_audio: return None
        
    final_audio = np.concatenate(sentence_audio)
    duration = len(final_audio) / sr
    
    # Süre kısıtlamaları (Min 3 saniye, Max 10 saniye)
    if duration < MIN_LEN_SEC:
        # Sonuna rastgele uzunlukta sessizlik ekle (kalan kısmı kapatmak için)
        padding = np.zeros(int((MIN_LEN_SEC - duration) * sr) + int(0.2*sr))
        final_audio = np.concatenate((final_audio, padding))
        
    if (len(final_audio) / sr) > MAX_LEN_SEC:
        final_audio = final_audio[:int(MAX_LEN_SEC * sr)]
        # Sert kesilmeyi engellemek için son 50ms'yi fade out yapalım (Fade-out efekti)
        fade_len = int(0.05 * sr)
        fade_out = np.linspace(1.0, 0.0, fade_len)
        final_audio[-fade_len:] = final_audio[-fade_len:] * fade_out
        
    # --- YÜKSEK KALİTE (HQ) MİKROFON + HAFİF ROBUST ETKİLERİ ---
    # İstikrarlı ses çıkışı (Gain ayarı)
    final_audio = final_audio * random.uniform(0.8, 1.1) 
    
    # %30 ihtimalle çok hafif, ortamın doğal dip (oda) gürültüsü
    if random.random() < 0.30: 
        final_audio += np.random.normal(0, random.uniform(0.001, 0.003), len(final_audio))
        
    # %15 ihtimalle mikrofon nefes çarpması veya ufacık seste cızırtı
    if random.random() < 0.15:
        noise_dur = int(sr * random.uniform(0.05, 0.1))
        noise_start = random.randint(0, max(0, len(final_audio) - noise_dur))
        # Kalın frekanslı bir rumble/pop gürültüsü
        pop = np.random.normal(0, 0.02, noise_dur) * np.linspace(1, 0, noise_dur)
        final_audio[noise_start:noise_start+noise_dur] += pop

    # Kırpılma (Clipping) engellemesi
    max_abs = np.max(np.abs(final_audio))
    if max_abs > 1.0: 
        final_audio = np.clip(final_audio, -1.0, 1.0)
    elif max_abs > 0: 
        # Cümleyi biraz normalize edip standardize edelim ki çok kısık sesli olmasın
        final_audio = final_audio / np.max(np.abs(final_audio)) * random.uniform(0.7, 0.95)
        
    return final_audio

def generate_dataset():
    print(f"Sistem: Sabit HQ Cümle Veriseti Üretiliyor...\nHedef Klasör: {OUTPUT_DIR}")
    
    for emo in EMOTIONS:
        source_folder = os.path.join(SOUND_DIR, emo)
        target_folder = os.path.join(OUTPUT_DIR, emo.lower())
        
        # Klasörleri oluştur
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
            
        wav_files = glob.glob(os.path.join(source_folder, "*.wav"))
        if not wav_files:
            print(f"HATA: {source_folder} içinde hiç .wav dosyası bulunamadı, atlanıyor.")
            continue
            
        print(f"⏳ {emo} kategorisinden {NUM_SENTENCES} cümle üretiliyor...")
        count = 0
        attempts = 0
        while count < NUM_SENTENCES and attempts < NUM_SENTENCES * 5:
            attempts += 1
            audio_array = create_synthetic_sentence(wav_files, sr=SAMPLE_RATE)
            if audio_array is None:
                continue
            
            file_name = f"sentence_{emo.lower()}_{count+1:02d}.wav"
            file_path = os.path.join(target_folder, file_name)
            
            sf.write(file_path, audio_array, SAMPLE_RATE)
            count += 1
            
        print(f"✅ {emo} için {count} dosya başarıyla oluşturuldu.")
        
    print(f"\n🎉 İşlem Tamamlandı. Dosyalarinizi '{OUTPUT_DIR}' klasöründen inceleyebilirsiniz.")

if __name__ == "__main__":
    generate_dataset()
