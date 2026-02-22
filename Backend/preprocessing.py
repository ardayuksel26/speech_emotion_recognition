import opensmile
import numpy as np
import os
import logging
import soundfile as sf
import noisereduce as nr
import uuid

logger = logging.getLogger('SER_API.preprocessing')

# =====================================================================
# GLOBAL OPENSMILE INITIALIZATION
# =====================================================================
try:
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.IS10,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    logger.info("✅ OpenSMILE (IS10) motoru başarıyla başlatıldı.")
except Exception as e:
    logger.error(f"❌ OpenSMILE başlatılamadı: {e}")
    smile = None

def extract_features(file_path):
    """
    1. Denoises incoming audio (removes static background mic noise).
    2. Extracts 1582 acoustic features using OpenSMILE IS10 config.
    3. Formats to 1584 dimensions for prediction.
    """
    if smile is None:
        logger.error("OpenSMILE modülü aktif değil. Özellik çıkarımı yapılamaz.")
        return None

    if not os.path.exists(file_path):
        logger.error(f"Ses dosyası bulunamadı: {file_path}")
        return None

    # Temporary file path for cleaned audio
    clean_audio_path = os.path.join(os.path.dirname(file_path), f"temp_clean_{uuid.uuid4().hex[:8]}.wav")

    try:
        # --- 1. REAL-TIME DENOISING ---
        # Read raw recording
        audio_data, rate = sf.read(file_path)
        
        # Apply stationary noise reduction algorithms
        logger.info(f"Yapay Zeka gürültü azaltma (Denoising) uygulanıyor: {file_path}")
        reduced_noise_audio = nr.reduce_noise(y=audio_data, sr=rate, prop_decrease=0.7)
        
        # Save temporary cleaned version
        sf.write(clean_audio_path, reduced_noise_audio, rate)

        # --- 2. OPENSMILE FEATURE EXTRACTION ---
        df = smile.process_file(clean_audio_path)
        
        if df is None or df.empty:
            logger.warning(f"Bağlam çıkarılamadı (Dosya bozuk veya aşırı kısa olabilir): {file_path}")
            return None
            
        features = df.to_numpy().flatten()
        
        if len(features) != 1582:
            logger.warning(f"OpenSMILE beklenmeyen boyutta (%d) özellik döndürdü: {file_path}", len(features))
            return None
        
        # --- 3. DIMENSIONAL FIX ---
        features_padded = np.zeros(1584, dtype=np.float32)
        features_padded[1:1583] = features  
        
        return features_padded
        
    except Exception as e:
        logger.error(f"Özellik çıkarma/gürültü silme hatası ({file_path}): {e}")
        return None
        
    finally:
        # Prevent temporary cleaned files from piling up
        if os.path.exists(clean_audio_path):
            try:
                os.remove(clean_audio_path)
            except:
                pass