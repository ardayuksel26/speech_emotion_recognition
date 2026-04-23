import os
import sys
import glob
import numpy as np
import joblib
import time
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Path Definitions
BASE_DIR = r"c:\Users\ilhan\Desktop\SER_PROJECT"
TEST_GENEL_DIR = os.path.join(BASE_DIR, "Test-Genel-1")
TEST_CODES_DIR = os.path.join(TEST_GENEL_DIR, "test_codes")
TEST_RESULTS_DIR = os.path.join(TEST_GENEL_DIR, "test_results")

BACKEND_DIR = os.path.join(BASE_DIR, "Backend")
MODELS_DIR = os.path.join(BASE_DIR, "Models")
MODELS_2_DIR = os.path.join(BASE_DIR, "Models_2")
SENTENCE_MODELS_DIR = os.path.join(MODELS_DIR, "Sentence_Models")
HF_DIR = os.path.join(BASE_DIR, "Huggingface")
DATASET_DIR = os.path.join(BASE_DIR, "Test", "sentencevoice_test")

os.makedirs(TEST_CODES_DIR, exist_ok=True)
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

sys.path.append(BACKEND_DIR)
sys.path.append(HF_DIR)

# Import feature extractors
try:
    from preprocessing import extract_features
except ImportError:
    print("Could not import standard extract_features from preprocessing.")
    extract_features = None

def extract_features_plain(file_path):
    """Plain IS10 extraction without denoising — matches Models_2 training features (1582 dims)."""
    try:
        import opensmile as _os2
        _smile2 = _os2.Smile(
            feature_set=_os2.FeatureSet.IS10,
            feature_level=_os2.FeatureLevel.Functionals,
        )
        df = _smile2.process_file(file_path)
        feats = df.to_numpy().flatten().astype(np.float32)
        return feats if len(feats) == 1582 else None
    except Exception as _e2:
        return None

# Load dataset paths
emotions = ['angry', 'calm', 'happy', 'sad']
dataset_files = []
true_labels = []

print("Scanning dataset files...")
for emotion in emotions:
    folder = os.path.join(DATASET_DIR, emotion)
    if os.path.isdir(folder):
        wavs = glob.glob(os.path.join(folder, "*.wav"))
        for w in wavs:
            dataset_files.append(w)
            true_labels.append(emotion)
print(f"Total test files: {len(dataset_files)}")

# Pre-Extraction Structures
print("Extracting features... This might take a few minutes as we use Vosk for word segmentation.")
try:
    from stt_service import transcribe
except ImportError:
    print("stt_service could not be imported! Make sure vosk is installed.")
    transcribe = None

import tempfile
import librosa
import soundfile as sf
import collections

# Memory for Full Sentence (Sentence_Models — uses 1582 plain IS10, same as training)
features_v1_sentence = []
valid_indices_sentence_v1 = []

# Memory for Words (Models and Models_2)
# Lists containing numpy arrays of features for each word in a sentence
word_features_v1 = []
word_features_v2 = []
valid_indices_words = []

for i, f in enumerate(dataset_files):
    # --- 1. FULL SENTENCE EXTRACTION (For Sentence Models - 1582 plain IS10) ---
    feat_sentence = extract_features_plain(f)
    if feat_sentence is not None:
        features_v1_sentence.append(feat_sentence)
        valid_indices_sentence_v1.append(i)
            
    # --- 2. WORD SEGMENTATION (For Word Models) ---
    if transcribe is not None:
        try:
            words = transcribe(f, engine="vosk")
            if not words:
                # No words detected: add empty lists and still track this index
                word_features_v1.append([])
                word_features_v2.append([])
                valid_indices_words.append(i)
                continue
                
            y, sr = librosa.load(f, sr=22050)
            
            file_word_feats_v1 = []
            file_word_feats_v2 = []
            
            for word_info in words:
                start_s = float(word_info.get('start', 0))
                end_s = float(word_info.get('end', 0))
                
                if end_s - start_s < 0.1:
                    continue # Skip very short glitches
                    
                start_idx = int(start_s * sr)
                end_idx = int((end_s + 0.1) * sr) # small buffer
                y_seg = y[start_idx:end_idx]
                
                # Write temp file for extraction
                fd, tmp_path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                sf.write(tmp_path, y_seg, sr)
                
                # Extract V1 (1584 dims)
                if extract_features:
                    feat_v1 = extract_features(tmp_path)
                    if feat_v1 is not None:
                        file_word_feats_v1.append(feat_v1)
                        
                # Extract V2 (1582 dims)
                feat_v2 = extract_features_plain(tmp_path)
                if feat_v2 is not None:
                    file_word_feats_v2.append(feat_v2)
                    
                try:
                    os.remove(tmp_path)
                except:
                    pass
                    
            word_features_v1.append(file_word_feats_v1)
            word_features_v2.append(file_word_feats_v2)
            valid_indices_words.append(i)
            
        except Exception as e:
            word_features_v1.append([])
            word_features_v2.append([])
            valid_indices_words.append(i)

features_v1_sentence = np.array(features_v1_sentence)

def generate_report(y_true, y_pred, model_name, output_folder, all_labels, model_path_info="Bilinmiyor"):
    os.makedirs(output_folder, exist_ok=True)
    report_path = os.path.join(output_folder, f"{model_name}_report.txt")
    
    unique_true = set(y_true)
    unique_pred = set(y_pred)
    all_present_labels = sorted(list(unique_true.union(unique_pred)))
    
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=all_present_labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=all_present_labels)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"=== {model_name} Test Sonuçları ===\n")
        f.write(f"Model Konumu: {model_path_info}\n")
        f.write(f"Doğruluk (Accuracy): %{acc*100:.2f}\n")
        f.write(f"Toplam Test Edilen Dosya: {len(y_true)}\n\n")
        f.write(report)
        
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=all_present_labels, yticklabels=all_present_labels)
    plt.title(f'{model_name} Karmaşıklık Matrisi')
    plt.ylabel('Gerçek Duygu')
    plt.xlabel('Tahmin Edilen Duygu')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{model_name}_cm.png"))
    plt.close()
    print(f"-> {model_name} tamamlandı. Sonuçlar: {output_folder}")

def evaluate_sklearn_group(group_name, config_dict, is_word_model, features_sentence, features_words, valid_indices):
    group_out_dir = os.path.join(TEST_RESULTS_DIR, group_name)
    y_true_group = [true_labels[i] for i in valid_indices]
    
    for name, paths in config_dict.items():
        try:
            model = joblib.load(paths['model'])
            scaler = joblib.load(paths['scaler'])
            encoder = joblib.load(paths['encoder'])
            
            final_predictions = []
            
            if is_word_model:
                # Majority Voting per sentence based on segmented words
                for file_word_feats in features_words:
                    if not file_word_feats:
                        final_predictions.append("calm") # Fallback for no words detected
                        continue
                        
                    feats_np = np.array(file_word_feats)
                    scaled_feats = scaler.transform(feats_np)
                    
                    if hasattr(model, 'predict_proba'):
                        probs = model.predict_proba(scaled_feats)
                        pred_indices = np.argmax(probs, axis=1)
                    else:
                        pred_indices = model.predict(scaled_feats)
                        if isinstance(pred_indices[0], np.ndarray):
                            pred_indices = [int(p.item()) for p in pred_indices]
                    
                    word_labels = encoder.inverse_transform(pred_indices)
                    word_labels = [str(p).lower() for p in word_labels]
                    word_labels = ['calm' if p == 'neutral' else p for p in word_labels]
                    
                    # Majority Voting
                    counter = collections.Counter(word_labels)
                    most_common = counter.most_common(1)[0][0]
                    final_predictions.append(most_common)
                    
            else:
                # Full Sentence Prediction
                scaled_feats = scaler.transform(features_sentence)
                preds = model.predict(scaled_feats)
                if isinstance(preds, list) or (isinstance(preds, np.ndarray) and len(preds.shape) > 1):
                    preds = np.array(preds).flatten()
                
                try:
                    if isinstance(preds[0], (int, np.integer, float, np.floating)):
                        pred_labels = encoder.inverse_transform([int(p) for p in preds])
                    else:
                        pred_labels = preds
                except Exception:
                    pred_labels = encoder.inverse_transform(preds)
                
                pred_labels = [str(p).lower() for p in pred_labels]
                final_predictions = ['calm' if p == 'neutral' else p for p in pred_labels]
            
            model_location = paths.get('model', 'Bilinmiyor')
            generate_report(y_true_group, final_predictions, name, os.path.join(group_out_dir, name), emotions, model_location)
        except Exception as e:
            print(f"Hata ({group_name} - {name}): {e}")

def discover_models_in_dir(base_dir):
    config = {}
    if not os.path.isdir(base_dir):
        return config
        
    def find_model_in_files(file_list, key_name):
        model_file, scaler_file, encoder_file = None, None, None
        for pkl in file_list:
            lower_name = os.path.basename(pkl).lower()
            if 'scaler' in lower_name:
                scaler_file = pkl
            elif 'encoder' in lower_name:
                encoder_file = pkl
            elif 'model' in lower_name or 'robust' in lower_name or 'svm' in lower_name or 'knn' in lower_name or 'mlp' in lower_name:
                model_file = pkl
        if model_file and scaler_file and encoder_file:
            config[key_name] = {
                'model': model_file,
                'scaler': scaler_file,
                'encoder': encoder_file
            }
            
    # 1. Root Models (e.g. Random Forest in Sentence_Models)
    pkl_in_base = glob.glob(os.path.join(base_dir, "*.pkl"))
    if pkl_in_base:
        find_model_in_files(pkl_in_base, os.path.basename(base_dir) + "_Root_Model")
        
    # 2. Subdirectories
    for folder in os.listdir(base_dir):
        full_path = os.path.join(base_dir, folder)
        if not os.path.isdir(full_path):
            continue
        pkl_files = glob.glob(os.path.join(full_path, "*.pkl"))
        find_model_in_files(pkl_files, folder)
        
    return config

# ================= 1. MODELS (V1 Word-Level) =================
models_v1_config = discover_models_in_dir(MODELS_DIR)
print(f"\n--- Değerlendiriliyor: Models (V1) - Word-Level Majority Voting ---")
evaluate_sklearn_group("Models", models_v1_config, is_word_model=True, features_sentence=None, features_words=word_features_v1, valid_indices=valid_indices_words)

# ================= 2. MODELS 2 (V2 Word-Level - IS10 1582 Dims) =================
models_v2_config = discover_models_in_dir(MODELS_2_DIR)
print(f"\n--- Değerlendiriliyor: Models_2 (V2) - Word-Level Majority Voting ---")
evaluate_sklearn_group("Models_2", models_v2_config, is_word_model=True, features_sentence=None, features_words=word_features_v2, valid_indices=valid_indices_words)

# ================= 3. SENTENCE MODELS (Full Sentence) =================
sentence_config = discover_models_in_dir(SENTENCE_MODELS_DIR)
print(f"\n--- Değerlendiriliyor: Sentence_Models - Full Sentence Analysis ---")
evaluate_sklearn_group("Sentence_Models", sentence_config, is_word_model=False, features_sentence=features_v1_sentence, features_words=None, valid_indices=valid_indices_sentence_v1)

# ================= 4. HUGGINGFACE MODELS (Full Sentence) =================
print("\n--- Değerlendiriliyor: Huggingface Modelleri ---")
hf_out_dir = os.path.join(TEST_RESULTS_DIR, "Huggingface")

hf_predictors = [
    ("SenseVoice", "sensevoice_model", "SenseVoiceEmotionPredictor"),
    ("Wav2Vec2Turkish", "wav2vec2_model", "Wav2Vec2TurkishPredictor"),
    ("WavLM", "wavlm_model", "WavLMEmotionPredictor"),
    ("WavLMBasePlus", "wavlm_model", "WavLMBasePlusEmotionPredictor"),
    ("XLSR", "xlsr_model", "XLSREmotionPredictor"),
    ("Wav2Vec2English", "xlsr_model", "Wav2Vec2EnglishPredictor"),
    ("HuBERT", "hubert_model", "HubertEmotionPredictor"),
    ("ExHuBERT", "exhubert_model", "ExHuBERTEmotionPredictor")
]

for name, module_name, class_name in hf_predictors:
    print(f"[{name}] Yükleniyor...")
    try:
        module = __import__(module_name)
        predictor_class = getattr(module, class_name)
        predictor = predictor_class()
        
        preds = []
        for f in dataset_files:
            try:
                # Hugging Face usually expects full files. End-to-end transformers handle dynamic length naturally.
                result = predictor.predict(f)
                emotion = result['emotion'].lower()
                preds.append(emotion)
            except Exception as e:
                preds.append("error")
                
        hf_script_path = os.path.join(HF_DIR, f"{module_name}.py")
        generate_report(true_labels, preds, name, os.path.join(hf_out_dir, name), emotions, hf_script_path)
    except Exception as e:
        print(f"Hata ({name}): {e}")

print("\n=== TÜM TESTLER TAMAMLANDI ===")
print(f"Sonuçlar {TEST_RESULTS_DIR} dizinine kaydedildi.")
