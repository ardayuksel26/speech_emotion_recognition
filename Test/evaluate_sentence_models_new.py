import os
import glob
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- AYARLAR ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "../Models/Sentence_Models"))
RESULTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "Test_Results"))

TEST_DATASETS = {
    "Synthetic_Actors (sentencevoice_test)": os.path.abspath(os.path.join(BASE_DIR, "sentencevoice_test/Extracted CSV")),
    "User_Voice (MyRecordings)": os.path.abspath(os.path.join(BASE_DIR, "MyRecordings/Sentence/Extracted CSV"))
}

try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

os.makedirs(RESULTS_DIR, exist_ok=True)

EMOTION_FILES = {
    "angry.csv": "angry",
    "calm.csv": "calm",
    "happy.csv": "happy",
    "sad.csv": "sad"
}

def apply_speaker_normalization(df):
    """Her konuşmacıyı kendi içinde normalize ederek bireysel farkları ortadan kaldırır."""
    target_col = 'label'
    speaker_col = 'speaker_id'
    
    # Sayısal sütunları bul (label ve speaker_id hariç)
    skip_cols = [target_col, speaker_col, 'filename', 'name', 'path', 'dosya_adi']
    feat_cols = [c for c in df.columns if c not in skip_cols and df[c].dtype in ['float64', 'int64']]
    
    # Grupla ve normalize et
    df[feat_cols] = df.groupby(speaker_col)[feat_cols].transform(lambda x: (x - x.mean()) / (x.std() + 1e-6))
    return df

def load_test_data(data_dir):
    all_dfs = []
    for filename, label in EMOTION_FILES.items():
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['label'] = label
            all_dfs.append(df)
            
    if not all_dfs:
        return None
        
    df_all = pd.concat(all_dfs, ignore_index=True)
    print(f"Konuşmacı bazlı normalizasyon uygulanıyor ({len(df_all)} satır)...")
    df_all = apply_speaker_normalization(df_all)
    return df_all


def evaluate_all():
    model_folders = sorted([f for f in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, f))])
    
    all_results = []
    
    for dataset_name, data_path in TEST_DATASETS.items():
        df_test = load_test_data(data_path)
        if df_test is None or len(df_test) == 0:
            continue

        target_col = 'label'
        drop_cols = [target_col, 'speaker_id']
        for col in ['filename', 'name', 'path', 'dosya_adi']:
            if col in df_test.columns: drop_cols.append(col)
            
        X_test_raw = df_test.drop(columns=drop_cols, errors='ignore').values
        y_test_raw = df_test[target_col].values

        for mf in model_folders:
            model_path_dir = os.path.join(MODELS_DIR, mf)
            model_files = glob.glob(os.path.join(model_path_dir, "*model*.*")) + glob.glob(os.path.join(model_path_dir, "*.h5")) + glob.glob(os.path.join(model_path_dir, "*.pkl"))
            scaler_files = glob.glob(os.path.join(model_path_dir, "scaler_*.pkl"))
            le_files = glob.glob(os.path.join(model_path_dir, "label_encoder_*.pkl"))
            
            model_file = next((f for f in model_files if 'model' in os.path.basename(f) or f.endswith('.h5')), None)
            if not model_file and model_files:
                model_file = [f for f in model_files if not 'scaler' in f and not 'encoder' in f][0]
                
            scaler_file = scaler_files[0] if scaler_files else None
            le_file = le_files[0] if le_files else None
            
            if not model_file or not scaler_file or not le_file:
                continue
                
            try:
                scaler = joblib.load(scaler_file)
                le = joblib.load(le_file)
                
                if model_file.endswith('.h5'):
                    if not TF_AVAILABLE: continue
                    model = load_model(model_file)
                else:
                    model = joblib.load(model_file)
                    
                y_true = le.transform(y_test_raw)
                X_scaled = scaler.transform(X_test_raw)
                
                if 'cnn1d' in mf.lower():
                    X_input = np.expand_dims(X_scaled, axis=2)
                    preds = model.predict(X_input, verbose=0)
                    y_pred = np.argmax(preds, axis=1)
                elif 'dnn' in mf.lower():
                    preds = model.predict(X_scaled, verbose=0)
                    y_pred = np.argmax(preds, axis=1)
                else:
                    y_pred = model.predict(X_scaled)
                    if hasattr(y_pred, 'flatten'): y_pred = y_pred.flatten()
                        
                acc = accuracy_score(y_true, y_pred)
                report = classification_report(y_true, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)
                
                res_dict = {
                    'Dataset': dataset_name,
                    'Model': mf,
                    'Accuracy': acc * 100,
                    'Macro_F1': report['macro avg']['f1-score'],
                    'Macro_Recall': report['macro avg']['recall']
                }
                
                # Duygu bazlı F1 skorlarını ekle
                for emo in le.classes_:
                    res_dict[f'F1_{emo.capitalize()}'] = report[emo]['f1-score']
                
                all_results.append(res_dict)
                
            except Exception as e:
                print(f"Hata: {mf} -> {e}")
                
    if all_results:
        df_res = pd.DataFrame(all_results)
        
        for dataset in TEST_DATASETS.keys():
            print(f"\n" + "="*80)
            print(f" DETAYLI DUYGU ANALİZİ: {dataset} ")
            print("="*80)
            sub_df = df_res[df_res['Dataset'] == dataset].drop(columns=['Dataset'])
            sub_df = sub_df.sort_values(by="Accuracy", ascending=False)
            
            # Daha okunaklı bir tablo çıktısı
            print(sub_df.to_string(index=False, float_format=lambda x: "{:.3f}".format(x) if x < 1 else "{:.2f}".format(x)))
            
        df_res.to_csv(os.path.join(RESULTS_DIR, "emotion_breakdown_report.csv"), index=False)
        
if __name__ == "__main__":
    evaluate_all()
