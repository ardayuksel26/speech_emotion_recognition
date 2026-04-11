import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import logging

# Modeller
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Keras / TensorFlow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.utils import to_categorical

# Test Araçları
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

# --- AYARLAR ---
DATA_DIR = "../TurEV-DB/Extracted CSV" 
EMOTION_FILES = {"angry.csv": "angry", "calm.csv": "calm", "happy.csv": "happy", "sad.csv": "sad"}
RESULTS_DIR = "TestsResults"

os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data():
    all_dfs = []
    if not os.path.exists(DATA_DIR):
        print(f"HATA: Veri klasörü bulunamadı: {DATA_DIR}")
        return None

    for fname, label in EMOTION_FILES.items():
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['label'] = label
            all_dfs.append(df)
    
    if not all_dfs: return None
    return pd.concat(all_dfs, ignore_index=True)

def build_keras_dnn_robust(input_dim, num_classes):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_dim,)),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_keras_cnn1d_robust(input_dim, num_classes):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(input_dim, 1)),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def run_robust_benchmark():
    print("\n" + "="*75)
    print("--- 🌪️ DIŞ SES / GÜRÜLTÜLÜ (ROBUST) 10 MODEL BÜYÜK KAPIŞMASI 🌪️ ---")
    print("="*75 + "\n")
    
    df = load_data()
    if df is None: return

    target_col = 'label'
    drop_cols = [target_col, 'filename', 'name', 'path', 'dosya_adi']
    
    X_clean = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore').to_numpy()
    y_raw = df[target_col].to_numpy()

    le = LabelEncoder()
    y_clean = le.fit_transform(y_raw)
    num_classes = len(np.unique(y_clean))

    models = [
        ('CatBoost Robust', CatBoostClassifier(iterations=600, learning_rate=0.08, depth=6, verbose=0, random_seed=42)),
        ('XGBoost Robust', xgb.XGBClassifier(n_estimators=300, learning_rate=0.08, max_depth=6, use_label_encoder=False, eval_metric='mlogloss', random_state=42)),
        ('LightGBM Robust', lgb.LGBMClassifier(n_estimators=300, learning_rate=0.08, max_depth=6, verbosity=-1, random_state=42)),
        ('Random Forest Rb', RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)),
        ('Grad Boosting Rb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=3, random_state=42)),
        ('SVM Robust', SVC(kernel='rbf', probability=True, random_state=42)),
        ('MLP Neural Net Rb', MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, random_state=42)),
        ('k-NN Robust', KNeighborsClassifier(n_neighbors=5)),
        ('DNN Robust (Keras)', 'keras_dnn'),
        ('CNN1D Robust (Keras)', 'keras_cnn1d')
    ]

    results = []
    names = []

    txt_report_path = os.path.join(RESULTS_DIR, "word_methods_robust_benchmark_results.txt")
    with open(txt_report_path, "w", encoding="utf-8") as rf:
        rf.write("================================================================================\n")
        rf.write(" GÜRÜLTÜLÜ (ROBUST) KELİME BAZLI MODEL PERFORMANS RAPORU\n")
        rf.write("================================================================================\n\n")

    print(f"{'MODEL':<20} | {'DURUM':<40}")
    print("-" * 75)

    for name, model_obj in models:
        print(f"\n   📍 {name} Dış Ses (Robust) Analizi Başlıyor:")
        
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_scores = []
        start_total = time.time()
        
        for i, (train_index, test_index) in enumerate(kfold.split(X_clean, y_clean), 1):
            
            X_train_c, X_test_c = X_clean[train_index], X_clean[test_index]
            y_train_c, y_test_c = y_clean[train_index], y_clean[test_index]
            
            X_train_light = X_train_c + np.random.normal(0, 0.03, X_train_c.shape)
            X_train_heavy = X_train_c + np.random.normal(0, 0.08, X_train_c.shape)
            
            X_train_robust = np.vstack((X_train_c, X_train_light, X_train_heavy))
            y_train_robust = np.concatenate((y_train_c, y_train_c, y_train_c))
            
            X_test_noisy = X_test_c + np.random.normal(0, 0.05, X_test_c.shape)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_robust)
            X_test_scaled = scaler.transform(X_test_noisy)
            
            start_fold = time.time()
            
            if model_obj == 'keras_dnn':
                model = build_keras_dnn_robust(X_train_scaled.shape[1], num_classes)
                y_train_cat = to_categorical(y_train_robust, num_classes)
                model.fit(X_train_scaled, y_train_cat, epochs=15, batch_size=32, verbose=0)
                preds = model.predict(X_test_scaled, verbose=0)
                y_pred = np.argmax(preds, axis=1)
                
            elif model_obj == 'keras_cnn1d':
                model = build_keras_cnn1d_robust(X_train_scaled.shape[1], num_classes)
                y_train_cat = to_categorical(y_train_robust, num_classes)
                X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
                X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
                model.fit(X_train_cnn, y_train_cat, epochs=15, batch_size=32, verbose=0)
                preds = model.predict(X_test_cnn, verbose=0)
                y_pred = np.argmax(preds, axis=1)
                
            else:
                model_obj.fit(X_train_scaled, y_train_robust)
                y_pred = model_obj.predict(X_test_scaled)
                if hasattr(y_pred, 'flatten') and not isinstance(model_obj, MLPClassifier):
                    y_pred = y_pred.flatten()

            acc = accuracy_score(y_test_c, y_pred)
            fold_scores.append(acc)
            elapsed_fold = time.time() - start_fold
            
            print(f"      🔹 Fold {i}/5 | Validasyon: %{acc*100:.2f} | Süre: {elapsed_fold:.2f}sn")

        mean_score = np.mean(fold_scores)
        std_dev = np.std(fold_scores)
        elapsed_total = time.time() - start_total
        
        results.append(fold_scores)
        names.append(name)
        
        summary = f"   ✅ {name} BİTTİ! -> Ort: %{mean_score*100:.2f} (+/- {std_dev:.4f}) | Toplam: {elapsed_total:.2f}sn"
        print(summary)
        print("-" * 75)
        
        with open(txt_report_path, "a", encoding="utf-8") as rf:
            rf.write(f"Model: {name}\n")
            rf.write(f"Ortalama Accuracy: %{mean_score*100:.4f}\n")
            rf.write(f"Standart Sapma: {std_dev:.4f}\n")
            rf.write(f"Fold Skorları: {[round(s*100, 2) for s in fold_scores]}\n")
            rf.write("-" * 50 + "\n")

    print("\n📊 Grafik oluşturuluyor...")
    plt.figure(figsize=(14, 7), facecolor='#0f172a')
    ax = plt.axes()
    ax.set_facecolor('#0f172a')
    
    box = plt.boxplot(results, labels=names, patch_artist=True)
    
    # Koyu Tema Uyumu
    for patch in box['boxes']:
        patch.set_facecolor('#ef4444')
        patch.set_edgecolor('white')
    for median in box['medians']:
        median.set_color('#f59e0b')
        median.set_linewidth(2)
        
    plt.title('Gürültülü (Robust) Modellerin 5-Fold Karşılaştırması (Leakage-Free)', color='white', fontsize=14, pad=15)
    plt.ylabel('Başarı Oranı (Accuracy)', color='white')
    plt.xticks(rotation=45, color='cyan')
    plt.yticks(color='white')
    plt.grid(True, linestyle='--', alpha=0.3, color='gray')
    plt.tight_layout()
    
    img_path = os.path.join(RESULTS_DIR, 'word_methods_robust_benchmark_results.png')
    plt.savefig(img_path, dpi=300, facecolor='#0f172a', edgecolor='none')
    plt.close()
    
    print(f"Graph saved to {img_path}")
    print(f"Text report saved to {txt_report_path}")
    print("🏁 İşlem Tamamlandı.")

if __name__ == "__main__":
    run_robust_benchmark()
