import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time  # Süre tutmak için ekledik

# Modeller
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Test Araçları
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score # Skoru manuel hesaplayacağız

# --- AYARLAR ---
DATA_DIR = "../TurEV-DB/Extracted CSV" 
EMOTION_FILES = {"angry.csv": "angry", "calm.csv": "calm", "happy.csv": "happy", "sad.csv": "sad"}

def load_data():
    all_dfs = []
    if not os.path.exists(DATA_DIR):
        print(f"HATA: Veri klasörü bulunamadı: {DATA_DIR}")
        print("Lütfen path ayarını kontrol et.")
        return None

    for fname, label in EMOTION_FILES.items():
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['label'] = label
            all_dfs.append(df)
    
    if not all_dfs: return None
    return pd.concat(all_dfs, ignore_index=True)

def run_benchmark():
    print("\n" + "="*60)
    print("--- 🏁 BÜYÜK KAPIŞMA BAŞLIYOR (DETAYLI MOD) 🏁 ---")
    print("="*60 + "\n")
    
    df = load_data()
    if df is None: return

    # Ön İşleme
    target_col = 'label'
    drop_cols = [target_col, 'filename', 'name', 'path', 'dosya_adi']
    
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    y = df[target_col]

    # Encoding & Scaling
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # --- MODELLERİN LİSTESİ ---
    models = [
        ('CatBoost', CatBoostClassifier(iterations=200, learning_rate=0.1, depth=6, verbose=0, random_seed=42)),
        ('XGBoost', xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss', random_state=42)),
        ('LightGBM', lgb.LGBMClassifier(n_estimators=100, verbosity=-1, random_state=42)),
        ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ('SVM', SVC(kernel='rbf', probability=True, random_state=42)),
        ('MLP (Neural Net)', MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)),
        ('k-NN', KNeighborsClassifier(n_neighbors=5))
    ]

    results = []
    names = []

    # Tablo Başlığı
    print(f"{'MODEL':<20} | {'DURUM':<40}")
    print("-" * 70)

    # HER MODEL İÇİN DÖNGÜ
    for name, model in models:
        print(f"👉 {name:<18} | Başlıyor...", end="\r") # Satırı silip yeniden yazmak için \r
        
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_scores = []
        
        print(f"\n   📍 {name} Analizi:")
        
        # HER FOLD (PARÇA) İÇİN MANUEL DÖNGÜ
        start_total = time.time()
        
        for i, (train_index, test_index) in enumerate(kfold.split(X, y), 1):
            # 1. Veriyi Böl
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # 2. Süre Başlat
            start_fold = time.time()
            
            # 3. Eğit ve Tahmin Et
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # 4. Skoru Hesapla
            acc = accuracy_score(y_test, y_pred)
            fold_scores.append(acc)
            
            # 5. Süre Bitir
            elapsed_fold = time.time() - start_fold
            
            # Ekrana Yaz (Anlık Durum)
            print(f"      🔹 Fold {i}/5 Tamamlandı | Süre: {elapsed_fold:.2f}sn | Skor: %{acc*100:.2f}")

        # Model Özeti
        mean_score = np.mean(fold_scores)
        std_dev = np.std(fold_scores)
        elapsed_total = time.time() - start_total
        
        results.append(fold_scores)
        names.append(name)
        
        print(f"   ✅ {name} BİTTİ! -> Ort: %{mean_score*100:.2f} (+/- {std_dev:.4f}) | Toplam: {elapsed_total:.2f}sn")
        print("-" * 70)

    # --- GRAFİK ÇİZ ---
    # --- GRAFİK ÇİZ ---
    print("\n📊 Grafik oluşturuluyor...")
    plt.figure(figsize=(12, 6))
    plt.boxplot(results, labels=names, patch_artist=True)
    plt.title('Modellerin Detaylı Cross-Validation Karşılaştırması')
    plt.ylabel('Başarı Oranı (Accuracy)')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    print("Graph saved to benchmark_results.png")
    # plt.show()
    print("🏁 İşlem Tamamlandı.")

if __name__ == "__main__":
    run_benchmark()