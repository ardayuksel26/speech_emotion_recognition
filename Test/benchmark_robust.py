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

# Test Araçları
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

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

def run_robust_benchmark():
    print("\n" + "="*75)
    print("--- 🌪️ DIŞ SES / GÜRÜLTÜLÜ (ROBUST) MODELLERİN BÜYÜK KAPIŞMASI 🌪️ ---")
    print("="*75 + "\n")
    
    df = load_data()
    if df is None: return

    # Ön İşleme
    target_col = 'label'
    drop_cols = [target_col, 'filename', 'name', 'path', 'dosya_adi']
    
    # ⚠️ Cross-validation için sadece Orijinal datayı alıyoruz
    X_clean = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore').to_numpy()
    y_raw = df[target_col].to_numpy()

    # Encoding
    le = LabelEncoder()
    y_clean = le.fit_transform(y_raw)

    # --- ROBUST EĞİTİMDE KULLANILAN PARAMETRELERLE MODELLER ---
    models = [
        ('CatBoost Robust', CatBoostClassifier(iterations=600, learning_rate=0.08, depth=6, verbose=0, random_seed=42)),
        ('XGBoost Robust', xgb.XGBClassifier(n_estimators=300, learning_rate=0.08, max_depth=6, use_label_encoder=False, eval_metric='mlogloss', random_state=42)),
        ('LightGBM Robust', lgb.LGBMClassifier(n_estimators=300, learning_rate=0.08, max_depth=6, verbosity=-1, random_state=42)),
        ('Random Forest Rb', RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)),
        ('Grad Boosting Rb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=3, random_state=42)),
        ('SVM Robust', SVC(kernel='rbf', probability=True, random_state=42)),
        ('MLP Neural Net Rb', MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, random_state=42)),
        ('k-NN Robust', KNeighborsClassifier(n_neighbors=5))
    ]

    results = []
    names = []

    # Tablo Başlığı
    print(f"{'MODEL':<20} | {'DURUM':<40}")
    print("-" * 75)

    # HER MODEL İÇİN DÖNGÜ
    for name, model in models:
        print(f"\n   📍 {name} Dış Ses (Robust) Analizi Başlıyor:")
        
        # Veri sızıntısını (Leakage) önlemek için önce Split yapıp sonra Noise ekliyoruz!
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_scores = []
        
        start_total = time.time()
        
        for i, (train_index, test_index) in enumerate(kfold.split(X_clean, y_clean), 1):
            
            # 1. Pürüzsüz veriyi Train ve Test olarak KESİYORUZ.
            X_train_c, X_test_c = X_clean[train_index], X_clean[test_index]
            y_train_c, y_test_c = y_clean[train_index], y_clean[test_index]
            
            # 2. Sadece TRAIN verisini 3 katına çıkarıp yapay gürültü ile bozuyoruz
            X_train_light = X_train_c + np.random.normal(0, 0.03, X_train_c.shape)
            X_train_heavy = X_train_c + np.random.normal(0, 0.08, X_train_c.shape)
            
            X_train_robust = np.vstack((X_train_c, X_train_light, X_train_heavy))
            y_train_robust = np.concatenate((y_train_c, y_train_c, y_train_c))
            
            # 3. Modelin dayanıklılığını (robustness) gerçekçi test etmek için
            # Test validasyon verisine (X_test) mikrofondan gelmiş gibi orta seviye gürültü ekliyoruz.
            X_test_noisy = X_test_c + np.random.normal(0, 0.05, X_test_c.shape)

            # 4. Scaling işlemini yapıyoruz (Robust veriye göre fit ediyoruz)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_robust)
            X_test_scaled = scaler.transform(X_test_noisy)
            
            # 5. Süre Başlat
            start_fold = time.time()
            
            # 6. Eğit ve Tahmin Et
            model.fit(X_train_scaled, y_train_robust)
            y_pred = model.predict(X_test_scaled)
            
            # Catboost flatten fix
            if hasattr(y_pred, 'flatten') and not isinstance(model, MLPClassifier):
                y_pred = y_pred.flatten()
            
            # 7. Skoru Hesapla
            acc = accuracy_score(y_test_c, y_pred)
            fold_scores.append(acc)
            
            # 8. Süre Bitir
            elapsed_fold = time.time() - start_fold
            
            # Ekrana Yaz (Anlık Durum)
            print(f"      🔹 Fold {i}/5 | Validasyon: %{acc*100:.2f} | Süre: {elapsed_fold:.2f}sn")

        # Model Özeti
        mean_score = np.mean(fold_scores)
        std_dev = np.std(fold_scores)
        elapsed_total = time.time() - start_total
        
        results.append(fold_scores)
        names.append(name)
        
        print(f"   ✅ {name} BİTTİ! -> Ort: %{mean_score*100:.2f} (+/- {std_dev:.4f}) | Toplam: {elapsed_total:.2f}sn")
        print("-" * 75)

    # --- GRAFİK ÇİZ ---
    print("\n📊 Grafik oluşturuluyor...")
    plt.figure(figsize=(14, 7))
    plt.boxplot(results, labels=names, patch_artist=True)
    plt.title('Dış Ses/Gürültülü Modellerin Zorlu K-Fold Karşılaştırması (Leakage-Free Test)', fontsize=14)
    plt.ylabel('Başarı Oranı (Accuracy)')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('benchmark_robust_results.png')
    print("Graph saved to benchmark_robust_results.png (Test klasörüne kaydedildi.)")
    print("🏁 İşlem Tamamlandı.")

if __name__ == "__main__":
    run_robust_benchmark()
