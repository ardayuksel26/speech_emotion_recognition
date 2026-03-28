import os
import glob
import requests
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SENTENCE_VOICE_DIR = os.path.join(BASE_DIR, "sentencevoice")
REPORT_PATH = os.path.join(BASE_DIR, "mastermind_benchmark_results.txt")
CHART_PATH = os.path.join(BASE_DIR, "mastermind_comparison.png")

API_URL = "http://localhost:5000/api/predict_mastermind"

EMOTIONS = ["angry", "calm", "happy", "sad"]

def run_test():
    print("="*60)
    print(" 🚀 THE MASTERMIND (ÜST AKIL) ALGORİTMASI TESTİ BAŞLIYOR... ")
    print(" Hedef: Production API (localhost:5000/api/predict_mastermind)")
    print("="*60)
    
    y_true = []
    y_pred = []
    
    veto_count = 0
    total_processed = 0
    
    detailed_logs = []
    
    for emo in EMOTIONS:
        folder = os.path.join(SENTENCE_VOICE_DIR, emo.capitalize())
        if not os.path.exists(folder):
            print(f"Uyarı: {folder} bulunamadı.")
            continue
            
        wav_files = glob.glob(os.path.join(folder, "*.wav"))
        print(f"\n⏳ İRDELENİYOR: {emo.upper()} Kategorisi ({len(wav_files)} dosya)")
        
        count = 0
        for wav_path in wav_files:
            try:
                with open(wav_path, "rb") as f:
                    response = requests.post(API_URL, files={"audio": f})
                    
                if response.status_code == 200:
                    data = response.json()
                    pred_emotion = data.get("final_emotion", "unknown").lower()
                    
                    y_true.append(emo)
                    y_pred.append(pred_emotion)
                    
                    if data.get("veto_applied"):
                        veto_count += 1
                        
                    rf_sad = data.get('rf_sad_score', '%0.00')
                    log_msg = f"[Detay] Gerçek Sınıf: {emo.upper():<6} | Üst Akıl Kararı: {pred_emotion.upper():<6} | RF_Sad_Skoru: {rf_sad}"
                    print(f"    {log_msg}")
                    detailed_logs.append(log_msg)
                        
                    count += 1
                    total_processed += 1
                else:
                    print(f"  [!] HTTP Hatası: {response.status_code} - {wav_path}")
            except Exception as e:
                print(f"  [!] İstek Hatası: {str(e)} - {wav_path}")
                
        print(f"  -> {count}/{len(wav_files)} ses başarıyla yorumlandı.")
        
    if total_processed == 0:
        print("❌ Test edilecek dosya bulunamadı veya API kapalı.")
        return
        
    print("\n" + "="*60)
    print(" 📊 SONUÇLAR HESAPLANIYOR...")
    print("="*60)
    
    acc = accuracy_score(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, target_names=EMOTIONS, output_dict=True, zero_division=0)
    report_str = classification_report(y_true, y_pred, target_names=EMOTIONS, zero_division=0)
    
    print("\n" + report_str)
    print(f"Genel Doğruluk (Accuracy): %{acc*100:.2f}")
    print(f"RF_Robust Vetosu Toplam {veto_count} kez devredeydi.")
    
    # TXT Raporu Oluşturma
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("================================================================================\n")
        f.write(" 👑 THE MASTERMIND (ÜST AKIL) - AKADEMİK BENCHMARK RAPORU\n")
        f.write("================================================================================\n")
        f.write("Açıklama: Bu test doğrudan Backend Production API'sine atılan gerçek HTTP\n")
        f.write("istekleriyle yapılmış olup, sistemin canlı (%60/40 Harman + VETO)\n")
        f.write("performansını yansıtmaktadır.\n")
        f.write(f"RF VETO Sistemi {veto_count} kez müdahale etmiştir.\n")
        f.write("================================================================================\n\n")
        
        f.write("Duygu      | Precision  | Recall     | F1-Score  \n")
        f.write("-------------------------------------------------------\n")
        precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, labels=EMOTIONS, zero_division=0)
        
        for i, class_name in enumerate(EMOTIONS):
            f.write(f"{class_name.capitalize().ljust(10)} | {precision[i]:.4f}     | {recall[i]:.4f}   | {fscore[i]:.4f}\n")
            
        f.write("-------------------------------------------------------\n")
        f.write(f"Accuracy  : {acc:.4f}\n")
        f.write(f"Macro F1  : {report_dict['macro avg']['f1-score']:.4f}\n")
        f.write("================================================================================\n\n")
        
        f.write("🔎 DETAYLI DOSYA BAZLI ANALİZ DÖKÜMÜ:\n")
        f.write("--------------------------------------------------------------------------------\n")
        for log in detailed_logs:
            f.write(log + "\n")
        f.write("================================================================================\n")
        
    print(f"\n✅ TXT Kaydedildi: {REPORT_PATH}")
    
    # Grafikleri Çizdir
    df_metrics = pd.DataFrame({
        'Duygu': EMOTIONS,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': fscore
    })
    df_melted = df_metrics.melt(id_vars="Duygu", var_name="Metrik", value_name="Skor")
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melted, x="Duygu", y="Skor", hue="Metrik", palette="viridis")
    plt.title(f"Üst Akıl (Mastermind) Performans Değerleri\nAccuracy: %{acc*100:.1f}")
    plt.ylim(0, 1.05)
    plt.axhline(0.5, ls='--', color='gray', alpha=0.5)
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150)
    plt.close()
    
    print(f"📸 Grafik Kaydedildi: {CHART_PATH}")

if __name__ == "__main__":
    run_test()
