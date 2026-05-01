import os
import glob
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SENTENCE_VOICE_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "sentencevoice_test"))
REPORT_PATH = os.path.join(BASE_DIR, "train2_seperate_models_benchmark_results.txt")
CHART_PATH = os.path.join(BASE_DIR, "train2_seperate_models_comparison.png")

API_URL = "http://127.0.0.1:5000/api/predict_advanced_sentence"

EMOTIONS = ["angry", "calm", "happy", "sad"]

def run_test():
    print("="*60)
    print(" 🚀 TRAIN_2 MODELLERİ (MODELS_2) AYRI AYRI TEST BAŞLIYOR... ")
    print(" Hedef: Production API (localhost:5000/api/predict_advanced_sentence)")
    print("="*60)
    
    # Model isimleri (app.py'deki MODEL_DISPLAY_NAMES'e göre)
    # İlk veriden dinamik olarak çekeceğiz
    y_true = []
    
    # Her model için tahminleri tutacağımız sözlük
    # Örn: {'LightGBM (V2)': ['happy', 'sad', ...], 'Random Forest (V2)': [...]}
    y_preds = {}
    
    total_processed = 0
    
    for emo in EMOTIONS:
        folder = os.path.join(SENTENCE_VOICE_DIR, emo.capitalize())
        if not os.path.exists(folder):
            folder = os.path.join(SENTENCE_VOICE_DIR, emo)
            if not os.path.exists(folder):
                print(f"Uyarı: {folder} bulunamadı.")
                continue
            
        wav_files = glob.glob(os.path.join(folder, "*.wav"))
        print(f"\n⏳ İRDELENİYOR: {emo.upper()} Kategorisi ({len(wav_files)} dosya)")
        
        count = 0
        for wav_path in wav_files:
            try:
                with open(wav_path, "rb") as f:
                    response = requests.post(API_URL, files={"file": f})
                    
                if response.status_code == 200:
                    data = response.json()
                    model_details = data.get("model_details", [])
                    
                    y_true.append(emo)
                    
                    for detail in model_details:
                        m_name = detail['model']
                        pred = detail['prediction'].lower()
                        
                        if m_name not in y_preds:
                            # İlk defa görüyorsak, önceki (kayıp) değerleri 'unknown' ile dolduralım
                            # Normalde hepsi aynı anda gelir ama güvenli olması için
                            y_preds[m_name] = ['unknown'] * total_processed
                            
                        y_preds[m_name].append(pred)
                        
                    count += 1
                    total_processed += 1
                else:
                    print(f"  [!] HTTP Hatası: {response.status_code} - {os.path.basename(wav_path)}")
            except Exception as e:
                print(f"  [!] İstek Hatası: {str(e)} - {os.path.basename(wav_path)}")
                
        print(f"  -> {count}/{len(wav_files)} ses başarıyla yorumlandı.")
        
    if total_processed == 0:
        print("❌ Test edilecek dosya bulunamadı veya API kapalı.")
        return
        
    print("\n" + "="*60)
    print(" 📊 SONUÇLAR HESAPLANIYOR...")
    print("="*60)
    
    # Rapor ve Grafikler için veriler
    metrics_list = []
    
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("================================================================================\n")
        f.write(" 🧪 TRAIN_2 (MODELS_2) - TEKİL MODELLERİN CÜMLE (SENTENCE) TEST RAPORU\n")
        f.write("================================================================================\n")
        f.write("Açıklama: Bu test, 5'li ensemble içerisinde yer alan her bir modelin cümleye \n")
        f.write("verdiği tekil tepkileri ve başarı oranlarını analiz etmek için yapılmıştır.\n")
        f.write("================================================================================\n\n")
        
        for m_name, preds in y_preds.items():
            acc = accuracy_score(y_true, preds)
            report_dict = classification_report(y_true, preds, target_names=EMOTIONS, output_dict=True, zero_division=0)
            precision, recall, fscore, _ = precision_recall_fscore_support(y_true, preds, labels=EMOTIONS, zero_division=0)
            macro_f1 = report_dict['macro avg']['f1-score']
            
            print(f"Model: {m_name} | Accuracy: %{acc*100:.2f} | Macro F1: {macro_f1:.4f}")
            
            # Text Raporuna Yazdır
            f.write(f"--- MODEL: {m_name} ---\n")
            f.write("Duygu      | Precision  | Recall     | F1-Score  \n")
            f.write("-------------------------------------------------------\n")
            for i, class_name in enumerate(EMOTIONS):
                f.write(f"{class_name.capitalize().ljust(10)} | {precision[i]:.4f}     | {recall[i]:.4f}   | {fscore[i]:.4f}\n")
            f.write("-------------------------------------------------------\n")
            f.write(f"Accuracy   : %{acc*100:.2f}\n")
            f.write(f"Macro F1   : {macro_f1:.4f}\n\n")
            
            # Grafik verisine ekle
            for i, emo in enumerate(EMOTIONS):
                metrics_list.append({
                    'Model': m_name.replace(' (V2)', ''),
                    'Duygu': emo.capitalize(),
                    'F1-Score': fscore[i]
                })
                
    print(f"\n✅ TXT Kaydedildi: {REPORT_PATH}")
    
    # Karşılaştırmalı Grafik Çizimi (F1-Score üzerinden)
    df_metrics = pd.DataFrame(metrics_list)
    
    plt.figure(figsize=(14, 8))
    sns.barplot(data=df_metrics, x="Duygu", y="F1-Score", hue="Model", palette="tab10")
    plt.title("Train_2 (Models_2) Tekil Model Performansları (F1-Score Karşılaştırması)")
    plt.ylim(0, 1.05)
    plt.axhline(0.5, ls='--', color='gray', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150)
    plt.close()
    
    print(f"📸 Grafik Kaydedildi: {CHART_PATH}")

if __name__ == "__main__":
    run_test()
