import os
import glob
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SENTENCE_VOICE_DIR = os.path.join(BASE_DIR, "sentencevoice")
CM_IMAGE_PATH = os.path.join(BASE_DIR, "mastermind_confusion_matrix.png")

API_URL = "http://localhost:5000/api/predict_mastermind"
EMOTIONS = ["angry", "calm", "happy", "sad"]

def generate_cm():
    print("==========================================================")
    print(" 🧠 ÜST AKIL (MASTERMIND) CONFUSION MATRIX OLUŞTURUCU")
    print(" Lütfen Backend API'nin 5000 portunda açık olduğundan emin olun.")
    print("==========================================================")

    y_true = []
    y_pred = []
    
    total_processed = 0
    total_files = sum([len(files) for r, d, files in os.walk(SENTENCE_VOICE_DIR)])
    
    if total_files == 0:
         print(f"❌ {SENTENCE_VOICE_DIR} dizininde test edilecek ses dosyası bulunamadı.")
         return

    print("Veriler analiz ediliyor...\n")

    for emo in EMOTIONS:
        folder = os.path.join(SENTENCE_VOICE_DIR, emo.capitalize())
        if not os.path.exists(folder):
            continue
            
        wav_files = glob.glob(os.path.join(folder, "*.wav"))
        for wav_path in wav_files:
            try:
                with open(wav_path, "rb") as f:
                    response = requests.post(API_URL, files={"audio": f})
                    
                if response.status_code == 200:
                    data = response.json()
                    pred_emotion = data.get("final_emotion", "unknown").lower()
                    
                    y_true.append(emo)
                    y_pred.append(pred_emotion)
                    
                    total_processed += 1
                    
                    # Basit ilerleme çubuğu
                    percent = (total_processed / total_files) * 100
                    print(f"\rİlerleme: %{percent:.1f} [{total_processed}/{total_files}]  - Son Tahmin: {emo.upper()} -> {pred_emotion.upper()}", end="")
            except Exception as e:
                print(f"\n[!] Hata ({wav_path}): {e}")
                
    print("\n\n✅ Analiz tamamlandı. Matris çiziliyor...")

    # Heatmap Oluştur
    cm = confusion_matrix(y_true, y_pred, labels=EMOTIONS)
    
    # Görsel Estetiğini 2026 Standartlarına Yükseltme
    plt.figure(figsize=(10, 8), facecolor='#0f172a') # Koyu Arka Plan
    ax = plt.axes()
    ax.set_facecolor('#0f172a')
    
    sns.heatmap(cm, annot=True, fmt="d", cmap="inferno", 
                xticklabels=[e.capitalize() for e in EMOTIONS], 
                yticklabels=[e.capitalize() for e in EMOTIONS],
                linewidths=.5, linecolor='gray', 
                cbar_kws={'label': 'Örnek Sayısı'})
                
    plt.title('Mastermind API - Karmaşıklık Matrisi (Confusion Matrix)', color='white', fontsize=16, pad=20)
    plt.xlabel('Tahmin Edilen Sınıf', color='white', fontsize=12, labelpad=15)
    plt.ylabel('Gerçek Sınıf', color='white', fontsize=12, labelpad=15)
    plt.xticks(color='cyan', fontsize=10)
    plt.yticks(color='cyan', fontsize=10, rotation=0)

    # Sağ kısımdaki renk skalasındaki etiketleri beyaz yap
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    plt.tight_layout()
    plt.savefig(CM_IMAGE_PATH, dpi=300, facecolor='#0f172a', edgecolor='none')
    plt.close()
    
    print(f"🎉 Karmaşıklık Matrisi (Confusion Matrix) başarıyla kaydedildi:\n -> {CM_IMAGE_PATH}")
    print("\nDetaylı Rapor:")
    print(classification_report(y_true, y_pred, target_names=[e.capitalize() for e in EMOTIONS], zero_division=0))

if __name__ == "__main__":
    generate_cm()
