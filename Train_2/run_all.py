import subprocess
import sys
import time
import os

def run_all():
    # Bu dosyanın bulunduğu dizini tam yol olarak al (Train2 dizini)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Train2 dizinindeki train_ ile başlayan tüm .py dosyalarını tam yollarıyla bul
    SCRIPTS = [
        os.path.join(current_dir, f) 
        for f in os.listdir(current_dir) 
        if f.startswith("train_") and f.endswith(".py")
    ]
    SCRIPTS.sort() # Alfabetik sıra ile çalıştır

    total = len(SCRIPTS)
    if total == 0:
        print("❌ Çalıştırılacak eğitim betiği bulunamadı!")
        return

    print("=" * 55)
    print(f"🚀 Toplam {total} model eğitimi bulundu:")
    for s in SCRIPTS:
        print(f"  - {os.path.basename(s)}")
    print("=" * 55)
    time.sleep(2) # Listeyi görmen için kısa bir bekleme

    for i, script in enumerate(SCRIPTS, 1):
        script_name = os.path.basename(script)
        print("\n" + "=" * 55)
        print(f"  [{i}/{total}] {script_name} ÇALIŞTIRILIYOR...")
        print("=" * 55)
        
        start = time.time()
        
        # Scripti proje kök dizininde (root) çalıştır
        # current_dir Train2 olduğu için, bir üst dizin root'tur.
        project_root = os.path.dirname(current_dir)
        result = subprocess.run([sys.executable, script], cwd=project_root)
        
        elapsed = time.time() - start
        mins, secs = divmod(int(elapsed), 60)
        
        if result.returncode == 0:
            print(f"\n✅ {script_name} Tamamlandı ({mins}dk {secs}sn)")
        else:
            print(f"\n❌ HATA — {script_name} başarısız oldu (kod: {result.returncode})")
            print("Devam etmek için Enter'a bas, çıkmak için Ctrl+C...")
            try:
                input()
            except KeyboardInterrupt:
                print("\nEğitim kullanıcı tarafından durduruldu.")
                sys.exit(1)

    print("\n" + "=" * 55)
    print("✨ TÜM EĞİTİMLER BAŞARIYLA TAMAMLANDI ✨")
    print("=" * 55)

if __name__ == "__main__":
    run_all()
