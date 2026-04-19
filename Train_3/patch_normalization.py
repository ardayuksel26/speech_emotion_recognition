import os
import glob

def patch_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Eğer zaten patchlenmişse atla
    if 'speaker_id' in content and 'groupby' in content:
        print(f"Skipping {filepath}, already patched.")
        return

    # Normalizasyon fonksiyonunu ekle
    norm_func = """
def apply_speaker_normalization(df):
    \"\"\"Her konuşmacıyı kendi içinde normalize ederek bireysel farkları ortadan kaldırır.\"\"\"
    target_col = 'label'
    speaker_col = 'speaker_id'
    
    # Sayısal sütunları bul (label ve speaker_id hariç)
    skip_cols = [target_col, speaker_col, 'filename', 'name', 'path', 'dosya_adi']
    feat_cols = [c for c in df.columns if c not in skip_cols and df[c].dtype in ['float64', 'int64']]
    
    # Grupla ve normalize et
    df[feat_cols] = df.groupby(speaker_col)[feat_cols].transform(lambda x: (x - x.mean()) / (x.std() + 1e-6))
    return df
"""
    
    # Fonksiyonu dosyanın üst kısımlarına (load_data'dan öncesine) ekle
    content = content.replace('def load_data():', norm_func + '\ndef load_data():')

    # Eğitim fonksiyonu içinde çağrıyı ekle
    # Genelde 'df = load_data()' satırından sonra gelir
    patch_call = """
    df = load_data()
    if df is None: return
    
    print("Konuşmacı bazlı normalizasyon uygulanıyor...")
    df = apply_speaker_normalization(df)
"""
    content = content.replace('    df = load_data()\n    if df is None: return', patch_call)

    # speaker_id'nin X'e girmesini ama model eğitiminden önce atılmasını sağla
    # drop_cols içine ekleyelim
    content = content.replace("drop_cols = [target_col]", "drop_cols = [target_col, 'speaker_id']")

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Patched {filepath}")

if __name__ == "__main__":
    train_files = glob.glob("train_*.py")
    for f in train_files:
        patch_file(f)
