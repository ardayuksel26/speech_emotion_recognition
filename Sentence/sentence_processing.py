import librosa
import numpy as np
import os
import uuid
import soundfile as sf
import warnings

# Suppress librosa warnings if any
warnings.filterwarnings("ignore")

class SentenceProcessor:
    def __init__(self, target_sr=22050, vad_db_threshold=40):
        """
        Initializes the sentence processor.
        
        Args:
            target_sr (int): Sampling rate to standardize all audio inputs.
            vad_db_threshold (int): Decibel threshold for Voice Activity Detection (silence removal).
        """
        self.target_sr = target_sr
        self.vad_db_threshold = vad_db_threshold

    def _advanced_vad_segmentation(self, y, sr):
        """
        Gelişmiş Kelime/Hece Bölütleme (Advanced VAD).
        Sürekli ve hızlı konuşmalarda standart sessizlik kesicileri kelimeleri ayıramaz.
        Bu algoritma uzun nefessiz blokları algılayıp, kelimelerin/hecelerin 
        enerji patlamalarını (onset) bularak kelime sınırlarından matematiksel olarak keser.
        """
        # 1. Standart Sessizlik Temizliği (top_db=25 ile yutulmuş nefesleri daha iyi yakala)
        intervals = librosa.effects.split(y, top_db=25, frame_length=2048, hop_length=512)
        
        final_intervals = []
        max_duration = 1.0  # 1 Saniyeden uzun, duraksaması olmayan bloklar muhtemelen birden çok kelimedir
        
        for start, end in intervals:
            duration = (end - start) / sr
            
            if duration > max_duration:
                # 2. Hızlı Konuşmayı Parçalama (Onset / Vuruş ve Hece Analizi)
                segment = y[start:end]
                
                # Sesin enerjisindeki ani patlamaları (hece/sessiz-sesli harf vuruşlarını) bulur
                onsets = librosa.onset.onset_detect(
                    y=segment, 
                    sr=sr, 
                    units='samples',
                    hop_length=512,
                    backtrack=True, # Vuruşun tam olarak başladığı o sessiz milisaniyeye (kelime başına) geri sarar
                    pre_max=20,     # Ön pencereleme
                    post_max=20,    # Arka pencereleme
                    pre_avg=100,    # Ortalama ses filtresi (pürüzleri yumuşatmak için)
                    delta=0.15,     # Duyarlılık: Sadece belirgin farklarda hece kes (çok absürt de bölmesin)
                    wait=int(sr * 0.3) # Kelime içi saçmalama yapmasın diye min 0.3sn nefes/durma bekleme şartı
                )
                
                valid_onsets = [o for o in onsets if o > sr * 0.3 and (len(segment) - o) > sr * 0.3]
                
                if len(valid_onsets) > 0:
                    current_pos = 0
                    for onset in valid_onsets:
                        final_intervals.append([start + current_pos, start + onset])
                        current_pos = onset
                    final_intervals.append([start + current_pos, end])
                else:
                    final_intervals.append([start, end])
            else:
                final_intervals.append([start, end])
                
        # 3. Nefes, Tıkırtı veya Yarım Heceleri (Çok Kısa Parçaları) Önle ve Onar
        merged_intervals = []
        for s, e in final_intervals:
            if not merged_intervals:
                merged_intervals.append([s, e])
                continue
                
            last_s, last_e = merged_intervals[-1]
            dur = (e - s) / sr
            last_dur = (last_e - last_s) / sr
            
            # Eğer bir kelime 0.25sn'den daha kısa kaldıysa (Yutulmuş kelime/yanlış hece),
            # kopartmak yerine yanındaki makul komşusuna ekleyip bütünlüğü koru.
            if dur < 0.25:
                merged_intervals[-1][1] = e
            elif last_dur < 0.25:
                merged_intervals[-1][1] = e
            else:
                merged_intervals.append([s, e])
                
        return merged_intervals

    def process_audio(self, file_path, output_dir="temp_segments"):
        """
        Full pipeline: Load -> Resample -> Advanced VAD -> Segment -> Save Segments.
        Returns a list of segment file paths.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        try:
            # 1. Load & Resample
            y, sr = librosa.load(file_path, sr=self.target_sr)
            
            # 2. Akıllı VAD & Kelime Parçalama (Yeni Nesil Motor)
            intervals = self._advanced_vad_segmentation(y, sr)
            
            segment_paths = []
            
            if len(intervals) == 0:
                print("⚠️ Sürekli akış çok belirsiz, tek blok analiz ediliyor.")
                intervals = [[0, len(y)]]

            for i, (start, end) in enumerate(intervals):
                segment = y[start:end]
                
                # Çok cılız tıkırtıları (örn: 0.2sn altı) anlamsız dosyalar olmaması adına atla
                if len(segment) / sr < 0.2:
                    continue
                    
                seg_name = f"seg_{uuid.uuid4().hex[:8]}_{i}.wav"
                seg_path = os.path.join(output_dir, seg_name)
                
                sf.write(seg_path, segment, sr)
                segment_paths.append(seg_path)
            
            if not segment_paths:
                 return [file_path]

            return segment_paths

        except Exception as e:
            print(f"❌ Error in sentence processing: {e}")
            return [file_path]

    def extract_segments_info(self, file_path):
        """
        Frontend'in görsel arayüzde (UI) kelimeleri hap/kutu şeklinde
        milisaniyesine kadar isabetli gösterebilmesi için zamanları çıkartır.
        """
        try:
            y, sr = librosa.load(file_path, sr=self.target_sr)
            
            # Aynı akıllı motoru burada da UI çizimi için çağırıyoruz
            intervals = self._advanced_vad_segmentation(y, sr)
            segments = []
            
            if len(intervals) == 0:
                return [{"start": 0.0, "end": float(len(y) / sr)}]

            for start, end in intervals:
                start_time = float(start) / sr
                end_time = float(end) / sr
                
                if end_time - start_time < 0.2:
                    continue
                    
                segments.append({
                    "start": round(start_time, 3),
                    "end": round(end_time, 3)
                })
            
            if not segments:
                 return [{"start": 0.0, "end": float(len(y) / sr)}]

            return segments

        except Exception as e:
            print(f"❌ Error in frontend processing (extracting info): {e}")
            return []

    def weighted_voting(self, results):
        """
        Gelişmiş Cümle Analizi ve Duygu Oylaması (Weighted Voting + Realistic Confidence).
        Eski mantıktaki 'toplam oy oranı üzerinden 100% güvenilirlik' hatasını çözer.
        Oy oranını ve modelin gerçek eminlik düzeyini (confidence) harmanlar.
        """
        if not results:
            return None
            
        # 1. Aşama İyileştirmesi: Statik Duygu Çarpanlarının Sıfırlanması (Pure Soft-Voting)
        # Eski sistemdeki sabit çarpanlar (Angry 1.6, Calm 0.7 vb.) Calm ve Sad gibi duyguların
        # Recall (Duyarlılık) değerlerini düşürüp Cümle F1 skorunu bozuyordu.
        # Artık her duygunun modelden gelen saf olasılığı ağırlık olarak kabul ediliyor.
        EMOTION_MULTIPLIERS = {
            'angry': 1.0,
            'happy': 1.0,
            'sad': 1.0,
            'calm': 1.0
        }

        # Accumulators
        weighted_scores = {e: 0.0 for e in EMOTION_MULTIPLIERS.keys()}
        emotion_counts = {e: 0 for e in EMOTION_MULTIPLIERS.keys()}
        raw_confidences = {e: [] for e in EMOTION_MULTIPLIERS.keys()}
        
        total_segments = len(results)

        for res in results:
            winning_emotion = res['emotion'].lower()
            
            # 3. Aşama İyileştirmesi: Gürültü/Nefes Susturma (Confidence Thresholding)
            # Eğer makinenin bu kelime için hiçbir duyguya olan güvenilirlik skoru %40'ı geçemiyorsa
            # o kelime ya çok bulanıktır ya da tamamen gürültü/nefes sesidir. Oylamayı zehirlemesini engelliyoruz.
            if res.get('confidence', 0.0) < 40.0:
                continue
                
            if winning_emotion in EMOTION_MULTIPLIERS:
                emotion_counts[winning_emotion] += 1
                raw_confidences[winning_emotion].append(res['confidence'])
                
            # Adım 0: Gerçek Olasılık Dağılımının İçine Gir (Full Probability Accumulation)
            # Modelin bu kelime için ürettiği "tüm" % ihtimallerin ortalamasını yansıt.
            segment_scores = res.get('all_scores', {winning_emotion: res['confidence']})
            
            # 2. Aşama İyileştirmesi: Segment Süresine Göre Oylama (Duration-Based Weighting)
            # 0.2 saniyelik bir nefes veya yutulmuş kelime ile 1.5 saniyelik net bir kelimenin oyu aynı olamaz.
            # Tahmin olasılığını, o kelimenin saniye cinsinden süresiyle çarparak adaleti sağlıyoruz.
            segment_duration = res.get('duration', 1.0)
            
            for em_key, prob_val in segment_scores.items():
                if em_key in EMOTION_MULTIPLIERS:
                    multiplier = EMOTION_MULTIPLIERS[em_key]
                    # Olasılığı (0-100) -> (0-1) yapıp ağırlık ve kelime SÜRESİ (duration) ile çarparak pastaya ekle
                    weighted_scores[em_key] += (prob_val / 100.0) * multiplier * segment_duration

        # 1. Hangi Duygu Kazandı? (En yüksek ağırlıklı puana sahip olan)
        final_emotion = max(weighted_scores, key=weighted_scores.get)
        
        # 2. Duygu Dağılımını Yüzdeye Çevir (Frontend Bar'ları İçin)
        # Sadece kazananı değil, cümlenin içerisindeki tüm duyguların yüzdelik payını çıkarır
        total_weight = sum(weighted_scores.values())
        emotion_distribution = {}
        if total_weight > 0:
            for k, v in weighted_scores.items():
                emotion_distribution[k] = round(float((v / total_weight) * 100.0), 2)
        else:
            emotion_distribution = {k: 0.0 for k in weighted_scores.keys()}
            
        # Ana "Confidence" skoru doğrudan kazanan duygunun pastadaki payı olsun
        # Böylece ekrandaki bar ile yukarıdaki ana yüzde birebir uyuşur.
        final_conf = emotion_distribution.get(final_emotion, 0.0)
        
        return {
            'final_emotion': final_emotion,
            'confidence': final_conf,
            'weighted_details': emotion_distribution
        }
