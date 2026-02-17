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

    def process_audio(self, file_path, output_dir="temp_segments"):
        """
        Full pipeline: Load -> Resample -> VAD -> Segment -> Save Segments.
        Returns a list of segment file paths.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        try:
            # 1. Load & Resample
            y, sr = librosa.load(file_path, sr=self.target_sr)
            
            # 2. VAD & Segmentation (Silence-Based)
            # top_db=40 matches the user requirement "below 40dB"
            intervals = librosa.effects.split(y, top_db=self.vad_db_threshold)
            
            segment_paths = []
            
            # If no intervals found (too quiet), process the whole file
            if len(intervals) == 0:
                print("⚠️ No speech detected above threshold, using full audio.")
                intervals = [[0, len(y)]]

            for i, (start, end) in enumerate(intervals):
                # Extract segment
                segment = y[start:end]
                
                # Filter extremely short segments (e.g. < 0.2s) that might be clicks/noise
                if len(segment) / sr < 0.2:
                    continue
                    
                # Generate unique filename
                seg_name = f"seg_{uuid.uuid4().hex[:8]}_{i}.wav"
                seg_path = os.path.join(output_dir, seg_name)
                
                # Save
                sf.write(seg_path, segment, sr)
                segment_paths.append(seg_path)
            
            if not segment_paths:
                # Fallback if all segments were filtered out
                 return [file_path]

            return segment_paths

        except Exception as e:
            print(f"❌ Error in sentence processing: {e}")
            return [file_path]

    def weighted_voting(self, results):
        """
        Implements the 'Emotion-Weighted Voting Algorithm'.
        
        Logic:
        - High Arousal (Anger, Happy) get higher weights.
        - Weight is multiplied by confidence/probability.
        - Neutral/Calm get standard or lower weights depending on context.
        
        Args:
            results (list): List of dicts with 'emotion' and 'confidence' (0-100) and 'probabilities'.
        
        Returns:
            dict: The final aggregated result.
        """
        if not results:
            return None
            
        # Weights based on "High Arousal Priority"
        # The prompt says: "reduces the neutral words... increasing the dominant angry word"
        # We assign multipliers.
        EMOTION_MULTIPLIERS = {
            'angry': 1.5,
            'happy': 1.2,
            'sad': 1.0,
            'calm': 0.8,
            'neutral': 0.8 
        }

        # Initialize score accumulators
        weighted_scores = {e: 0.0 for e in EMOTION_MULTIPLIERS.keys()}
        
        for res in results:
            emotion = res['emotion'].lower()
            confidence = res['confidence'] / 100.0 # Normalize to 0-1
            
            # If we have full probability distribution, use it? 
            # The prompt implies looking at the specific predicted emotion's confidence.
            # "coefficient for the emotion with a higher confidence level is increased"
            
            multiplier = EMOTION_MULTIPLIERS.get(emotion, 1.0)
            
            # Score = Confidence * Multiplier
            score = confidence * multiplier
            
            if emotion in weighted_scores:
                weighted_scores[emotion] += score
            else:
                weighted_scores[emotion] = score

        # Find the emotion with the highest total weighted score
        final_emotion = max(weighted_scores, key=weighted_scores.get)
        
        # Calculate a pseudo-confidence for the final result
        total_score = sum(weighted_scores.values())
        final_conf = (weighted_scores[final_emotion] / total_score * 100) if total_score > 0 else 0.0
        
        return {
            'final_emotion': final_emotion,
            'confidence': final_conf,
            'weighted_details': weighted_scores
        }
