import sys
import numpy as np
import librosa
from pathlib import Path

# Add Backend to path
backend_dir = Path(__file__).parent.absolute()
sys.path.append(str(backend_dir))

from sentence_analysis.feature_extractor import FeatureExtractor
from sentence_analysis.audio_segmenter import AudioSegmenter
from sentence_analysis.word_predictor import WordLevelPredictor
from sentence_analysis.aggregation_engine import AggregationEngine

def create_simulated_sentence(files, sr=16000, pause_duration=0.2):
    """Concatenates multiple audio files into one 'sentence'"""
    combined = []
    pause = np.zeros(int(pause_duration * sr))
    
    for f in files:
        y, _ = librosa.load(str(f), sr=sr)
        combined.append(y)
        combined.append(pause) # Add pause between words
        
    return np.concatenate(combined)

def debug_sentence_repro():
    print("--- DEBUGGING SENTENCE misclassification ---")
    
    # 1. Setup
    project_root = backend_dir.parent
    model_path = project_root / "Demo" / "models" / "best_emotion_model.pkl"
    sound_source = backend_dir / "Sound_Source"
    
    segmenter = AudioSegmenter() # Uses default params from class (0.5s pad)
    extractor = FeatureExtractor()
    predictor = WordLevelPredictor(str(model_path))
    aggregator = AggregationEngine()
    
    # 4. Test Aggregator Bias Directly
    from sentence_analysis.word_predictor import PredictionResult
    
    print("\n--- Testing Aggregator Bias ---")
    
    # Case: 1 vs 2 (High Conf Happy vs Lower Conf Sad)
    # Happy is the "Minority" we fixed. Did we over-fix it?
    
    mock_preds = [
        # Segment 1: High Confidence Happy (e.g. a burst of noise)
        PredictionResult(
            emotion="happy",
            confidence=0.99,
            probabilities={"happy": 0.99, "sad": 0.0, "calm": 0.0, "angry": 0.01},
            is_uncertain=False
        ),
        # Segment 2: Moderate Sad
        PredictionResult(
            emotion="sad",
            confidence=0.60,
            probabilities={"happy": 0.1, "sad": 0.60, "calm": 0.2, "angry": 0.1},
            is_uncertain=False
        ),
        # Segment 3: Moderate Sad
        PredictionResult(
            emotion="sad",
            confidence=0.60,
            probabilities={"happy": 0.1, "sad": 0.60, "calm": 0.2, "angry": 0.1},
            is_uncertain=False
        )
    ]
    
    res = aggregator.aggregate(mock_preds, strategy="weighted_average")
    print(f"Scenario 1 (1 High Happy vs 2 Mod Sad) -> {res.primary_emotion.upper()} (Conf: {res.confidence:.2f})")
    print(f"  Probs: {res.probabilities}")
    
    # Case: Normal Sentence (All same)
    mock_preds_2 = [
         PredictionResult(
            emotion="calm",
            confidence=0.55,
            probabilities={"happy": 0.2, "sad": 0.1, "calm": 0.55, "angry": 0.15},
            is_uncertain=False
        ) for _ in range(5)
    ]
    # Test Majority Voting
    res_mv = aggregator.aggregate(mock_preds, strategy="majority_voting")
    print(f"Scenario 1 (Majority Voting) -> {res_mv.primary_emotion.upper()} (Conf: {res_mv.confidence:.2f})")
    
    return # End test here

if __name__ == "__main__":
    debug_sentence_repro()
