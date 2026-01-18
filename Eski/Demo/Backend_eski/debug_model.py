import sys
import pickle
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

def debug_prediction():
    print("--- DEBUGGING MODEL (FULL PIPELINE) ---")
    
    # 1. Load Model
    project_root = backend_dir.parent
    model_path = project_root / "Demo" / "models" / "best_emotion_model.pkl"
    
    print(f"Checking model at: {model_path}")
    if not model_path.exists():
        print("FAIL: Model file not found!")
        return

    # Initialize components
    try:
        segmenter = AudioSegmenter()
        extractor = FeatureExtractor()
        predictor = WordLevelPredictor(str(model_path))
        aggregator = AggregationEngine()
        print("SUCCESS: Components initialized.")
        
        # Verify Label Encoder
        if predictor.model_artifacts and 'label_encoder' in predictor.model_artifacts:
             print(f"Label Encoder Classes: {predictor.model_artifacts['label_encoder'].classes_}")
    except Exception as e:
        print(f"FAIL: Initialization error. {e}")
        return

    # 3. Test ALL Categories
    sound_source = backend_dir / "Sound_Source"
    categories = ['Angry', 'Calm', 'Happy', 'Sad']
    
    total_correct = 0
    total_tested = 0
    
    for category in categories:
        cat_dir = sound_source / category
        if not cat_dir.exists():
            print(f"SKIP: {category} directory not found")
            continue

        test_files = list(cat_dir.glob("*.wav"))
        if not test_files:
            continue
            
        print(f"\n=== Testing Category: {category} ===")
        # Test 5 files
        for i, test_file in enumerate(test_files[:5]):
            try:
                # 1. Load (simulate upload)
                audio, sr = librosa.load(str(test_file), sr=16000)
                
                # 2. Segment (simulate API pipeline)
                # Note: Training data is already "one word" usually, so segmenter might find 1 segment
                # OR it might trim silence differently. This is crucial to check.
                segments = segmenter.segment_audio(audio, sr)
                
                word_predictions = []
                for segment in segments:
                    features = extractor.extract_features(segment.audio_data, segment.sample_rate, apply_scaling=False)
                    prediction = predictor.predict(features)
                    word_predictions.append(prediction)
                
                if not word_predictions:
                    print(f"  {test_file.name}: NO SEGMENTS FOUND")
                    continue
                    
                # 4. Aggregate
                final_result = aggregator.aggregate(word_predictions, strategy="weighted_average")
                pred = final_result.primary_emotion.lower()
                actual = category.lower()
                
                is_correct = (pred == actual)
                if is_correct:
                    total_correct += 1
                total_tested += 1
                
                marker = "✓" if is_correct else f"✗ (Got {pred.upper()})"
                print(f"  {test_file.name}: {marker} | Conf: {final_result.confidence:.2f} | Segments: {len(segments)}")
                
            except Exception as e:
                print(f"Error testing {test_file.name}: {e}")

    if total_tested > 0:
        print(f"\nOVERALL ACCURACY: {total_correct}/{total_tested} ({total_correct/total_tested*100:.1f}%)")

if __name__ == "__main__":
    debug_prediction()
