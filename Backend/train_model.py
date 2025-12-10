import os
import sys
import numpy as np
import pickle
import librosa
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

# Add Backend directory to path to import sentence_analysis module
backend_dir = Path(__file__).parent.absolute()
sys.path.append(str(backend_dir))

from sentence_analysis.feature_extractor import FeatureExtractor

def train_model():
    print("Initializing training...")
    
    # Paths
    sound_source_path = backend_dir / "Sound_Source"
    models_dir = backend_dir.parent / "Demo" / "models"
    model_save_path = models_dir / "best_emotion_model.pkl"
    
    if not sound_source_path.exists():
        print(f"Error: Sound_Source not found at {sound_source_path}")
        return

    # Initialize components
    extractor = FeatureExtractor()
    # Use the SAME segmenter settings as defined in audio_segmenter.py default
    # Ensuring training data matches inference pipeline exactly
    from sentence_analysis.audio_segmenter import AudioSegmenter
    segmenter = AudioSegmenter(
        silence_threshold=0.01,
        min_silence_duration=0.3,
        segment_padding=0.3
    )

    X = []
    y = []
    
    # Emotion mapping (Folder Name -> Label)
    emotions = {
        'Angry': 'angry',
        'Calm': 'calm',
        'Happy': 'happy',
        'Sad': 'sad'
    }
    
    print("Extracting features from audio files (with Segmentation)...")
    total_files = 0
    total_segments = 0
    
    for folder_name, label in emotions.items():
        folder_path = sound_source_path / folder_name
        if not folder_path.exists():
            print(f"Warning: Folder {folder_name} not found, skipping.")
            continue
            
        files = list(folder_path.glob("*.wav"))
        print(f"Processing {len(files)} files for emotion: {label}")
        
        for file_path in files:
            try:
                # Load audio
                audio, sr = librosa.load(str(file_path), sr=16000, duration=3.0)
                
                # SEGMENT FIRST (Crucial for matching inference pipeline)
                segments = segmenter.segment_audio(audio, sr)
                
                # If segmenter finds nothing (e.g. file is all silence?), use whole file fallback
                # or skip? For robustness, let's use the segments if found, else whole.
                
                audio_chunks = []
                if len(segments) > 0:
                    for seg in segments:
                        audio_chunks.append(seg.audio_data)
                else:
                    # Fallback: Use trimmed whole audio if segmentation fails
                    # This prevents discarding valid data that VAD might miss
                    trimmed, _ = librosa.effects.trim(audio, top_db=20)
                    if len(trimmed) > 0:
                        audio_chunks.append(trimmed)
                    else:
                        audio_chunks.append(audio) # Last resort
                
                for chunk in audio_chunks:
                     # Extract features (matches the new 378-feature extractor)
                    features = extractor.extract_features(chunk, sr, apply_scaling=False)
                    
                    # Handle NaN immediately (just in case)
                    if np.isnan(features).any():
                         features = np.nan_to_num(features)

                    X.append(features)
                    y.append(label)
                    total_segments += 1
                
                total_files += 1
                
                if total_files % 100 == 0:
                    print(f"Processed {total_files} files ({total_segments} segments)...")
                    
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")

    X = np.array(X)
    y = np.array(y)
    
    print(f"\nFeature extraction complete. Shape: {X.shape}")
    
    # Imputation
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Encoding labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
    
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report, confusion_matrix
    
    # Train - Using Random Forest (Balanced) for robust, fast performance
    print("Training Random Forest Classifier...")
    
    model = RandomForestClassifier(
        n_estimators=500,  # High number of trees for stability
        max_depth=None,    # Allow full depth to capture word nuances
        class_weight='balanced', 
        random_state=42, 
        n_jobs=-1          # Fast parallel training
    )
    
    model.fit(X_train, y_train)
    
    # Detailed Evaluation
    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Save artifacts
    artifacts = {
        'model': model,
        'scaler': scaler,
        'label_encoder': le,
        'imputer': imputer
    }
    
    models_dir.mkdir(parents=True, exist_ok=True)
    with open(model_save_path, 'wb') as f:
        pickle.dump(artifacts, f)
        
    print(f"\nModel saved to: {model_save_path}")
    print("You can now restart the backend server.")

if __name__ == "__main__":
    train_model()
