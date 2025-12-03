"""
Feature Extraction Module

Extracts audio features consistent with word-level model training methodology.
"""

from typing import Dict, Optional
import numpy as np
import librosa
import pickle
from pathlib import Path


class FeatureExtractor:
    """Extracts audio features for emotion recognition"""
    
    def __init__(self, n_mfcc: int = 40, scaler=None, model_path: Optional[str] = None):
        """
        Initialize FeatureExtractor
        
        Args:
            n_mfcc: Number of MFCC coefficients to extract (default: 40)
            scaler: Pre-loaded StandardScaler object (optional)
            model_path: Path to model file containing scaler (optional)
        """
        self.n_mfcc = n_mfcc
        self.scaler = scaler
        
        # Load scaler from model file if path provided
        if model_path is not None and scaler is None:
            self._load_scaler_from_model(model_path)
    
    def _load_scaler_from_model(self, model_path: str):
        """
        Load scaler from trained model file
        
        Args:
            model_path: Path to the model pickle file
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        if isinstance(model_data, dict) and 'scaler' in model_data:
            self.scaler = model_data['scaler']
        else:
            raise ValueError(f"Model file does not contain a 'scaler' key")
    
    def set_scaler(self, scaler):
        """
        Set the scaler to use for feature transformation
        
        Args:
            scaler: StandardScaler object from scikit-learn
        """
        self.scaler = scaler
    
    def extract_features(self, audio: np.ndarray, sr: int, apply_scaling: bool = True) -> np.ndarray:
        """
        Extract complete feature vector from audio segment
        
        Feature composition (378 features total):
        - MFCC: 40 coefficients × 2 (mean, std) = 80 features
        - Chroma: 12 coefficients × 2 = 24 features
        - Mel Spectrogram: 128 bands × 2 = 256 features
        - Spectral Contrast: 7 bands × 2 = 14 features
        - Zero Crossing Rate: 2 features (mean, std)
        - RMS Energy: 2 features (mean, std)
        
        Args:
            audio: Audio signal as numpy array
            sr: Sample rate
            apply_scaling: Whether to apply StandardScaler transformation (default: True)
            
        Returns:
            Feature vector as numpy array (378 features)
        """
        features = []
        
        # 1. MFCC features (80 features)
        mfcc_features = self.extract_mfcc(audio, sr, self.n_mfcc)
        features.extend(mfcc_features)
        
        # 2. Chroma features (24 features)
        chroma_features = self.extract_chroma(audio, sr)
        features.extend(chroma_features)
        
        # 3. Mel Spectrogram features (256 features)
        mel_features = self.extract_mel_spectrogram(audio, sr)
        features.extend(mel_features)
        
        # 4. Spectral Contrast features (14 features)
        contrast_features = self.extract_spectral_contrast(audio, sr)
        features.extend(contrast_features)
        
        # 5. Temporal features (4 features: ZCR and RMS)
        temporal_features = self.extract_temporal_features(audio)
        features.extend([
            temporal_features['zcr_mean'],
            temporal_features['zcr_std'],
            temporal_features['rms_mean'],
            temporal_features['rms_std']
        ])
        
        feature_vector = np.array(features)
        
        # Apply scaler transformation if available and requested
        if apply_scaling and self.scaler is not None:
            # Reshape to 2D array for scaler (1 sample, n features)
            feature_vector = feature_vector.reshape(1, -1)
            feature_vector = self.scaler.transform(feature_vector)
            # Flatten back to 1D
            feature_vector = feature_vector.flatten()
        
        return feature_vector
    
    def extract_mfcc(self, audio: np.ndarray, sr: int, n_mfcc: int = 40) -> np.ndarray:
        """
        Extract MFCC features
        
        Args:
            audio: Audio signal
            sr: Sample rate
            n_mfcc: Number of MFCC coefficients
            
        Returns:
            Array of MFCC features (mean and std for each coefficient)
        """
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        
        features = []
        features.extend(np.mean(mfccs, axis=1))
        features.extend(np.std(mfccs, axis=1))
        
        return np.array(features)
    
    def extract_chroma(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract chroma features (pitch and tone)
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Array of chroma features (mean and std for 12 pitch classes)
        """
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        
        features = []
        features.extend(np.mean(chroma, axis=1))
        features.extend(np.std(chroma, axis=1))
        
        return np.array(features)
    
    def extract_mel_spectrogram(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract mel spectrogram features
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Array of mel spectrogram features (mean and std for 128 mel bands)
        """
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        
        features = []
        features.extend(np.mean(mel, axis=1))
        features.extend(np.std(mel, axis=1))
        
        return np.array(features)
    
    def extract_spectral_contrast(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract spectral contrast features
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Array of spectral contrast features (mean and std for 7 bands)
        """
        # Adjust parameters based on sample rate to avoid exceeding Nyquist frequency
        # Default n_bands=6 gives 7 bands total (6+1)
        # For low sample rates, reduce fmin to ensure bands don't exceed Nyquist
        nyquist = sr / 2
        fmin = min(200.0, nyquist / 64)  # Ensure fmin is reasonable for the sample rate
        
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, fmin=fmin, n_bands=6)
        
        features = []
        features.extend(np.mean(contrast, axis=1))
        features.extend(np.std(contrast, axis=1))
        
        return np.array(features)
    
    def extract_temporal_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract temporal features (ZCR and RMS energy)
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary with ZCR and RMS statistics
        """
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        
        # RMS Energy
        rms = librosa.feature.rms(y=audio)
        
        return {
            'zcr_mean': float(np.mean(zcr)),
            'zcr_std': float(np.std(zcr)),
            'rms_mean': float(np.mean(rms)),
            'rms_std': float(np.std(rms))
        }
