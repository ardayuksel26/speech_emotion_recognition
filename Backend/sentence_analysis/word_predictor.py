"""
Word-Level Prediction Module

Applies trained Gradient Boosting model to word segments.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pickle
import os


@dataclass
class PredictionResult:
    """Word-level emotion prediction result"""
    emotion: str
    confidence: float
    probabilities: Dict[str, float]
    is_uncertain: bool  # True if confidence < 0.4


class WordLevelPredictor:
    """Applies trained model to predict word-level emotions"""
    
    def __init__(self, model_path: str, uncertainty_threshold: float = 0.4):
        """
        Initialize WordLevelPredictor
        
        Args:
            model_path: Path to trained model pickle file
            uncertainty_threshold: Confidence threshold for uncertainty flagging
        """
        self.model_path = model_path
        self.uncertainty_threshold = uncertainty_threshold
        self.model_artifacts = None
        self.load_model()
    
    def load_model(self) -> Dict:
        """
        Load trained model artifacts from pickle file
        
        Returns:
            Dictionary containing model, scaler, label_encoder, and imputer
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        with open(self.model_path, 'rb') as f:
            self.model_artifacts = pickle.load(f)
        
        # Validate required components
        required_keys = ['model', 'scaler', 'label_encoder']
        for key in required_keys:
            if key not in self.model_artifacts:
                raise ValueError(f"Model artifacts missing required key: {key}")
        
        return self.model_artifacts
    
    def predict(self, features: np.ndarray) -> PredictionResult:
        """
        Predict emotion for a single feature vector
        
        Args:
            features: Feature vector (378 features)
            
        Returns:
            PredictionResult object
        """
        if self.model_artifacts is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Reshape if needed
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Handle NaN values if imputer is available
            if 'imputer' in self.model_artifacts:
                features = self.model_artifacts['imputer'].transform(features)
            
            # Apply scaling
            features_scaled = self.model_artifacts['scaler'].transform(features)
            
            # Get predictions
            probabilities = self.model_artifacts['model'].predict_proba(features_scaled)[0]
            pred_idx = np.argmax(probabilities)
            confidence = float(probabilities[pred_idx])
            
            # Get emotion label
            emotion = self.model_artifacts['label_encoder'].classes_[pred_idx]
            
            # Create probability dictionary
            prob_dict = {
                str(cls): float(prob)
                for cls, prob in zip(
                    self.model_artifacts['label_encoder'].classes_,
                    probabilities
                )
            }
            
            # Check uncertainty
            is_uncertain = confidence < self.uncertainty_threshold
            
            return PredictionResult(
                emotion=emotion,
                confidence=confidence,
                probabilities=prob_dict,
                is_uncertain=is_uncertain
            )
            
        except Exception as e:
            # LOG THE ERROR
            print("!!! PREDICTION ERROR !!!")
            print(f"Error details: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Fallback: return uniform distribution
            return self._get_fallback_prediction(str(e))
    
    def predict_batch(self, features_list: List[np.ndarray]) -> List[PredictionResult]:
        """
        Predict emotions for multiple feature vectors
        
        Args:
            features_list: List of feature vectors
            
        Returns:
            List of PredictionResult objects
        """
        results = []
        for features in features_list:
            result = self.predict(features)
            results.append(result)
        return results
    
    def _get_fallback_prediction(self, error_msg: str = "") -> PredictionResult:
        """
        Return uniform probability distribution as fallback
        
        Args:
            error_msg: Error message for logging
            
        Returns:
            PredictionResult with uniform probabilities
        """
        if self.model_artifacts and 'label_encoder' in self.model_artifacts:
            classes = self.model_artifacts['label_encoder'].classes_
        else:
            classes = ['angry', 'calm', 'happy', 'sad']
        
        uniform_prob = 1.0 / len(classes)
        prob_dict = {cls: uniform_prob for cls in classes}
        
        return PredictionResult(
            emotion=classes[0],  # Default to first class
            confidence=uniform_prob,
            probabilities=prob_dict,
            is_uncertain=True
        )
