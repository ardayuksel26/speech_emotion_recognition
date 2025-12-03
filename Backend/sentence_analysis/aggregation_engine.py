"""
Aggregation Engine Module

Combines word-level predictions into sentence-level predictions using multiple strategies.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from collections import Counter

from .word_predictor import PredictionResult


@dataclass
class SentencePrediction:
    """Sentence-level emotion prediction result"""
    primary_emotion: str
    confidence: float
    probabilities: Dict[str, float]
    is_mixed: bool
    secondary_emotion: Optional[str]
    strategy_used: str
    metadata: Dict


class AggregationEngine:
    """Combines word-level predictions into sentence-level predictions"""
    
    def __init__(self):
        """Initialize AggregationEngine"""
        self.strategies = {
            'weighted_average': self.weighted_average,
            'majority_voting': self.majority_voting,
            'temporal_weighted': self.temporal_weighted,
            'confidence_threshold': self.confidence_threshold,
        }
    
    def aggregate(
        self,
        predictions: List[PredictionResult],
        strategy: str = 'weighted_average'
    ) -> SentencePrediction:
        """
        Aggregate word-level predictions into sentence-level prediction
        
        Args:
            predictions: List of word-level PredictionResult objects
            strategy: Aggregation strategy to use
            
        Returns:
            SentencePrediction object
        """
        if not predictions:
            raise ValueError("Cannot aggregate empty predictions list")
        
        if strategy not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {list(self.strategies.keys())}")
        
        # Apply the selected strategy
        aggregation_func = self.strategies[strategy]
        return aggregation_func(predictions)
    
    def weighted_average(self, predictions: List[PredictionResult]) -> SentencePrediction:
        """
        Weighted average strategy using confidence scores as weights
        
        Formula: P(emotion) = Σ(confidence_i × prob_i) / Σ(confidence_i)
        
        Args:
            predictions: List of word-level predictions
            
        Returns:
            SentencePrediction object
        """
        # Get all emotion classes
        emotion_classes = list(predictions[0].probabilities.keys())
        
        # Calculate weighted probabilities
        weighted_probs = {emotion: 0.0 for emotion in emotion_classes}
        total_weight = 0.0
        
        for pred in predictions:
            weight = pred.confidence
            total_weight += weight
            
            for emotion, prob in pred.probabilities.items():
                weighted_probs[emotion] += weight * prob
        
        # Normalize by total weight
        if total_weight > 0:
            for emotion in weighted_probs:
                weighted_probs[emotion] /= total_weight
        
        # Find primary emotion
        primary_emotion = max(weighted_probs, key=weighted_probs.get)
        confidence = weighted_probs[primary_emotion]
        
        # Detect mixed emotions
        is_mixed, secondary_emotion = self.detect_mixed_emotions(weighted_probs)
        
        # Create metadata
        metadata = {
            'num_words': len(predictions),
            'avg_confidence': sum(p.confidence for p in predictions) / len(predictions),
            'uncertain_words': sum(1 for p in predictions if p.is_uncertain),
        }
        
        return SentencePrediction(
            primary_emotion=primary_emotion,
            confidence=confidence,
            probabilities=weighted_probs,
            is_mixed=is_mixed,
            secondary_emotion=secondary_emotion,
            strategy_used='weighted_average',
            metadata=metadata
        )
    
    def majority_voting(self, predictions: List[PredictionResult]) -> SentencePrediction:
        """
        Majority voting strategy where most frequent emotion is selected
        
        Tie-breaking: Use highest average confidence
        
        Args:
            predictions: List of word-level predictions
            
        Returns:
            SentencePrediction object
        """
        # Count emotion occurrences
        emotion_counts = Counter(pred.emotion for pred in predictions)
        
        # Find the maximum count
        max_count = max(emotion_counts.values())
        
        # Get all emotions with max count (for tie-breaking)
        top_emotions = [emotion for emotion, count in emotion_counts.items() if count == max_count]
        
        # Tie-breaking: select emotion with highest average confidence
        if len(top_emotions) > 1:
            avg_confidences = {}
            for emotion in top_emotions:
                confidences = [p.confidence for p in predictions if p.emotion == emotion]
                avg_confidences[emotion] = sum(confidences) / len(confidences)
            primary_emotion = max(avg_confidences, key=avg_confidences.get)
        else:
            primary_emotion = top_emotions[0]
        
        # Calculate confidence as proportion of votes
        confidence = emotion_counts[primary_emotion] / len(predictions)
        
        # Calculate probability distribution based on vote counts
        probabilities = {
            emotion: count / len(predictions)
            for emotion, count in emotion_counts.items()
        }
        
        # Ensure all emotion classes are present
        emotion_classes = list(predictions[0].probabilities.keys())
        for emotion in emotion_classes:
            if emotion not in probabilities:
                probabilities[emotion] = 0.0
        
        # Detect mixed emotions
        is_mixed, secondary_emotion = self.detect_mixed_emotions(probabilities)
        
        # Create metadata
        metadata = {
            'num_words': len(predictions),
            'vote_counts': dict(emotion_counts),
            'uncertain_words': sum(1 for p in predictions if p.is_uncertain),
        }
        
        return SentencePrediction(
            primary_emotion=primary_emotion,
            confidence=confidence,
            probabilities=probabilities,
            is_mixed=is_mixed,
            secondary_emotion=secondary_emotion,
            strategy_used='majority_voting',
            metadata=metadata
        )
    
    def temporal_weighted(self, predictions: List[PredictionResult]) -> SentencePrediction:
        """
        Temporal-weighted strategy where later words receive higher weights
        
        Formula: weight_i = (i + 1) / n (linear decay from start to end)
        
        Args:
            predictions: List of word-level predictions
            
        Returns:
            SentencePrediction object
        """
        n = len(predictions)
        emotion_classes = list(predictions[0].probabilities.keys())
        
        # Calculate temporal-weighted probabilities
        weighted_probs = {emotion: 0.0 for emotion in emotion_classes}
        total_weight = 0.0
        
        for i, pred in enumerate(predictions):
            # Linear weight: later words get higher weight
            weight = (i + 1) / n
            total_weight += weight
            
            for emotion, prob in pred.probabilities.items():
                weighted_probs[emotion] += weight * prob
        
        # Normalize by total weight
        if total_weight > 0:
            for emotion in weighted_probs:
                weighted_probs[emotion] /= total_weight
        
        # Find primary emotion
        primary_emotion = max(weighted_probs, key=weighted_probs.get)
        confidence = weighted_probs[primary_emotion]
        
        # Detect mixed emotions
        is_mixed, secondary_emotion = self.detect_mixed_emotions(weighted_probs)
        
        # Create metadata
        metadata = {
            'num_words': len(predictions),
            'recency_bias': True,
            'uncertain_words': sum(1 for p in predictions if p.is_uncertain),
        }
        
        return SentencePrediction(
            primary_emotion=primary_emotion,
            confidence=confidence,
            probabilities=weighted_probs,
            is_mixed=is_mixed,
            secondary_emotion=secondary_emotion,
            strategy_used='temporal_weighted',
            metadata=metadata
        )
    
    def confidence_threshold(
        self,
        predictions: List[PredictionResult],
        threshold: float = 0.5
    ) -> SentencePrediction:
        """
        Confidence-threshold strategy where only high-confidence predictions contribute
        
        Only predictions with confidence > threshold are included.
        Fallback to all predictions if none meet threshold.
        
        Args:
            predictions: List of word-level predictions
            threshold: Confidence threshold (default: 0.5)
            
        Returns:
            SentencePrediction object
        """
        # Filter predictions by confidence threshold
        high_confidence_preds = [p for p in predictions if p.confidence > threshold]
        
        # Fallback: use all predictions if none meet threshold
        if not high_confidence_preds:
            high_confidence_preds = predictions
            used_fallback = True
        else:
            used_fallback = False
        
        # Apply weighted average to filtered predictions
        emotion_classes = list(predictions[0].probabilities.keys())
        weighted_probs = {emotion: 0.0 for emotion in emotion_classes}
        total_weight = 0.0
        
        for pred in high_confidence_preds:
            weight = pred.confidence
            total_weight += weight
            
            for emotion, prob in pred.probabilities.items():
                weighted_probs[emotion] += weight * prob
        
        # Normalize by total weight
        if total_weight > 0:
            for emotion in weighted_probs:
                weighted_probs[emotion] /= total_weight
        
        # Find primary emotion
        primary_emotion = max(weighted_probs, key=weighted_probs.get)
        confidence = weighted_probs[primary_emotion]
        
        # Detect mixed emotions
        is_mixed, secondary_emotion = self.detect_mixed_emotions(weighted_probs)
        
        # Create metadata
        metadata = {
            'num_words': len(predictions),
            'high_confidence_words': len(high_confidence_preds),
            'threshold': threshold,
            'used_fallback': used_fallback,
            'uncertain_words': sum(1 for p in predictions if p.is_uncertain),
        }
        
        return SentencePrediction(
            primary_emotion=primary_emotion,
            confidence=confidence,
            probabilities=weighted_probs,
            is_mixed=is_mixed,
            secondary_emotion=secondary_emotion,
            strategy_used='confidence_threshold',
            metadata=metadata
        )
    
    def detect_mixed_emotions(
        self,
        probabilities: Dict[str, float],
        threshold: float = 0.5
    ) -> tuple[bool, Optional[str]]:
        """
        Detect if sentence has mixed emotions
        
        Mixed emotion: no single emotion has probability > threshold
        
        Args:
            probabilities: Emotion probability distribution
            threshold: Threshold for dominant emotion (default: 0.5)
            
        Returns:
            Tuple of (is_mixed, secondary_emotion)
        """
        # Sort emotions by probability
        sorted_emotions = sorted(
            probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Check if top emotion exceeds threshold
        top_emotion, top_prob = sorted_emotions[0]
        
        if top_prob <= threshold:
            # Mixed emotions detected
            secondary_emotion = sorted_emotions[1][0] if len(sorted_emotions) > 1 else None
            return True, secondary_emotion
        else:
            # Clear dominant emotion
            return False, None
    
    def aggregate_multi_strategy(
        self,
        predictions: List[PredictionResult],
        strategies: Optional[List[str]] = None
    ) -> Dict[str, SentencePrediction]:
        """
        Apply multiple aggregation strategies and return all results
        
        Args:
            predictions: List of word-level predictions
            strategies: List of strategy names (default: all strategies)
            
        Returns:
            Dictionary mapping strategy names to SentencePrediction objects
        """
        if strategies is None:
            strategies = list(self.strategies.keys())
        
        results = {}
        for strategy in strategies:
            if strategy in self.strategies:
                results[strategy] = self.aggregate(predictions, strategy)
        
        return results
