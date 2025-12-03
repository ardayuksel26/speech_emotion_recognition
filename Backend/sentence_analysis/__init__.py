"""
Turkish Sentence Emotion Analysis Module

This module provides sentence-level emotion analysis by segmenting audio,
extracting features, and aggregating word-level predictions.
"""

from .audio_segmenter import AudioSegmenter, AudioSegment
from .feature_extractor import FeatureExtractor
from .word_predictor import WordLevelPredictor, PredictionResult
from .aggregation_engine import AggregationEngine, SentencePrediction
from .job_queue import JobQueueManager, JobStatus

__all__ = [
    'AudioSegmenter',
    'AudioSegment',
    'FeatureExtractor',
    'WordLevelPredictor',
    'PredictionResult',
    'AggregationEngine',
    'SentencePrediction',
    'JobQueueManager',
    'JobStatus',
]
