"""
Property-Based Tests for Word-Level Prediction Module

Feature: turkish-sentence-emotion-analysis
Tests for WordLevelPredictor class validating model output format,
uncertainty flagging, temporal order preservation, and prediction fallback.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from sentence_analysis.word_predictor import WordLevelPredictor, PredictionResult
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import tempfile
import os


# Helper strategies for generating test data
@st.composite
def feature_vector_strategy(draw):
    """
    Generate random feature vector with 378 dimensions.
    """
    # Generate 378 features with reasonable ranges
    features = draw(st.lists(
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=378,
        max_size=378
    ))
    return np.array(features)


@st.composite
def feature_vector_list_strategy(draw):
    """
    Generate list of feature vectors for batch testing.
    """
    n_vectors = draw(st.integers(min_value=1, max_value=10))
    vectors = [draw(feature_vector_strategy()) for _ in range(n_vectors)]
    return vectors


def create_mock_model(temp_dir):
    """
    Create a mock trained model for testing.
    
    Returns:
        Path to the mock model file
    """
    # Create mock training data
    np.random.seed(42)
    X_train = np.random.randn(100, 378)
    y_train = np.random.choice(['angry', 'calm', 'happy', 'sad'], size=100)
    
    # Create and train a simple model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)
    
    model = GradientBoostingClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X_scaled, y_encoded)
    
    # Save model artifacts
    model_artifacts = {
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder
    }
    
    model_path = os.path.join(temp_dir, 'test_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_artifacts, f)
    
    return model_path


# Property 11: Model Output Format
@settings(max_examples=100, deadline=None)
@given(features=feature_vector_strategy())
def test_model_output_format(features):
    """
    **Feature: turkish-sentence-emotion-analysis, Property 11: Model Output Format**
    **Validates: Requirements 4.1, 4.2**
    
    Property: For any valid feature vector input to the Gradient Boosting Classifier,
    the output should be a probability distribution across all four emotion classes
    (angry, calm, happy, sad) that sums to 1.0.
    
    This test verifies that:
    1. Output contains all four emotion classes
    2. Probabilities sum to 1.0
    3. All probabilities are in range [0, 1]
    4. Primary emotion is one of the four valid emotions
    5. Confidence matches the highest probability
    """
    # Create temporary directory for mock model
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = create_mock_model(temp_dir)
        predictor = WordLevelPredictor(model_path)
        
        # Get prediction
        result = predictor.predict(features)
        
        # Property 1: Result should be a PredictionResult object
        assert isinstance(result, PredictionResult), \
            "Result should be a PredictionResult object"
        
        # Property 2: Probabilities should contain all four emotion classes
        expected_emotions = {'angry', 'calm', 'happy', 'sad'}
        assert set(result.probabilities.keys()) == expected_emotions, \
            f"Probabilities should contain all four emotions: {expected_emotions}"
        
        # Property 3: Probabilities should sum to 1.0 (within floating point tolerance)
        prob_sum = sum(result.probabilities.values())
        assert abs(prob_sum - 1.0) < 1e-6, \
            f"Probabilities should sum to 1.0, got {prob_sum}"
        
        # Property 4: All probabilities should be in range [0, 1]
        for emotion, prob in result.probabilities.items():
            assert 0.0 <= prob <= 1.0, \
                f"Probability for {emotion} should be in [0, 1], got {prob}"
        
        # Property 5: Primary emotion should be one of the four valid emotions
        assert result.emotion in expected_emotions, \
            f"Primary emotion should be one of {expected_emotions}, got {result.emotion}"
        
        # Property 6: Confidence should match the highest probability
        max_prob = max(result.probabilities.values())
        assert abs(result.confidence - max_prob) < 1e-6, \
            f"Confidence ({result.confidence}) should match max probability ({max_prob})"
        
        # Property 7: Primary emotion should correspond to highest probability
        emotion_with_max_prob = max(result.probabilities.items(), key=lambda x: x[1])[0]
        assert result.emotion == emotion_with_max_prob, \
            f"Primary emotion ({result.emotion}) should match emotion with highest probability ({emotion_with_max_prob})"
        
        # Property 8: Confidence should be in range [0, 1]
        assert 0.0 <= result.confidence <= 1.0, \
            f"Confidence should be in [0, 1], got {result.confidence}"


# Property 12: Uncertainty Flagging
@settings(max_examples=100, deadline=None)
@given(features=feature_vector_strategy())
def test_uncertainty_flagging(features):
    """
    **Feature: turkish-sentence-emotion-analysis, Property 12: Uncertainty Flagging**
    **Validates: Requirements 4.3**
    
    Property: For any word-level prediction, if the highest probability is below 0.4,
    then the prediction should be flagged as uncertain.
    
    This test verifies that:
    1. is_uncertain flag is set correctly based on confidence threshold
    2. Predictions with confidence < 0.4 are flagged as uncertain
    3. Predictions with confidence >= 0.4 are not flagged as uncertain
    4. The uncertainty threshold can be configured
    """
    # Create temporary directory for mock model
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = create_mock_model(temp_dir)
        
        # Test with default threshold (0.4)
        predictor = WordLevelPredictor(model_path, uncertainty_threshold=0.4)
        result = predictor.predict(features)
        
        # Property 1: is_uncertain should be a boolean
        assert isinstance(result.is_uncertain, bool), \
            "is_uncertain should be a boolean"
        
        # Property 2: is_uncertain should be True if confidence < 0.4
        if result.confidence < 0.4:
            assert result.is_uncertain is True, \
                f"Prediction with confidence {result.confidence} < 0.4 should be flagged as uncertain"
        
        # Property 3: is_uncertain should be False if confidence >= 0.4
        if result.confidence >= 0.4:
            assert result.is_uncertain is False, \
                f"Prediction with confidence {result.confidence} >= 0.4 should not be flagged as uncertain"
        
        # Property 4: Test with different threshold (0.5)
        predictor_high_threshold = WordLevelPredictor(model_path, uncertainty_threshold=0.5)
        result_high = predictor_high_threshold.predict(features)
        
        if result_high.confidence < 0.5:
            assert result_high.is_uncertain is True, \
                f"With threshold 0.5, confidence {result_high.confidence} < 0.5 should be uncertain"
        else:
            assert result_high.is_uncertain is False, \
                f"With threshold 0.5, confidence {result_high.confidence} >= 0.5 should not be uncertain"
        
        # Property 5: Test with low threshold (0.2)
        predictor_low_threshold = WordLevelPredictor(model_path, uncertainty_threshold=0.2)
        result_low = predictor_low_threshold.predict(features)
        
        if result_low.confidence < 0.2:
            assert result_low.is_uncertain is True, \
                f"With threshold 0.2, confidence {result_low.confidence} < 0.2 should be uncertain"
        else:
            assert result_low.is_uncertain is False, \
                f"With threshold 0.2, confidence {result_low.confidence} >= 0.2 should not be uncertain"


# Property 13: Temporal Order Preservation
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.data_too_large])
@given(features_list=feature_vector_list_strategy())
def test_temporal_order_preservation(features_list):
    """
    **Feature: turkish-sentence-emotion-analysis, Property 13: Temporal Order Preservation**
    **Validates: Requirements 4.4**
    
    Property: For any sequence of word segments processed, the output predictions
    should maintain the same temporal order as the input segments.
    
    This test verifies that:
    1. Batch prediction returns same number of results as inputs
    2. Results are in the same order as inputs
    3. Each result corresponds to its input feature vector
    4. Order is preserved even with varying feature values
    """
    # Create temporary directory for mock model
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = create_mock_model(temp_dir)
        predictor = WordLevelPredictor(model_path)
        
        # Get batch predictions
        results = predictor.predict_batch(features_list)
        
        # Property 1: Number of results should match number of inputs
        assert len(results) == len(features_list), \
            f"Number of results ({len(results)}) should match number of inputs ({len(features_list)})"
        
        # Property 2: Each result should be a PredictionResult
        for i, result in enumerate(results):
            assert isinstance(result, PredictionResult), \
                f"Result at index {i} should be a PredictionResult"
        
        # Property 3: Predict each individually and verify order is preserved
        individual_results = [predictor.predict(features) for features in features_list]
        
        for i, (batch_result, individual_result) in enumerate(zip(results, individual_results)):
            # Emotions should match
            assert batch_result.emotion == individual_result.emotion, \
                f"Emotion at index {i} should match: batch={batch_result.emotion}, individual={individual_result.emotion}"
            
            # Confidences should match (within floating point tolerance)
            assert abs(batch_result.confidence - individual_result.confidence) < 1e-6, \
                f"Confidence at index {i} should match: batch={batch_result.confidence}, individual={individual_result.confidence}"
            
            # Probabilities should match
            for emotion in batch_result.probabilities:
                assert abs(batch_result.probabilities[emotion] - individual_result.probabilities[emotion]) < 1e-6, \
                    f"Probability for {emotion} at index {i} should match"
        
        # Property 4: Verify determinism - running batch prediction again should give same results
        results_2 = predictor.predict_batch(features_list)
        
        for i, (result1, result2) in enumerate(zip(results, results_2)):
            assert result1.emotion == result2.emotion, \
                f"Emotion at index {i} should be deterministic"
            assert abs(result1.confidence - result2.confidence) < 1e-6, \
                f"Confidence at index {i} should be deterministic"


# Property 14: Prediction Fallback
@settings(max_examples=100, deadline=None)
@given(seed=st.integers(min_value=0, max_value=10000))
def test_prediction_fallback(seed):
    """
    **Feature: turkish-sentence-emotion-analysis, Property 14: Prediction Fallback**
    **Validates: Requirements 4.5**
    
    Property: For any word segment where model prediction fails, the system should
    assign a uniform probability distribution (0.25 for each emotion) as a fallback.
    
    This test verifies that:
    1. Invalid feature vectors trigger fallback
    2. Fallback returns uniform distribution (0.25 for each emotion)
    3. Fallback result is marked as uncertain
    4. Fallback handles various error conditions gracefully
    """
    np.random.seed(seed)
    
    # Create temporary directory for mock model
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = create_mock_model(temp_dir)
        predictor = WordLevelPredictor(model_path)
        
        # Test 1: Invalid feature vector (wrong dimensions)
        invalid_features_short = np.random.randn(100)  # Too few features
        result_short = predictor.predict(invalid_features_short)
        
        # Should return fallback with uniform distribution
        assert isinstance(result_short, PredictionResult), \
            "Fallback should return PredictionResult"
        
        # Check if probabilities are uniform (0.25 each)
        expected_prob = 0.25
        for emotion, prob in result_short.probabilities.items():
            assert abs(prob - expected_prob) < 1e-6, \
                f"Fallback probability for {emotion} should be {expected_prob}, got {prob}"
        
        # Should be marked as uncertain
        assert result_short.is_uncertain is True, \
            "Fallback prediction should be marked as uncertain"
        
        # Test 2: Invalid feature vector (too many features)
        invalid_features_long = np.random.randn(500)  # Too many features
        result_long = predictor.predict(invalid_features_long)
        
        # Should return fallback with uniform distribution
        for emotion, prob in result_long.probabilities.items():
            assert abs(prob - expected_prob) < 1e-6, \
                f"Fallback probability for {emotion} should be {expected_prob}, got {prob}"
        
        assert result_long.is_uncertain is True, \
            "Fallback prediction should be marked as uncertain"
        
        # Test 3: Feature vector with NaN values
        invalid_features_nan = np.random.randn(378)
        invalid_features_nan[0] = np.nan
        result_nan = predictor.predict(invalid_features_nan)
        
        # Should return fallback with uniform distribution
        for emotion, prob in result_nan.probabilities.items():
            assert abs(prob - expected_prob) < 1e-6, \
                f"Fallback probability for {emotion} should be {expected_prob}, got {prob}"
        
        assert result_nan.is_uncertain is True, \
            "Fallback prediction should be marked as uncertain"
        
        # Test 4: Feature vector with Inf values
        invalid_features_inf = np.random.randn(378)
        invalid_features_inf[0] = np.inf
        result_inf = predictor.predict(invalid_features_inf)
        
        # Should return fallback with uniform distribution
        for emotion, prob in result_inf.probabilities.items():
            assert abs(prob - expected_prob) < 1e-6, \
                f"Fallback probability for {emotion} should be {expected_prob}, got {prob}"
        
        assert result_inf.is_uncertain is True, \
            "Fallback prediction should be marked as uncertain"
        
        # Property: All fallback results should have same structure
        fallback_results = [result_short, result_long, result_nan, result_inf]
        for result in fallback_results:
            # Should have all four emotions
            assert set(result.probabilities.keys()) == {'angry', 'calm', 'happy', 'sad'}, \
                "Fallback should include all four emotions"
            
            # Probabilities should sum to 1.0
            prob_sum = sum(result.probabilities.values())
            assert abs(prob_sum - 1.0) < 1e-6, \
                f"Fallback probabilities should sum to 1.0, got {prob_sum}"
            
            # Should be marked as uncertain
            assert result.is_uncertain is True, \
                "Fallback should be marked as uncertain"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
