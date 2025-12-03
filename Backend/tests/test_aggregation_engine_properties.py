"""
Property-Based Tests for Aggregation Engine Module

Feature: turkish-sentence-emotion-analysis
Tests for AggregationEngine class validating weighted average, majority voting,
temporal weighting, confidence threshold, multi-strategy execution, and mixed emotion detection.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from sentence_analysis.aggregation_engine import AggregationEngine, SentencePrediction
from sentence_analysis.word_predictor import PredictionResult


# Helper strategies for generating test data
@st.composite
def prediction_result_strategy(draw):
    """
    Generate random PredictionResult object.
    """
    emotions = ['angry', 'calm', 'happy', 'sad']
    
    # Generate random probabilities that sum to 1.0
    raw_probs = [draw(st.floats(min_value=0.01, max_value=1.0)) for _ in range(4)]
    total = sum(raw_probs)
    probabilities = {emotion: prob / total for emotion, prob in zip(emotions, raw_probs)}
    
    # Primary emotion is the one with highest probability
    primary_emotion = max(probabilities, key=probabilities.get)
    confidence = probabilities[primary_emotion]
    
    # Determine if uncertain (confidence < 0.4)
    is_uncertain = confidence < 0.4
    
    return PredictionResult(
        emotion=primary_emotion,
        confidence=confidence,
        probabilities=probabilities,
        is_uncertain=is_uncertain
    )


@st.composite
def prediction_list_strategy(draw):
    """
    Generate list of PredictionResult objects.
    """
    n_predictions = draw(st.integers(min_value=1, max_value=20))
    predictions = [draw(prediction_result_strategy()) for _ in range(n_predictions)]
    return predictions


# Property 15: Weighted Average Aggregation
@settings(max_examples=100, deadline=None)
@given(predictions=prediction_list_strategy())
def test_weighted_average_aggregation(predictions):
    """
    **Feature: turkish-sentence-emotion-analysis, Property 15: Weighted Average Aggregation**
    **Validates: Requirements 5.1**
    
    Property: For any list of word-level predictions with confidence scores,
    the weighted average aggregation should compute sentence-level probabilities
    as Σ(confidence_i × prob_i) / Σ(confidence_i).
    
    This test verifies that:
    1. Weighted average formula is correctly applied
    2. Result probabilities sum to 1.0
    3. Primary emotion has highest probability
    4. Confidence matches primary emotion probability
    5. Metadata includes correct information
    """
    engine = AggregationEngine()
    
    # Apply weighted average aggregation
    result = engine.weighted_average(predictions)
    
    # Property 1: Result should be a SentencePrediction object
    assert isinstance(result, SentencePrediction), \
        "Result should be a SentencePrediction object"
    
    # Property 2: Strategy should be 'weighted_average'
    assert result.strategy_used == 'weighted_average', \
        f"Strategy should be 'weighted_average', got {result.strategy_used}"
    
    # Property 3: Manually compute weighted average and verify
    emotion_classes = ['angry', 'calm', 'happy', 'sad']
    expected_probs = {emotion: 0.0 for emotion in emotion_classes}
    total_weight = sum(p.confidence for p in predictions)
    
    for pred in predictions:
        weight = pred.confidence
        for emotion, prob in pred.probabilities.items():
            expected_probs[emotion] += weight * prob
    
    for emotion in expected_probs:
        expected_probs[emotion] /= total_weight
    
    # Verify computed probabilities match expected
    for emotion in emotion_classes:
        assert abs(result.probabilities[emotion] - expected_probs[emotion]) < 1e-6, \
            f"Probability for {emotion} should be {expected_probs[emotion]}, got {result.probabilities[emotion]}"
    
    # Property 4: Probabilities should sum to 1.0
    prob_sum = sum(result.probabilities.values())
    assert abs(prob_sum - 1.0) < 1e-6, \
        f"Probabilities should sum to 1.0, got {prob_sum}"
    
    # Property 5: Primary emotion should have highest probability
    max_prob = max(result.probabilities.values())
    assert abs(result.confidence - max_prob) < 1e-6, \
        f"Confidence should match max probability"
    
    assert result.probabilities[result.primary_emotion] == max_prob, \
        f"Primary emotion should have highest probability"
    
    # Property 6: Metadata should include num_words
    assert 'num_words' in result.metadata, \
        "Metadata should include num_words"
    assert result.metadata['num_words'] == len(predictions), \
        f"num_words should be {len(predictions)}, got {result.metadata['num_words']}"
    
    # Property 7: All probabilities should be in [0, 1]
    for emotion, prob in result.probabilities.items():
        assert 0.0 <= prob <= 1.0, \
            f"Probability for {emotion} should be in [0, 1], got {prob}"


# Property 16: Majority Voting Aggregation
@settings(max_examples=100, deadline=None)
@given(predictions=prediction_list_strategy())
def test_majority_voting_aggregation(predictions):
    """
    **Feature: turkish-sentence-emotion-analysis, Property 16: Majority Voting Aggregation**
    **Validates: Requirements 5.2**
    
    Property: For any list of word-level predictions, the majority voting aggregation
    should select the emotion that appears most frequently, with ties broken by
    highest average confidence.
    
    This test verifies that:
    1. Most frequent emotion is selected
    2. Ties are broken by average confidence
    3. Confidence equals vote proportion
    4. All emotions are represented in probabilities
    5. Metadata includes vote counts
    """
    engine = AggregationEngine()
    
    # Apply majority voting aggregation
    result = engine.majority_voting(predictions)
    
    # Property 1: Result should be a SentencePrediction object
    assert isinstance(result, SentencePrediction), \
        "Result should be a SentencePrediction object"
    
    # Property 2: Strategy should be 'majority_voting'
    assert result.strategy_used == 'majority_voting', \
        f"Strategy should be 'majority_voting', got {result.strategy_used}"
    
    # Property 3: Count emotion occurrences
    from collections import Counter
    emotion_counts = Counter(pred.emotion for pred in predictions)
    max_count = max(emotion_counts.values())
    
    # Property 4: Primary emotion should be one with max count
    assert emotion_counts[result.primary_emotion] == max_count, \
        f"Primary emotion should have max count {max_count}, got {emotion_counts[result.primary_emotion]}"
    
    # Property 5: Confidence should equal vote proportion
    expected_confidence = emotion_counts[result.primary_emotion] / len(predictions)
    assert abs(result.confidence - expected_confidence) < 1e-6, \
        f"Confidence should be {expected_confidence}, got {result.confidence}"
    
    # Property 6: Probabilities should match vote proportions
    for emotion, count in emotion_counts.items():
        expected_prob = count / len(predictions)
        assert abs(result.probabilities[emotion] - expected_prob) < 1e-6, \
            f"Probability for {emotion} should be {expected_prob}, got {result.probabilities[emotion]}"
    
    # Property 7: All emotion classes should be present in probabilities
    emotion_classes = {'angry', 'calm', 'happy', 'sad'}
    assert set(result.probabilities.keys()) == emotion_classes, \
        f"All emotions should be in probabilities"
    
    # Property 8: Probabilities should sum to 1.0
    prob_sum = sum(result.probabilities.values())
    assert abs(prob_sum - 1.0) < 1e-6, \
        f"Probabilities should sum to 1.0, got {prob_sum}"
    
    # Property 9: Metadata should include vote_counts
    assert 'vote_counts' in result.metadata, \
        "Metadata should include vote_counts"
    
    # Property 10: Metadata should include num_words
    assert result.metadata['num_words'] == len(predictions), \
        f"num_words should be {len(predictions)}"


# Property 17: Temporal Weighting Aggregation
@settings(max_examples=100, deadline=None)
@given(predictions=prediction_list_strategy())
def test_temporal_weighting_aggregation(predictions):
    """
    **Feature: turkish-sentence-emotion-analysis, Property 17: Temporal Weighting Aggregation**
    **Validates: Requirements 5.3**
    
    Property: For any list of n word-level predictions, the temporal-weighted
    aggregation should apply weights where weight_i = (i + 1) / n, giving later
    words progressively higher influence.
    
    This test verifies that:
    1. Temporal weights are correctly applied (linear increase)
    2. Later words have higher influence
    3. Result probabilities sum to 1.0
    4. Formula weight_i = (i + 1) / n is correctly implemented
    5. Metadata indicates recency bias
    """
    engine = AggregationEngine()
    
    # Apply temporal weighted aggregation
    result = engine.temporal_weighted(predictions)
    
    # Property 1: Result should be a SentencePrediction object
    assert isinstance(result, SentencePrediction), \
        "Result should be a SentencePrediction object"
    
    # Property 2: Strategy should be 'temporal_weighted'
    assert result.strategy_used == 'temporal_weighted', \
        f"Strategy should be 'temporal_weighted', got {result.strategy_used}"
    
    # Property 3: Manually compute temporal-weighted probabilities
    n = len(predictions)
    emotion_classes = ['angry', 'calm', 'happy', 'sad']
    expected_probs = {emotion: 0.0 for emotion in emotion_classes}
    total_weight = 0.0
    
    for i, pred in enumerate(predictions):
        # Temporal weight: (i + 1) / n
        weight = (i + 1) / n
        total_weight += weight
        
        for emotion, prob in pred.probabilities.items():
            expected_probs[emotion] += weight * prob
    
    # Normalize
    for emotion in expected_probs:
        expected_probs[emotion] /= total_weight
    
    # Verify computed probabilities match expected
    for emotion in emotion_classes:
        assert abs(result.probabilities[emotion] - expected_probs[emotion]) < 1e-6, \
            f"Probability for {emotion} should be {expected_probs[emotion]}, got {result.probabilities[emotion]}"
    
    # Property 4: Probabilities should sum to 1.0
    prob_sum = sum(result.probabilities.values())
    assert abs(prob_sum - 1.0) < 1e-6, \
        f"Probabilities should sum to 1.0, got {prob_sum}"
    
    # Property 5: Primary emotion should have highest probability
    max_prob = max(result.probabilities.values())
    assert result.probabilities[result.primary_emotion] == max_prob, \
        "Primary emotion should have highest probability"
    
    # Property 6: Metadata should indicate recency bias
    assert 'recency_bias' in result.metadata, \
        "Metadata should include recency_bias"
    assert result.metadata['recency_bias'] is True, \
        "recency_bias should be True"
    
    # Property 7: Metadata should include num_words
    assert result.metadata['num_words'] == len(predictions), \
        f"num_words should be {len(predictions)}"
    
    # Property 8: Verify that later predictions have more influence
    # If we have at least 2 predictions, verify weight increases
    if len(predictions) >= 2:
        weight_first = 1 / n
        weight_last = n / n
        assert weight_last > weight_first, \
            "Last word should have higher weight than first word"


# Property 18: Confidence Threshold Aggregation
@settings(max_examples=100, deadline=None)
@given(predictions=prediction_list_strategy(), threshold=st.floats(min_value=0.3, max_value=0.7))
def test_confidence_threshold_aggregation(predictions, threshold):
    """
    **Feature: turkish-sentence-emotion-analysis, Property 18: Confidence Threshold Aggregation**
    **Validates: Requirements 5.4**
    
    Property: For any list of word-level predictions, the confidence-threshold
    aggregation should only include predictions with confidence > threshold in
    the final calculation.
    
    This test verifies that:
    1. Only high-confidence predictions are used
    2. Fallback to all predictions if none meet threshold
    3. Result probabilities sum to 1.0
    4. Metadata indicates threshold and fallback usage
    5. High-confidence count is correct
    """
    engine = AggregationEngine()
    
    # Apply confidence threshold aggregation
    result = engine.confidence_threshold(predictions, threshold=threshold)
    
    # Property 1: Result should be a SentencePrediction object
    assert isinstance(result, SentencePrediction), \
        "Result should be a SentencePrediction object"
    
    # Property 2: Strategy should be 'confidence_threshold'
    assert result.strategy_used == 'confidence_threshold', \
        f"Strategy should be 'confidence_threshold', got {result.strategy_used}"
    
    # Property 3: Count high-confidence predictions
    high_confidence_preds = [p for p in predictions if p.confidence > threshold]
    
    # Property 4: Metadata should include threshold
    assert 'threshold' in result.metadata, \
        "Metadata should include threshold"
    assert abs(result.metadata['threshold'] - threshold) < 1e-6, \
        f"Threshold should be {threshold}, got {result.metadata['threshold']}"
    
    # Property 5: Metadata should include high_confidence_words count
    assert 'high_confidence_words' in result.metadata, \
        "Metadata should include high_confidence_words"
    
    expected_high_conf_count = len(high_confidence_preds) if high_confidence_preds else len(predictions)
    assert result.metadata['high_confidence_words'] == expected_high_conf_count, \
        f"high_confidence_words should be {expected_high_conf_count}, got {result.metadata['high_confidence_words']}"
    
    # Property 6: Metadata should indicate if fallback was used
    assert 'used_fallback' in result.metadata, \
        "Metadata should include used_fallback"
    
    if not high_confidence_preds:
        assert result.metadata['used_fallback'] is True, \
            "used_fallback should be True when no predictions meet threshold"
    else:
        assert result.metadata['used_fallback'] is False, \
            "used_fallback should be False when predictions meet threshold"
    
    # Property 7: Probabilities should sum to 1.0
    prob_sum = sum(result.probabilities.values())
    assert abs(prob_sum - 1.0) < 1e-6, \
        f"Probabilities should sum to 1.0, got {prob_sum}"
    
    # Property 8: All probabilities should be in [0, 1]
    for emotion, prob in result.probabilities.items():
        assert 0.0 <= prob <= 1.0, \
            f"Probability for {emotion} should be in [0, 1], got {prob}"
    
    # Property 9: Primary emotion should have highest probability
    max_prob = max(result.probabilities.values())
    assert result.probabilities[result.primary_emotion] == max_prob, \
        "Primary emotion should have highest probability"
    
    # Property 10: Metadata should include num_words
    assert result.metadata['num_words'] == len(predictions), \
        f"num_words should be {len(predictions)}"


# Property 19: Multi-Strategy Execution
@settings(max_examples=100, deadline=None)
@given(predictions=prediction_list_strategy())
def test_multi_strategy_execution(predictions):
    """
    **Feature: turkish-sentence-emotion-analysis, Property 19: Multi-Strategy Execution**
    **Validates: Requirements 5.5**
    
    Property: For any sentence analysis request, when multiple aggregation strategies
    are applied, the response should contain results from all requested strategies.
    
    This test verifies that:
    1. All requested strategies are executed
    2. Each strategy returns a valid SentencePrediction
    3. Results are keyed by strategy name
    4. Each result has correct strategy_used field
    5. Default behavior executes all strategies
    """
    engine = AggregationEngine()
    
    # Test 1: Execute all strategies (default)
    results_all = engine.aggregate_multi_strategy(predictions)
    
    # Property 1: Should return dictionary
    assert isinstance(results_all, dict), \
        "aggregate_multi_strategy should return a dictionary"
    
    # Property 2: Should include all available strategies
    expected_strategies = {'weighted_average', 'majority_voting', 'temporal_weighted', 'confidence_threshold'}
    assert set(results_all.keys()) == expected_strategies, \
        f"Should include all strategies: {expected_strategies}, got {set(results_all.keys())}"
    
    # Property 3: Each result should be a SentencePrediction
    for strategy_name, result in results_all.items():
        assert isinstance(result, SentencePrediction), \
            f"Result for {strategy_name} should be a SentencePrediction"
        
        # Property 4: strategy_used should match the key
        assert result.strategy_used == strategy_name, \
            f"strategy_used should be {strategy_name}, got {result.strategy_used}"
        
        # Property 5: Result should have valid probabilities
        prob_sum = sum(result.probabilities.values())
        assert abs(prob_sum - 1.0) < 1e-6, \
            f"Probabilities for {strategy_name} should sum to 1.0, got {prob_sum}"
    
    # Test 2: Execute specific strategies
    specific_strategies = ['weighted_average', 'majority_voting']
    results_specific = engine.aggregate_multi_strategy(predictions, strategies=specific_strategies)
    
    # Property 6: Should only include requested strategies
    assert set(results_specific.keys()) == set(specific_strategies), \
        f"Should only include {specific_strategies}, got {set(results_specific.keys())}"
    
    # Property 7: Each requested strategy should have a result
    for strategy_name in specific_strategies:
        assert strategy_name in results_specific, \
            f"Should include result for {strategy_name}"
        assert isinstance(results_specific[strategy_name], SentencePrediction), \
            f"Result for {strategy_name} should be a SentencePrediction"
    
    # Test 3: Execute single strategy
    single_strategy = ['temporal_weighted']
    results_single = engine.aggregate_multi_strategy(predictions, strategies=single_strategy)
    
    # Property 8: Should include only the single strategy
    assert set(results_single.keys()) == set(single_strategy), \
        f"Should only include {single_strategy}, got {set(results_single.keys())}"
    
    # Property 9: Results should be consistent with individual calls
    for strategy_name in expected_strategies:
        individual_result = engine.aggregate(predictions, strategy=strategy_name)
        multi_result = results_all[strategy_name]
        
        # Primary emotions should match
        assert individual_result.primary_emotion == multi_result.primary_emotion, \
            f"Primary emotion for {strategy_name} should match"
        
        # Confidences should match
        assert abs(individual_result.confidence - multi_result.confidence) < 1e-6, \
            f"Confidence for {strategy_name} should match"
        
        # Probabilities should match
        for emotion in individual_result.probabilities:
            assert abs(individual_result.probabilities[emotion] - multi_result.probabilities[emotion]) < 1e-6, \
                f"Probability for {emotion} in {strategy_name} should match"


# Property 21: Mixed Emotion Detection
@settings(max_examples=100, deadline=None)
@given(predictions=prediction_list_strategy())
def test_mixed_emotion_detection(predictions):
    """
    **Feature: turkish-sentence-emotion-analysis, Property 21: Mixed Emotion Detection**
    **Validates: Requirements 6.4**
    
    Property: For any sentence-level prediction, if no single emotion has
    probability > 0.5, then the result should be flagged as "mixed" and include
    the top two emotions.
    
    This test verifies that:
    1. Mixed emotion flag is set correctly
    2. Secondary emotion is provided when mixed
    3. Secondary emotion has second-highest probability
    4. Non-mixed emotions have no secondary emotion
    5. Threshold of 0.5 is correctly applied
    """
    engine = AggregationEngine()
    
    # Apply weighted average aggregation (any strategy works)
    result = engine.weighted_average(predictions)
    
    # Property 1: is_mixed should be a boolean
    assert isinstance(result.is_mixed, bool), \
        "is_mixed should be a boolean"
    
    # Property 2: Get max probability
    max_prob = max(result.probabilities.values())
    
    # Property 3: If max_prob <= 0.5, should be flagged as mixed
    if max_prob <= 0.5:
        assert result.is_mixed is True, \
            f"Should be flagged as mixed when max probability ({max_prob}) <= 0.5"
        
        # Property 4: Should have secondary emotion
        assert result.secondary_emotion is not None, \
            "Mixed emotion should have secondary emotion"
        
        # Property 5: Secondary emotion should be different from primary
        assert result.secondary_emotion != result.primary_emotion, \
            "Secondary emotion should be different from primary"
        
        # Property 6: Secondary emotion should have second-highest probability
        sorted_emotions = sorted(
            result.probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        expected_secondary = sorted_emotions[1][0]
        assert result.secondary_emotion == expected_secondary, \
            f"Secondary emotion should be {expected_secondary}, got {result.secondary_emotion}"
    
    # Property 7: If max_prob > 0.5, should not be flagged as mixed
    if max_prob > 0.5:
        assert result.is_mixed is False, \
            f"Should not be flagged as mixed when max probability ({max_prob}) > 0.5"
    
    # Property 8: Test detect_mixed_emotions method directly
    is_mixed_direct, secondary_direct = engine.detect_mixed_emotions(result.probabilities, threshold=0.5)
    
    # Should match the result
    assert is_mixed_direct == result.is_mixed, \
        "Direct detection should match result.is_mixed"
    
    if is_mixed_direct:
        assert secondary_direct == result.secondary_emotion, \
            "Direct detection should match result.secondary_emotion"
    
    # Property 9: Test with different threshold
    is_mixed_high, secondary_high = engine.detect_mixed_emotions(result.probabilities, threshold=0.7)
    
    max_prob = max(result.probabilities.values())
    if max_prob <= 0.7:
        assert is_mixed_high is True, \
            f"Should be mixed with threshold 0.7 when max prob is {max_prob}"
    else:
        assert is_mixed_high is False, \
            f"Should not be mixed with threshold 0.7 when max prob is {max_prob}"
    
    # Property 10: Test with low threshold
    is_mixed_low, secondary_low = engine.detect_mixed_emotions(result.probabilities, threshold=0.3)
    
    if max_prob <= 0.3:
        assert is_mixed_low is True, \
            f"Should be mixed with threshold 0.3 when max prob is {max_prob}"
    else:
        assert is_mixed_low is False, \
            f"Should not be mixed with threshold 0.3 when max prob is {max_prob}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
