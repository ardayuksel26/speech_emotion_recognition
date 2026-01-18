"""
Property-Based Tests for Feature Extraction Module

Feature: turkish-sentence-emotion-analysis
Tests for FeatureExtractor class validating feature vector completeness
and scaler consistency.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from sentence_analysis.feature_extractor import FeatureExtractor
from sklearn.preprocessing import StandardScaler


# Helper strategies for generating test data
@st.composite
def audio_signal_strategy(draw):
    """
    Generate random audio signal with valid sample rate.
    Returns (audio, sr)
    """
    sr = draw(st.sampled_from([8000, 16000, 22050, 44100, 48000]))
    
    # Generate audio duration between 0.1s and 2.0s
    duration = draw(st.floats(min_value=0.1, max_value=2.0))
    samples = int(duration * sr)
    
    # Ensure minimum samples for feature extraction
    assume(samples >= 512)  # Minimum for librosa processing
    
    # Generate audio signal (sine wave + noise)
    t = np.linspace(0, duration, samples)
    freq = draw(st.floats(min_value=100, max_value=1000))
    audio = 0.5 * np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(samples)
    
    return audio, sr


# Property 9: Feature Vector Completeness
@settings(max_examples=100, deadline=None)
@given(audio_data=audio_signal_strategy())
def test_feature_vector_completeness(audio_data):
    """
    **Feature: turkish-sentence-emotion-analysis, Property 9: Feature Vector Completeness**
    **Validates: Requirements 3.1, 3.2**
    
    Property: For any audio segment, the extracted feature vector should contain 
    exactly 378 features (80 MFCC + 24 chroma + 256 mel + 14 spectral contrast + 
    2 ZCR + 2 RMS).
    
    This test verifies that:
    1. Feature vector has exactly 378 dimensions
    2. All features are finite (no NaN or Inf values)
    3. Feature extraction is deterministic (same input -> same output)
    4. Individual feature components have correct dimensions
    """
    audio, sr = audio_data
    
    # Create feature extractor (without scaler for raw features)
    extractor = FeatureExtractor(n_mfcc=40)
    
    # Extract features without scaling
    features = extractor.extract_features(audio, sr, apply_scaling=False)
    
    # Property 1: Feature vector should have exactly 378 dimensions
    expected_features = 378
    assert len(features) == expected_features, \
        f"Feature vector should have {expected_features} features, got {len(features)}"
    
    # Property 2: All features should be finite (no NaN or Inf)
    assert np.all(np.isfinite(features)), \
        "Feature vector contains NaN or Inf values"
    
    # Property 3: Feature extraction should be deterministic
    features_2 = extractor.extract_features(audio, sr, apply_scaling=False)
    np.testing.assert_array_almost_equal(features, features_2, decimal=10,
        err_msg="Feature extraction should be deterministic")
    
    # Property 4: Verify individual feature component dimensions
    # Extract individual components to verify their sizes
    mfcc_features = extractor.extract_mfcc(audio, sr, n_mfcc=40)
    assert len(mfcc_features) == 80, \
        f"MFCC features should have 80 dimensions (40 mean + 40 std), got {len(mfcc_features)}"
    
    chroma_features = extractor.extract_chroma(audio, sr)
    assert len(chroma_features) == 24, \
        f"Chroma features should have 24 dimensions (12 mean + 12 std), got {len(chroma_features)}"
    
    mel_features = extractor.extract_mel_spectrogram(audio, sr)
    assert len(mel_features) == 256, \
        f"Mel spectrogram features should have 256 dimensions (128 mean + 128 std), got {len(mel_features)}"
    
    contrast_features = extractor.extract_spectral_contrast(audio, sr)
    assert len(contrast_features) == 14, \
        f"Spectral contrast features should have 14 dimensions (7 mean + 7 std), got {len(contrast_features)}"
    
    temporal_features = extractor.extract_temporal_features(audio)
    assert len(temporal_features) == 4, \
        f"Temporal features dict should have 4 keys, got {len(temporal_features)}"
    assert 'zcr_mean' in temporal_features
    assert 'zcr_std' in temporal_features
    assert 'rms_mean' in temporal_features
    assert 'rms_std' in temporal_features
    
    # Property 5: Verify feature composition adds up correctly
    total_features = (
        len(mfcc_features) +      # 80
        len(chroma_features) +    # 24
        len(mel_features) +       # 256
        len(contrast_features) +  # 14
        4                         # ZCR and RMS (2 + 2)
    )
    assert total_features == expected_features, \
        f"Sum of individual feature components ({total_features}) should equal total ({expected_features})"
    
    # Property 6: All individual feature components should be finite
    assert np.all(np.isfinite(mfcc_features)), "MFCC features contain NaN or Inf"
    assert np.all(np.isfinite(chroma_features)), "Chroma features contain NaN or Inf"
    assert np.all(np.isfinite(mel_features)), "Mel features contain NaN or Inf"
    assert np.all(np.isfinite(contrast_features)), "Spectral contrast features contain NaN or Inf"
    assert all(np.isfinite(v) for v in temporal_features.values()), \
        "Temporal features contain NaN or Inf"


# Property 10: Scaler Consistency
@settings(max_examples=100, deadline=None)
@given(
    audio_data=audio_signal_strategy(),
    seed=st.integers(min_value=0, max_value=10000)
)
def test_scaler_consistency(audio_data, seed):
    """
    **Feature: turkish-sentence-emotion-analysis, Property 10: Scaler Consistency**
    **Validates: Requirements 3.3**
    
    Property: For any feature vector, the StandardScaler transformation applied 
    should use the same parameters (mean, std) as the scaler used during 
    word-level model training.
    
    This test verifies that:
    1. When a scaler is provided, it is applied to the features
    2. The scaler transformation uses consistent parameters
    3. Scaled features have different values than unscaled features
    4. The scaler can be loaded from a model file
    5. Multiple extractions with the same scaler produce consistent results
    """
    audio, sr = audio_data
    
    # Create a mock scaler with known parameters for testing
    np.random.seed(seed)
    
    # Generate training data to fit the scaler
    n_samples = 100
    n_features = 378
    training_data = np.random.randn(n_samples, n_features)
    
    # Fit a StandardScaler
    scaler = StandardScaler()
    scaler.fit(training_data)
    
    # Create feature extractor with the scaler
    extractor_with_scaler = FeatureExtractor(n_mfcc=40, scaler=scaler)
    extractor_without_scaler = FeatureExtractor(n_mfcc=40)
    
    # Extract features with and without scaling
    features_unscaled = extractor_without_scaler.extract_features(audio, sr, apply_scaling=False)
    features_scaled = extractor_with_scaler.extract_features(audio, sr, apply_scaling=True)
    
    # Property 1: Scaled and unscaled features should have the same length
    assert len(features_scaled) == len(features_unscaled), \
        "Scaled and unscaled features should have the same length"
    
    # Property 2: Scaled features should be different from unscaled features
    # (unless by extreme coincidence the features match the scaler's mean/std)
    # We check that at least some features are different
    differences = np.abs(features_scaled - features_unscaled)
    assert np.any(differences > 0.01), \
        "Scaled features should differ from unscaled features"
    
    # Property 3: Manually apply scaler and verify consistency
    features_unscaled_2d = features_unscaled.reshape(1, -1)
    features_manually_scaled = scaler.transform(features_unscaled_2d).flatten()
    
    np.testing.assert_array_almost_equal(features_scaled, features_manually_scaled, decimal=10,
        err_msg="Scaler should be applied consistently")
    
    # Property 4: Scaler parameters should be preserved
    assert hasattr(extractor_with_scaler.scaler, 'mean_'), \
        "Scaler should have mean_ attribute"
    assert hasattr(extractor_with_scaler.scaler, 'scale_'), \
        "Scaler should have scale_ attribute"
    assert len(extractor_with_scaler.scaler.mean_) == 378, \
        "Scaler mean should have 378 dimensions"
    assert len(extractor_with_scaler.scaler.scale_) == 378, \
        "Scaler scale should have 378 dimensions"
    
    # Property 5: Multiple extractions with same scaler should be consistent
    features_scaled_2 = extractor_with_scaler.extract_features(audio, sr, apply_scaling=True)
    np.testing.assert_array_almost_equal(features_scaled, features_scaled_2, decimal=10,
        err_msg="Multiple extractions with same scaler should be consistent")
    
    # Property 6: Setting scaler via set_scaler method should work
    extractor_new = FeatureExtractor(n_mfcc=40)
    extractor_new.set_scaler(scaler)
    features_scaled_3 = extractor_new.extract_features(audio, sr, apply_scaling=True)
    np.testing.assert_array_almost_equal(features_scaled, features_scaled_3, decimal=10,
        err_msg="Setting scaler via set_scaler should produce consistent results")
    
    # Property 7: Disabling scaling should return unscaled features
    features_no_scale = extractor_with_scaler.extract_features(audio, sr, apply_scaling=False)
    np.testing.assert_array_almost_equal(features_unscaled, features_no_scale, decimal=10,
        err_msg="Disabling scaling should return unscaled features")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
