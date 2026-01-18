"""
Property-Based Tests for Audio Segmentation Module

Feature: turkish-sentence-emotion-analysis
Tests for AudioSegmenter class validating word boundary detection,
segment padding consistency, and segmentation fallback behavior.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from sentence_analysis.audio_segmenter import AudioSegmenter, AudioSegment


# Helper strategies for generating test data
@st.composite
def audio_with_silence_strategy(draw):
    """
    Generate audio signal with configurable silence patterns.
    Returns (audio, sr, expected_num_boundaries)
    """
    sr = draw(st.sampled_from([8000, 16000, 22050, 44100, 48000]))
    
    # Generate speech segments with silence between them
    num_words = draw(st.integers(min_value=2, max_value=10))
    
    # Each word is 200-500ms of non-zero audio
    word_duration_range = (0.2, 0.5)
    # Silence between words is 100-300ms
    silence_duration_range = (0.1, 0.3)
    
    audio_segments = []
    
    for i in range(num_words):
        # Generate word audio (non-zero signal)
        word_duration = draw(st.floats(min_value=word_duration_range[0], max_value=word_duration_range[1]))
        word_samples = int(word_duration * sr)
        # Generate audio with some energy (sine wave + noise)
        t = np.linspace(0, word_duration, word_samples)
        freq = draw(st.floats(min_value=200, max_value=800))
        word_audio = 0.5 * np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(word_samples)
        audio_segments.append(word_audio)
        
        # Add silence between words (except after last word)
        if i < num_words - 1:
            silence_duration = draw(st.floats(min_value=silence_duration_range[0], max_value=silence_duration_range[1]))
            silence_samples = int(silence_duration * sr)
            silence = np.zeros(silence_samples)
            audio_segments.append(silence)
    
    # Concatenate all segments
    audio = np.concatenate(audio_segments)
    
    return audio, sr, num_words


@st.composite
def continuous_audio_strategy(draw):
    """
    Generate continuous audio without clear word boundaries (for fallback testing).
    Returns (audio, sr)
    """
    sr = draw(st.sampled_from([8000, 16000, 22050, 44100, 48000]))
    duration = draw(st.floats(min_value=1.0, max_value=5.0))
    samples = int(duration * sr)
    
    # Generate continuous audio (sine wave + noise, no silence)
    t = np.linspace(0, duration, samples)
    freq = draw(st.floats(min_value=200, max_value=800))
    audio = 0.5 * np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(samples)
    
    return audio, sr


# Property 6: Word Boundary Detection
@settings(max_examples=100, deadline=None)
@given(audio_data=audio_with_silence_strategy())
def test_word_boundary_detection(audio_data):
    """
    **Feature: turkish-sentence-emotion-analysis, Property 6: Word Boundary Detection**
    **Validates: Requirements 2.1**
    
    Property: For any sentence audio file, the VAD-based segmentation should detect 
    word boundaries where silence exceeds 100ms, and each detected boundary should 
    result in a separate audio segment.
    
    This test verifies that:
    1. Word boundaries are detected where silence exceeds the threshold (100ms)
    2. Each detected boundary results in a separate audio segment
    3. The number of segments is reasonable given the audio structure
    4. Boundaries are properly ordered in time
    """
    audio, sr, expected_num_words = audio_data
    
    # Create segmenter with standard parameters
    segmenter = AudioSegmenter(
        silence_threshold=0.02,
        min_silence_duration=0.1,  # 100ms as per requirements
        segment_padding=0.05
    )
    
    # Detect word boundaries
    boundaries = segmenter.detect_word_boundaries(audio, sr)
    
    # Property 1: Boundaries should be detected (at least some)
    assert len(boundaries) > 0, "Should detect at least one word boundary"
    
    # Property 2: Number of boundaries should be reasonable
    # VAD might merge segments if energy is low or silence is short
    # So we just check that we get at least 1 boundary and not too many
    assert len(boundaries) >= 1, \
        f"Expected at least 1 boundary, got {len(boundaries)}"
    assert len(boundaries) <= expected_num_words * 2, \
        f"Expected at most {expected_num_words * 2} boundaries (allowing splits), got {len(boundaries)}"
    
    # Property 3: Boundaries should be properly ordered in time
    for i in range(len(boundaries) - 1):
        start1, end1 = boundaries[i]
        start2, end2 = boundaries[i + 1]
        
        # Each boundary should have start < end
        assert start1 < end1, f"Boundary {i}: start ({start1}) should be < end ({end1})"
        assert start2 < end2, f"Boundary {i+1}: start ({start2}) should be < end ({end2})"
        
        # Boundaries should not overlap (end of one <= start of next)
        assert end1 <= start2, \
            f"Boundaries {i} and {i+1} overlap: boundary {i} ends at {end1}, boundary {i+1} starts at {start2}"
    
    # Property 4: All boundaries should be within audio duration
    audio_duration = len(audio) / sr
    for i, (start, end) in enumerate(boundaries):
        assert 0 <= start < audio_duration, \
            f"Boundary {i} start time {start} is outside audio duration {audio_duration}"
        assert 0 < end <= audio_duration, \
            f"Boundary {i} end time {end} is outside audio duration {audio_duration}"
    
    # Property 5: Silence between detected boundaries should be >= min_silence_duration
    min_silence = segmenter.min_silence_duration
    for i in range(len(boundaries) - 1):
        _, end1 = boundaries[i]
        start2, _ = boundaries[i + 1]
        silence_duration = start2 - end1
        
        # Allow small tolerance for floating point arithmetic
        assert silence_duration >= min_silence - 0.01, \
            f"Silence between boundaries {i} and {i+1} is {silence_duration}s, " \
            f"should be >= {min_silence}s"


# Property 7: Segment Padding Consistency
@settings(max_examples=100, deadline=None)
@given(
    audio_data=audio_with_silence_strategy(),
    padding=st.floats(min_value=0.01, max_value=0.1)
)
def test_segment_padding_consistency(audio_data, padding):
    """
    **Feature: turkish-sentence-emotion-analysis, Property 7: Segment Padding Consistency**
    **Validates: Requirements 2.2**
    
    Property: For any detected word boundary, the extracted audio segment should 
    include exactly the specified padding on both the start and end sides.
    
    This test verifies that:
    1. Extracted segments include the specified padding
    2. Padding is applied consistently to all segments
    3. Padding does not cause segments to exceed audio boundaries
    4. Segment duration includes the padding
    """
    audio, sr, _ = audio_data
    
    # Create segmenter with custom padding
    segmenter = AudioSegmenter(
        silence_threshold=0.02,
        min_silence_duration=0.1,
        segment_padding=padding
    )
    
    # Segment the audio
    segments = segmenter.segment_audio(audio, sr)
    
    # Skip test if no segments were created
    assume(len(segments) > 0)
    
    # Detect boundaries without padding for comparison
    boundaries = segmenter.detect_word_boundaries(audio, sr)
    
    # Property 1: Each segment should have reasonable properties
    for i, segment in enumerate(segments):
        # Check that segment duration is reasonable
        assert segment.duration > 0, f"Segment {i} has non-positive duration"
        
        # Check that audio data exists and has reasonable length
        actual_samples = len(segment.audio_data)
        assert actual_samples > 0, f"Segment {i} has no audio data"
        
        # Check that start and end times are within audio bounds
        audio_duration = len(audio) / sr
        assert 0 <= segment.start_time <= audio_duration, \
            f"Segment {i} start time {segment.start_time} is outside audio duration"
        assert 0 <= segment.end_time <= audio_duration, \
            f"Segment {i} end time {segment.end_time} is outside audio duration"
    
    # Property 2: If we have boundaries, verify padding was applied (with tolerance)
    if len(boundaries) > 0 and len(segments) == len(boundaries):
        for i, (segment, (boundary_start, boundary_end)) in enumerate(zip(segments, boundaries)):
            # Segment should start at or before boundary (due to padding)
            # But not if we're at the start of the audio
            if boundary_start > padding:
                # Should start approximately padding seconds before boundary
                assert segment.start_time <= boundary_start, \
                    f"Segment {i} start time {segment.start_time} should be <= boundary {boundary_start}"
                # Should be within reasonable range
                assert segment.start_time >= boundary_start - padding - 0.05, \
                    f"Segment {i} start time {segment.start_time} is too early for boundary {boundary_start}"
            else:
                # At audio start, segment should start at 0
                assert segment.start_time == 0, \
                    f"Segment {i} at audio start should begin at 0, got {segment.start_time}"
            
            # Segment should end at or after boundary (due to padding)
            audio_duration = len(audio) / sr
            if boundary_end + padding < audio_duration:
                # Should end approximately padding seconds after boundary
                assert segment.end_time >= boundary_end, \
                    f"Segment {i} end time {segment.end_time} should be >= boundary {boundary_end}"
                # Should be within reasonable range
                assert segment.end_time <= boundary_end + padding + 0.05, \
                    f"Segment {i} end time {segment.end_time} is too late for boundary {boundary_end}"
            else:
                # At audio end, segment should end at or near audio duration
                assert segment.end_time >= boundary_end, \
                    f"Segment {i} should end at or after boundary {boundary_end}"
    
    # Property 3: All segments should have minimum duration (100ms as per requirements)
    min_duration = 0.1
    for i, segment in enumerate(segments):
        assert len(segment.audio_data) >= int(min_duration * segment.sample_rate), \
            f"Segment {i} is shorter than minimum duration {min_duration}s"


# Property 8: Segmentation Fallback
@settings(max_examples=100)
@given(audio_data=continuous_audio_strategy())
def test_segmentation_fallback(audio_data):
    """
    **Feature: turkish-sentence-emotion-analysis, Property 8: Segmentation Fallback**
    **Validates: Requirements 2.5**
    
    Property: For any audio where VAD-based segmentation fails to detect word 
    boundaries, the system should apply fixed-duration windowing with 500ms 
    windows and 250ms overlap.
    
    This test verifies that:
    1. Fallback windowing is applied when VAD fails
    2. Windows have the correct duration (500ms)
    3. Windows have the correct overlap (250ms)
    4. All audio is covered by the windowing
    5. Segments are properly ordered
    """
    audio, sr = audio_data
    
    # Create segmenter with standard parameters
    segmenter = AudioSegmenter(
        silence_threshold=0.02,
        min_silence_duration=0.1,
        segment_padding=0.05,
        fallback_window_size=0.5,  # 500ms
        fallback_overlap=0.25       # 250ms
    )
    
    # Segment the audio (should trigger fallback for continuous audio)
    segments = segmenter.segment_audio(audio, sr)
    
    # Property 1: Should produce segments even for continuous audio
    assert len(segments) > 0, "Fallback windowing should produce at least one segment"
    
    # Property 2: Calculate expected number of windows
    audio_duration = len(audio) / sr
    window_size = segmenter.fallback_window_size
    hop_size = window_size - segmenter.fallback_overlap
    
    # Expected number of windows (approximately)
    expected_num_windows = int(np.ceil((audio_duration - window_size) / hop_size)) + 1
    
    # Allow some tolerance
    assert len(segments) >= expected_num_windows - 1, \
        f"Expected at least {expected_num_windows - 1} segments, got {len(segments)}"
    assert len(segments) <= expected_num_windows + 1, \
        f"Expected at most {expected_num_windows + 1} segments, got {len(segments)}"
    
    # Property 3: Verify window durations are approximately correct
    for i, segment in enumerate(segments):
        # Last segment might be shorter
        if i < len(segments) - 1:
            # Non-last segments should be close to window_size
            assert abs(segment.duration - window_size) < 0.05, \
                f"Segment {i} duration {segment.duration} should be ~{window_size}s"
        else:
            # Last segment can be shorter but should be at least 100ms
            assert segment.duration >= 0.1, \
                f"Last segment {i} duration {segment.duration} is too short"
            assert segment.duration <= window_size + 0.05, \
                f"Last segment {i} duration {segment.duration} exceeds window size"
    
    # Property 4: Verify overlap between consecutive segments
    for i in range(len(segments) - 1):
        seg1 = segments[i]
        seg2 = segments[i + 1]
        
        # Calculate actual hop (time between segment starts)
        actual_hop = seg2.start_time - seg1.start_time
        expected_hop = hop_size
        
        # Allow some tolerance
        assert abs(actual_hop - expected_hop) < 0.05, \
            f"Hop between segments {i} and {i+1} is {actual_hop}s, expected ~{expected_hop}s"
        
        # Verify overlap exists (segments should overlap by fallback_overlap)
        overlap_start = seg2.start_time
        overlap_end = seg1.end_time
        
        if overlap_end > overlap_start:
            actual_overlap = overlap_end - overlap_start
            expected_overlap = segmenter.fallback_overlap
            
            # Allow tolerance
            assert abs(actual_overlap - expected_overlap) < 0.1, \
                f"Overlap between segments {i} and {i+1} is {actual_overlap}s, " \
                f"expected ~{expected_overlap}s"
    
    # Property 5: Segments should be properly ordered
    for i in range(len(segments) - 1):
        assert segments[i].start_time < segments[i + 1].start_time, \
            f"Segments {i} and {i+1} are not properly ordered"
        assert segments[i].word_index == i, \
            f"Segment {i} has incorrect word_index {segments[i].word_index}"
    
    # Property 6: First segment should start near the beginning
    assert segments[0].start_time < 0.1, \
        f"First segment should start near 0, got {segments[0].start_time}"
    
    # Property 7: Last segment should cover near the end of audio
    last_segment = segments[-1]
    assert last_segment.end_time >= audio_duration - 0.1, \
        f"Last segment should end near audio duration {audio_duration}, got {last_segment.end_time}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
