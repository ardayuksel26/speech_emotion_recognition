"""
Audio Segmentation Module

Segments sentence audio into individual word segments using Voice Activity Detection (VAD).
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import librosa


@dataclass
class AudioSegment:
    """Represents a single word segment"""
    audio_data: np.ndarray
    start_time: float
    end_time: float
    duration: float
    sample_rate: int
    word_index: int


class AudioSegmenter:
    """Segments audio into word-level chunks using VAD"""
    
    def __init__(
        self,
        silence_threshold: float = 0.02,
        min_silence_duration: float = 0.1,
        segment_padding: float = 0.05,
        fallback_window_size: float = 0.5,
        fallback_overlap: float = 0.25
    ):
        """
        Initialize AudioSegmenter
        
        Args:
            silence_threshold: Energy threshold for silence detection (default: 0.02)
            min_silence_duration: Minimum silence duration in seconds (default: 0.1s = 100ms)
            segment_padding: Padding to add on both sides in seconds (default: 0.05s = 50ms)
            fallback_window_size: Window size for fallback windowing (default: 0.5s = 500ms)
            fallback_overlap: Overlap for fallback windowing (default: 0.25s = 250ms)
        """
        self.silence_threshold = silence_threshold
        self.min_silence_duration = min_silence_duration
        self.segment_padding = segment_padding
        self.fallback_window_size = fallback_window_size
        self.fallback_overlap = fallback_overlap
    
    def segment_audio(self, audio: np.ndarray, sr: int) -> List[AudioSegment]:
        """
        Segment audio into word-level chunks
        
        Args:
            audio: Audio signal as numpy array
            sr: Sample rate
            
        Returns:
            List of AudioSegment objects
        """
        # Try VAD-based segmentation first
        boundaries = self.detect_word_boundaries(audio, sr)
        
        # If VAD fails (no boundaries or too few), use fallback windowing
        if len(boundaries) < 2:
            return self._apply_fallback_windowing(audio, sr)
        
        # Extract segments from boundaries
        segments = []
        for idx, (start, end) in enumerate(boundaries):
            segment_audio = self.extract_segment(audio, start, end, sr)
            
            segments.append(AudioSegment(
                audio_data=segment_audio,
                start_time=start,
                end_time=end,
                duration=end - start,
                sample_rate=sr,
                word_index=idx
            ))
        
        return segments
    
    def detect_word_boundaries(self, audio: np.ndarray, sr: int) -> List[Tuple[float, float]]:
        """
        Detect word boundaries using Voice Activity Detection
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            List of (start_time, end_time) tuples in seconds
        """
        # Apply VAD to get voice activity mask
        vad_mask = self.apply_vad(audio, sr, self.silence_threshold)
        
        # Convert mask to time boundaries
        boundaries = []
        in_speech = False
        start_idx = 0
        
        min_silence_samples = int(self.min_silence_duration * sr)
        silence_counter = 0
        
        for idx, is_voice in enumerate(vad_mask):
            if is_voice:
                if not in_speech:
                    # Start of speech segment
                    start_idx = idx
                    in_speech = True
                silence_counter = 0
            else:
                if in_speech:
                    silence_counter += 1
                    # Check if silence duration exceeds threshold
                    if silence_counter >= min_silence_samples:
                        # End of speech segment
                        end_idx = idx - silence_counter
                        start_time = start_idx / sr
                        end_time = end_idx / sr
                        
                        if end_time > start_time:  # Valid segment
                            boundaries.append((start_time, end_time))
                        
                        in_speech = False
                        silence_counter = 0
        
        # Handle case where audio ends while in speech
        if in_speech:
            end_time = len(audio) / sr
            start_time = start_idx / sr
            if end_time > start_time:
                boundaries.append((start_time, end_time))
        
        return boundaries
    
    def apply_vad(self, audio: np.ndarray, sr: int, threshold: float = 0.02) -> np.ndarray:
        """
        Apply Voice Activity Detection using energy-based thresholding
        
        Args:
            audio: Audio signal
            sr: Sample rate
            threshold: Energy threshold for voice detection
            
        Returns:
            Boolean mask where True indicates voice activity
        """
        # Calculate RMS energy in frames
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop
        
        rms = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
        
        # Normalize RMS
        if np.max(rms) > 0:
            rms = rms / np.max(rms)
        
        # Apply threshold
        vad_frames = rms > threshold
        
        # Expand frame-level decisions to sample-level
        vad_mask = np.repeat(vad_frames, hop_length)
        
        # Ensure mask matches audio length
        if len(vad_mask) < len(audio):
            vad_mask = np.pad(vad_mask, (0, len(audio) - len(vad_mask)), mode='edge')
        elif len(vad_mask) > len(audio):
            vad_mask = vad_mask[:len(audio)]
        
        return vad_mask
    
    def extract_segment(
        self,
        audio: np.ndarray,
        start: float,
        end: float,
        sr: int,
        padding: float = None
    ) -> np.ndarray:
        """
        Extract audio segment with padding
        
        Args:
            audio: Full audio signal
            start: Start time in seconds
            end: End time in seconds
            sr: Sample rate
            padding: Padding in seconds (uses self.segment_padding if None)
            
        Returns:
            Extracted audio segment
        """
        if padding is None:
            padding = self.segment_padding
        
        # Convert times to sample indices
        start_idx = max(0, int((start - padding) * sr))
        end_idx = min(len(audio), int((end + padding) * sr))
        
        segment = audio[start_idx:end_idx]
        
        # Ensure minimum duration (100ms as per requirements)
        min_samples = int(0.1 * sr)
        if len(segment) < min_samples:
            # Pad with silence
            padding_needed = min_samples - len(segment)
            segment = np.pad(segment, (0, padding_needed), mode='constant')
        
        return segment
    
    def _apply_fallback_windowing(self, audio: np.ndarray, sr: int) -> List[AudioSegment]:
        """
        Apply fixed-duration windowing as fallback when VAD fails
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            List of AudioSegment objects
        """
        window_samples = int(self.fallback_window_size * sr)
        hop_samples = int((self.fallback_window_size - self.fallback_overlap) * sr)
        
        segments = []
        word_index = 0
        
        for start_sample in range(0, len(audio), hop_samples):
            end_sample = min(start_sample + window_samples, len(audio))
            
            # Skip if segment is too short
            if end_sample - start_sample < int(0.1 * sr):
                continue
            
            segment_audio = audio[start_sample:end_sample]
            start_time = start_sample / sr
            end_time = end_sample / sr
            
            segments.append(AudioSegment(
                audio_data=segment_audio,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                sample_rate=sr,
                word_index=word_index
            ))
            
            word_index += 1
            
            # Stop if we've reached the end
            if end_sample >= len(audio):
                break
        
        return segments
