"""
Quick integration test for the sentence analysis API

This script tests the complete API workflow:
1. Start the API server
2. Upload an audio file
3. Poll for results
4. Display the analysis
"""

import requests
import time
import numpy as np
from scipy.io import wavfile
import tempfile
import os


def create_test_audio(duration=2.0, sample_rate=16000):
    """Create a test WAV file"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Create a simple sine wave
    frequency = 440  # A4 note
    audio = np.sin(2 * np.pi * frequency * t) * 0.3
    audio_int16 = (audio * 32767).astype(np.int16)
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    wavfile.write(temp_file.name, sample_rate, audio_int16)
    temp_file.close()
    
    return temp_file.name


def test_api_integration():
    """Test the complete API workflow"""
    base_url = "http://localhost:8000"
    
    print("=" * 60)
    print("Turkish Sentence Emotion Analysis API - Integration Test")
    print("=" * 60)
    
    # Check if server is running
    try:
        response = requests.get(f"{base_url}/health", timeout=2)
        if response.status_code != 200:
            print("❌ Server is not healthy")
            return False
        print("✓ Server is healthy")
        health_data = response.json()
        print(f"  Queue size: {health_data['queue_size']}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server: {e}")
        print("\nPlease start the server first:")
        print("  python Backend/main_sentence_analysis.py")
        return False
    
    # Create test audio
    print("\n1. Creating test audio file...")
    audio_path = create_test_audio(duration=2.0, sample_rate=16000)
    print(f"✓ Created: {audio_path}")
    
    try:
        # Upload audio
        print("\n2. Uploading audio for analysis...")
        with open(audio_path, 'rb') as f:
            files = {"file": ("test.wav", f, "audio/wav")}
            data = {"aggregation_strategy": "weighted_average"}
            response = requests.post(f"{base_url}/api/analyze-sentence", files=files, data=data)
        
        if response.status_code != 200:
            print(f"❌ Upload failed: {response.status_code}")
            print(response.json())
            return False
        
        job_data = response.json()
        job_id = job_data["job_id"]
        print(f"✓ Job created: {job_id}")
        
        # Poll for results
        print("\n3. Waiting for analysis to complete...")
        max_attempts = 30
        attempt = 0
        
        while attempt < max_attempts:
            response = requests.get(f"{base_url}/api/status/{job_id}")
            if response.status_code != 200:
                print(f"❌ Status check failed: {response.status_code}")
                return False
            
            status_data = response.json()
            status = status_data["status"]
            progress = status_data["progress"]
            
            print(f"  Status: {status} ({progress*100:.0f}%)", end="\r")
            
            if status == "completed":
                print("\n✓ Analysis completed!")
                
                # Display results
                result = status_data["result"]
                print("\n" + "=" * 60)
                print("RESULTS")
                print("=" * 60)
                print(f"Primary Emotion: {result['primary_emotion'].upper()}")
                print(f"Confidence: {result['confidence']:.2%}")
                print(f"Mixed Emotions: {'Yes' if result['is_mixed'] else 'No'}")
                
                print("\nProbability Distribution:")
                for emotion, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
                    bar = "█" * int(prob * 50)
                    print(f"  {emotion:8s}: {bar} {prob:.2%}")
                
                print("\nWord-by-Word Analysis:")
                for word in result['word_predictions']:
                    uncertain = " (uncertain)" if word['is_uncertain'] else ""
                    print(f"  Word {word['word_index']}: {word['emotion']:8s} "
                          f"({word['confidence']:.2%}){uncertain} "
                          f"[{word['start_time']:.2f}s - {word['end_time']:.2f}s]")
                
                print("\nMetadata:")
                metadata = result['metadata']
                print(f"  Number of words: {metadata['num_words']}")
                print(f"  Audio duration: {metadata['audio_duration']:.2f}s")
                print(f"  Sample rate: {metadata['sample_rate']} Hz")
                print(f"  Aggregation strategy: {metadata['aggregation_strategy']}")
                print(f"  Processing time: {result['processing_time']:.2f}s")
                
                print("=" * 60)
                print("✓ Integration test PASSED")
                return True
            
            elif status in ["failed", "timeout"]:
                print(f"\n❌ Analysis failed: {status_data.get('error', 'Unknown error')}")
                return False
            
            time.sleep(1)
            attempt += 1
        
        print("\n❌ Timeout waiting for results")
        return False
        
    finally:
        # Clean up
        if os.path.exists(audio_path):
            os.remove(audio_path)
            print(f"\n✓ Cleaned up test file")


if __name__ == "__main__":
    success = test_api_integration()
    exit(0 if success else 1)

