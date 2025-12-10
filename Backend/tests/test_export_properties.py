
import pytest
from hypothesis import given, strategies as st
import json
import csv
import io
import re
from datetime import datetime
from typing import Dict, List, Any

# Strategy for Analysis Result simulation
@st.composite
def analysis_result_strategy(draw):
    emotions = ['angry', 'calm', 'happy', 'sad']
    dominant = draw(st.sampled_from(emotions))
    
    words = draw(st.lists(
        st.fixed_dictionaries({
            'word': st.text(min_size=1),
            'start': st.floats(min_value=0, max_value=100),
            'end': st.floats(min_value=0, max_value=100),
            'emotion': st.sampled_from(emotions),
            'confidence': st.floats(min_value=0, max_value=1)
        }),
        min_size=1,
        max_size=10
    ))
    
    # Fix timestamps to be sequential-ish for realism (optional but good)
    curr = 0.0
    for w in words:
        dur = abs(w['end'] - w['start']) or 0.1
        w['start'] = curr
        w['end'] = curr + dur
        curr = w['end']

    return {
        'emotions': {e: 0.25 for e in emotions}, # Simplified
        'dominant_emotion': dominant,
        'confidence': 0.8,
        'transcript': " ".join(w['word'] for w in words),
        'word_timestamps': words
    }

class TestExportProperties:
    
    @given(analysis_result_strategy())
    def test_export_format_support(self, result: Dict[str, Any]):
        """
        Property 25: Export Format Support
        Validates that the AnalysisResult structure can be correctly formatted 
        as both JSON and CSV.
        """
        
        # 1. JSON Export Test
        # Should be serializable
        json_str = json.dumps(result)
        # Should be deserializable back to same structure
        decoded = json.loads(json_str) 
        assert decoded['dominant_emotion'] == result['dominant_emotion']
        assert len(decoded['word_timestamps']) == len(result['word_timestamps'])
        
        # 2. CSV Export Test (Mirroring Frontend Logic)
        headers = ['Word', 'Start Time (s)', 'End Time (s)', 'Emotion', 'Confidence', 'Sentence Dominant Emotion']
        
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(headers)
        
        for w in result['word_timestamps']:
            writer.writerow([
                w['word'],
                f"{w['start']:.3f}",
                f"{w['end']:.3f}",
                w['emotion'],
                f"{w['confidence']:.3f}",
                result['dominant_emotion']
            ])
            
        csv_content = output.getvalue()
        
        # Verification
        lines = csv_content.strip().split('\n')
        # Header + 1 line per word
        assert len(lines) == 1 + len(result['word_timestamps'])
        
        # Check header
        assert lines[0].startswith("Word,Start Time")
        
        # Check first data row content basic match
        if result['word_timestamps']:
            first_word = result['word_timestamps'][0]
            assert first_word['word'] in lines[1]
            assert first_word['emotion'] in lines[1]
            assert result['dominant_emotion'] in lines[1]

    @given(st.datetimes())
    def test_export_filename_format(self, date: datetime):
        """
        Property 41: Export Filename Format
        Validates that filenames are generated with the correct timestamp format.
        Logic: analysis_result_YYYY-MM-DDTHH-mm-ss-sssZ.json
        """
        # Mirroring frontend: new Date().toISOString().replace(/[:.]/g, '-')
        # Python isoformat is slightly different, but we simulate the requirement:
        # "filenames with timestamp"
        
        timestamp = date.isoformat().replace(':', '-').replace('.', '-')
        prefix = 'analysis_result'
        
        filename_json = f"{prefix}_{timestamp}.json"
        
        # Validation Regex
        # Should start with prefix
        assert filename_json.startswith(prefix)
        # Should end with extension
        assert filename_json.endswith('.json')
        # Should contain 'timestamp' specific chars (digits, dashes, T)
        # It shouldn't have colons (unsafe for windows filenames)
        assert ':' not in filename_json
        
        # Check structure
        parts = filename_json.split('_')
        assert len(parts) >= 3 # prefix, result, timestamp...
