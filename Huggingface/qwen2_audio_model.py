import torch
import librosa
import numpy as np
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import re

class Qwen2AudioEmotionPredictor:
    MODEL_NAME = "Qwen/Qwen2-Audio-7B-Instruct"
    
    # Emotion labels we are interested in
    EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Qwen2-Audio model: {self.MODEL_NAME} on {self.device}")
        
        # 4-bit quantization config to save VRAM (requires bitsandbytes and accelerate)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        self.processor = AutoProcessor.from_pretrained(self.MODEL_NAME)
        
        try:
            # Note: 7B model in 4-bit requires approx 5-6GB VRAM
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                self.MODEL_NAME,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Failed to load with quantization: {e}")
            print("Attempting to load without quantization (requires ~16GB VRAM)...")
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                self.MODEL_NAME,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
        self.model.eval()

    def predict(self, audio_path: str) -> dict:
        # Load and resample audio
        sampling_rate = self.processor.feature_extractor.sampling_rate
        audio, _ = librosa.load(audio_path, sr=sampling_rate)
        
        # Prepare the prompt
        prompt = f"What is the emotion of the speaker in this audio? Please choose from: {', '.join(self.EMOTIONS)}."
        
        conversation = [
            {"role": "user", "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": prompt},
            ]},
        ]
        
        # Preprocess inputs
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=text, audios=audio, return_tensors="pt", padding=True)
        
        # Move inputs to the same device as the model's first parameter
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            generate_ids = self.model.generate(**inputs, max_new_tokens=64)
            generate_ids = generate_ids[:, inputs["input_ids"].size(1):]
            response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        # Parse response to find the emotion
        response_lower = response.lower()
        predicted_emotion = "neutral" # Default
        for emo in self.EMOTIONS:
            if emo in response_lower:
                predicted_emotion = emo
                break
        
        # Since it's an LLM, we don't get direct probabilities easily for all classes without extra compute
        # We provide a mock scores list with 1.0 for the detected emotion
        scores = [
            {"label": emo, "score": 1.0 if emo == predicted_emotion else 0.0}
            for emo in self.EMOTIONS
        ]
        
        return {
            "emotion": predicted_emotion,
            "scores": scores,
            "raw_text": response
        }

if __name__ == "__main__":
    pass
