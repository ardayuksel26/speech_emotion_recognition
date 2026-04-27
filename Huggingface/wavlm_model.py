import torch
import torch.nn.functional as F
import numpy as np
import librosa
from transformers import AutoModelForAudioClassification


class WavLMBasePlusEmotionPredictor:
    MODEL_NAME = "jihedjabnoun/wavlm-base-emotion"

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading WavLM Base Plus model on {self.device}")
        from transformers import AutoFeatureExtractor
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForAudioClassification.from_pretrained(self.MODEL_NAME).to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def predict(self, audio_path: str) -> dict:
        speech, _ = librosa.load(audio_path, sr=16000)
        inputs = self.feature_extractor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
        best_idx = int(np.argmax(scores))

        results = [
            {"label": self.id2label[i].lower(), "score": float(s)}
            for i, s in enumerate(scores)
        ]
        return {
            "emotion": self.id2label[best_idx].lower(),
            "scores": results,
        }


if __name__ == "__main__":
    pass
