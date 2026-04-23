import torch
import torch.nn.functional as F
import numpy as np
import librosa
from transformers import AutoModelForAudioClassification


class WavLMEmotionPredictor:
    MODEL_NAME = "3loi/SER-Odyssey-Baseline-WavLM-Categorical"

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading WavLM model on {self.device}")
        self.model = AutoModelForAudioClassification.from_pretrained(
            self.MODEL_NAME,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        self.mean = self.model.config.mean
        self.std = self.model.config.std
        self.sampling_rate = self.model.config.sampling_rate
        self.id2label = self.model.config.id2label

    def predict(self, audio_path: str) -> dict:
        raw_wav, _ = librosa.load(audio_path, sr=self.sampling_rate)
        norm_wav = (raw_wav - self.mean) / (self.std + 1e-6)

        wavs = torch.tensor(norm_wav, dtype=torch.float32).unsqueeze(0).to(self.device)
        mask = torch.ones(1, len(norm_wav)).to(self.device)

        with torch.no_grad():
            logits = self.model(wavs, mask)

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
