import sys
import types
import torch
import torch.nn.functional as F
import numpy as np
import librosa

# transformers.deepspeed yeni versiyonlarda kaldırıldı,
# ExHuBERT'in custom kodu buna bağımlı — stub ile patch'leyelim
if 'transformers.deepspeed' not in sys.modules:
    import transformers as _tf
    _fake_ds = types.ModuleType('transformers.deepspeed')
    _fake_ds.is_deepspeed_zero3_enabled = lambda: False
    _fake_ds.deepspeed_config = lambda: {}
    sys.modules['transformers.deepspeed'] = _fake_ds
    _tf.deepspeed = _fake_ds

from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor


class ExHuBERTEmotionPredictor:
    MODEL_NAME = "amiriparian/ExHuBERT"
    FEATURE_EXTRACTOR_NAME = "facebook/hubert-base-ls960"
    REVISION = "b158d45ed8578432468f3ab8d46cbe5974380812"
    MAX_LENGTH = 48000  # 3 saniye @ 16kHz

    # Arousal-Valence boyutlarına göre 6 sınıf
    # Low Arousal: disgust(neg), neutral(neut), kind(pos)
    # High Arousal: anger(neg), surprise(neut), joy(pos)
    ID2LABEL = {
        0: "disgust",
        1: "neutral",
        2: "kind",
        3: "angry",
        4: "surprised",
        5: "happy",
    }

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading ExHuBERT model on {self.device}")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.FEATURE_EXTRACTOR_NAME
        )
        self.model = AutoModelForAudioClassification.from_pretrained(
            self.MODEL_NAME,
            trust_remote_code=True,
            revision=self.REVISION,
        ).to(self.device)
        self.model.eval()

    def predict(self, audio_path: str) -> dict:
        speech, _ = librosa.load(audio_path, sr=16000)

        inputs = self.feature_extractor(
            speech,
            sampling_rate=16000,
            padding="max_length",
            max_length=self.MAX_LENGTH,
            return_tensors="pt",
        )
        input_values = inputs["input_values"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_values).logits

        scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
        best_idx = int(np.argmax(scores))

        results = [
            {"label": self.ID2LABEL[i], "score": float(s)}
            for i, s in enumerate(scores)
        ]
        return {
            "emotion": self.ID2LABEL[best_idx],
            "scores": results,
        }


if __name__ == "__main__":
    pass
