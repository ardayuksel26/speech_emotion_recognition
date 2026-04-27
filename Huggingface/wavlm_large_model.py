import torch
import torch.nn.functional as F
import numpy as np
import librosa
from transformers import AutoModelForAudioClassification

# {0: 'Angry', 1: 'Sad', 2: 'Happy', 3: 'Surprise',
#  4: 'Fear',  5: 'Disgust', 6: 'Contempt', 7: 'Neutral'}
# MSP-Podcast Odyssey 2024 Challenge — WavLM backbone, 0.3B params


class WavLMLargeEmotionPredictor:
    MODEL_NAME = "3loi/SER-Odyssey-Baseline-WavLM-Categorical"

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading WavLM Large emotion model on {self.device}")

        # Pre-download to HF cache so we have a local path the model code can use
        from huggingface_hub import snapshot_download as _hf_dl
        local_path = _hf_dl(self.MODEL_NAME)

        # The model's trust_remote_code calls modelscope SDK to download weights.
        # Patch modelscope's snapshot_download (at every likely import path)
        # to return our already-downloaded local HF cache path.
        _ms_patches = {}
        for _mod_name in (
            'modelscope.hub.snapshot_download',
            'modelscope',
        ):
            try:
                import importlib
                _mod = importlib.import_module(_mod_name)
                if hasattr(_mod, 'snapshot_download'):
                    _ms_patches[_mod_name] = (_mod, getattr(_mod, 'snapshot_download'))
                    setattr(_mod, 'snapshot_download', lambda *a, **kw: local_path)
            except Exception:
                pass

        # Protect against trust_remote_code leaving accelerate's init_empty_weights
        # context active, which would cause subsequent models' parameters to land on
        # the meta device and raise "Cannot copy out of meta tensor".
        import torch.nn as _nn
        _orig_reg_param = _nn.Module.register_parameter
        _orig_reg_buf = _nn.Module.register_buffer
        try:
            self.model = AutoModelForAudioClassification.from_pretrained(
                local_path, trust_remote_code=True
            ).to(self.device)
        finally:
            _nn.Module.register_parameter = _orig_reg_param
            _nn.Module.register_buffer = _orig_reg_buf
            for _mod_name, (_mod, _orig_fn) in _ms_patches.items():
                setattr(_mod, 'snapshot_download', _orig_fn)

        self.mean = self.model.config.mean
        self.std = self.model.config.std
        self.sr = self.model.config.sampling_rate
        self.id2label = self.model.config.id2label
        self.model.eval()

    def predict(self, audio_path: str) -> dict:
        raw_wav, _ = librosa.load(audio_path, sr=self.sr)

        # Normalize using model's own mean/std
        norm_wav = (raw_wav - self.mean) / (self.std + 1e-6)

        wavs = torch.tensor(norm_wav).unsqueeze(0).float().to(self.device)
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


if __name__ == "__main__":
    pass
