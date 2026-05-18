import re
import torch


class SenseVoiceEmotionPredictor:
    MODEL_ID = "FunAudioLLM/SenseVoiceSmall"

    # All emotion tags the model can output
    EMOTION_TAGS = {"HAPPY", "SAD", "ANGRY", "NEUTRAL", "FEARFUL", "DISGUSTED", "SURPRISED"}

    # Map to display-friendly lowercase keys
    LABEL_MAP = {
        "HAPPY":     "happy",
        "SAD":       "sad",
        "ANGRY":     "angry",
        "NEUTRAL":   "calm",
        "FEARFUL":   "fearful",
        "DISGUSTED": "disgusted",
        "SURPRISED": "surprised",
    }

    def __init__(self):
        import os
        # Windows'ta HuggingFace symlink hatasını önle
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

        # Windows Application Control politikası funasr'ın pip subprocess çağrısını
        # engelliyor ([WinError 4551]). Gereksinimler zaten kurulu olduğu için
        # bu adımı monkey-patch ile atlıyoruz.
        try:
            import funasr.utils.install_model_requirements as _imr
            _imr.install_requirements = lambda *a, **kw: None
        except Exception:
            pass

        from funasr import AutoModel
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SenseVoice model on {device}")
        self._model = AutoModel(
            model=self.MODEL_ID,
            device=device,
            hub="hf",
            trust_remote_code=True,
        )

    def predict(self, audio_path: str) -> dict:
        res = self._model.generate(
            input=audio_path,
            cache={},
            language="auto",
            use_itn=False,
            ban_emo_unk=True,
            batch_size_s=60,
        )

        raw_text = res[0]["text"] if res else ""
        print(f"[SenseVoice] raw_text: {repr(raw_text)}")

        # Tags are uppercase: <|HAPPY|>, <|NEUTRAL|>, etc.
        # Case-insensitive search for robustness
        pattern = r"<\|(" + "|".join(sorted(self.EMOTION_TAGS)) + r")\|>"
        match = re.search(pattern, raw_text, re.IGNORECASE)
        emotion_raw = match.group(1).upper() if match else "NEUTRAL"
        emotion = self.LABEL_MAP.get(emotion_raw, "calm")

        all_labels = list(self.LABEL_MAP.values())
        scores = [
            {"label": lbl, "score": 1.0 if lbl == emotion else 0.0}
            for lbl in all_labels
        ]

        return {
            "emotion": emotion,
            "scores": scores,
            "raw_text": raw_text,
        }


if __name__ == "__main__":
    pass
