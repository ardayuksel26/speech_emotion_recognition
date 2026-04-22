import torch
import torch.nn.functional as F
import numpy as np
import librosa
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers import AutoConfig, Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2PreTrainedModel, Wav2Vec2Model
from transformers.file_utils import ModelOutput
from torch import nn


@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class Wav2Vec2ClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)
        self.init_weights()

    def merged_strategy(self, hidden_states, mode="mean"):
        if mode == "mean":
            return torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            return torch.sum(hidden_states, dim=1)
        elif mode == "max":
            return torch.max(hidden_states, dim=1)[0]
        raise ValueError(f"Unknown pooling mode: {mode}")

    def forward(self, input_values, attention_mask=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, labels=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = self.merged_strategy(outputs[0], mode=self.pooling_mode)
        logits = self.classifier(hidden_states)
        if not return_dict:
            return (logits,) + outputs[2:]
        return SpeechClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Wav2Vec2TurkishPredictor:
    MODEL_NAME = "sefa-alper/wav2vec2-xlsr-turkish-speech-emotion-recognition-v3"

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Wav2Vec2 Turkish model on {self.device}")
        self.config = AutoConfig.from_pretrained(self.MODEL_NAME)
        self.processor = Wav2Vec2Processor.from_pretrained(self.MODEL_NAME)
        self.model = Wav2Vec2ForSpeechClassification.from_pretrained(self.MODEL_NAME).to(self.device)
        self.model.eval()
        self.id2label = self.config.id2label

    def predict(self, audio_path):
        speech, _ = librosa.load(audio_path, sr=16000)
        features = self.processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = features.input_values.to(self.device)
        attention_mask = features.attention_mask.to(self.device)

        with torch.no_grad():
            logits = self.model(input_values, attention_mask=attention_mask).logits

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
