import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class Wav2Vec2GrammarScoring(nn.Module):
    def __init__(self, pretrained_model_name="facebook/wav2vec2-base-960h"):
        super(Wav2Vec2GrammarScoring, self).__init__()

        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_model_name)

        hidden_size = self.wav2vec2.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_values, attention_mask):
        outputs = self.wav2vec2(input_values, attention_mask)

        hidden_states = outputs.last_hidden_state
        pooled_output = torch.mean(hidden_states, dim=1)

        score = 5 * self.classifier(pooled_output)

        return score

    def freeze_feature_encoder(self):
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False

    def unfreeze_feature_encoder(self):
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = True

    def freeze_base_model(self):
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    def unfreeze_transformer_layers(self, num_layers=4):
        self.freeze_base_model()
        for i in range(
            len(self.wav2vec2.encoder.layers) - num_layers,
            len(self.wav2vec2.encoder.layers),
        ):
            for param in self.wav2vec2.encoder.layers[i].parameters():
                param.requires_grad = True