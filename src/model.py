from transformers import AutoModel, AutoConfig, PreTrainedModel
import torch
import torch.nn as nn
from labels import LABEL2ID, ID2LABEL


class TokenClassifierConfig(AutoConfig):
    model_type = "token_classifier"


class TokenClassifier(PreTrainedModel):
    config_class = TokenClassifierConfig

    def __init__(self, config):
        super().__init__(config)

        model_name = config.model_name

        self.encoder = AutoModel.from_pretrained(model_name)

        hidden = self.encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, config.num_labels)
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        seq_hidden = out.last_hidden_state
        logits = self.classifier(seq_hidden)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.config.num_labels),
                labels.view(-1)
            )

        return {"logits": logits, "loss": loss}


def create_model(model_name):
    config = TokenClassifierConfig.from_pretrained(
        model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    config.model_name = model_name
    return TokenClassifier(config)
