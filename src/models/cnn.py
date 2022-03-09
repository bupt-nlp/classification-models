from __future__ import annotations
from typing import List

from paddle.nn import Layer
import paddle.nn as nn
from paddlenlp.seq2vec.encoder import CNNEncoder
from paddlenlp.transformers.ernie.modeling import ErnieModel

from src.config import Config
from .utils import get_activation


class CNNConfig(Config):
    num_filter: int = 2
    filter_sizes: List[int] = [2, 3, 4, 5]
    activation: str = 'tanh'
    dropout: float = 0.3


class CNNClassifier(Layer):
    def __init__(self, config: CNNConfig,  name_scope=None, dtype="float32"):
        super().__init__(name_scope, dtype)
        self.pretrained_model = ErnieModel.from_pretrained(config.pretrained_model)
        self.encoder = CNNEncoder(
            emb_dim=768,
            num_filter=config.num_filter,
            ngram_filter_sizes=config.filter_sizes,
            conv_layer_activation=get_activation(config.activation)
        )
        self.dropout = nn.Dropout(p=config.dropout)

        self.fc = nn.Linear(
            in_features=self.encoder.get_output_dim(),
            out_features=config.num_labels
        )
    
    def forward(self, input_ids, token_type_ids):
        _, embedding = self.pretrained_model(input_ids=input_ids, token_type_ids=token_type_ids)
        embedding = self.dropout(self.encoder(embedding))
        logit = self.fc(embedding)
        return logit
    