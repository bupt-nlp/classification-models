from __future__ import annotations
from typing import List, Optional
from matplotlib.pyplot import axis

import paddle
from paddle.nn import Layer
import paddle.nn as nn
from paddlenlp.seq2vec.encoder import RNNEncoder 
from paddlenlp.transformers.model_utils import PretrainedModel
from paddlenlp.transformers.ernie.modeling import ErnieModel

from src.config import Config
from .utils import get_activation


class RNNConfig(Config):
    hidden_size: int = 100
    num_layers: int = 2
    bidirection: bool = True
    reduction: str = 'mean'
    dropout: float = 0.3
    pooling_type: Optional[str] = None

class RNNClassifier(Layer):
    def __init__(self, config: RNNConfig,  name_scope=None, dtype="float32"):
        super().__init__(name_scope, dtype)
        self.pretrained_model = ErnieModel.from_pretrained(config.pretrained_model)
        self.encoder = RNNEncoder(
            input_size=768, 
            hidden_size=config.hidden_size, 
            num_layers=config.num_layers, 
            direction='bidirect' if  config.bidirection else 'forward',
            dropout=config.dropout,
            pooling_type=config.pooling_type
        )
        self.dropout = nn.Dropout(p=config.dropout)

        self.fc = nn.Linear(
            in_features=self.encoder.get_output_dim(),
            out_features=config.num_labels
        )
        self.config = config
    
    def forward(self, input_ids, token_type_ids):
        embedding, _ = self.pretrained_model(input_ids=input_ids, token_type_ids=token_type_ids)
        embedding = self.dropout(self.encoder(embedding, self.config.max_seq_length))
        
        if self.config.reduction == 'mean':
            embedding = paddle.mean(embedding, axis=-1)

        logit = self.fc(embedding)
        return logit
    