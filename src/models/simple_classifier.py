"""Simple Classifier: Ernie + Linear"""
from paddlenlp.transformers.ernie.modeling import ErnieForSequenceClassification, ErnieModel
from src.config import Config

class SimpleConfig(Config):
    dropout: float = 0.3


class SimpleClassifier(ErnieForSequenceClassification):
    """Ernie For Sequence Classification"""

    def __init__(self, config: SimpleConfig):
        ernie = ErnieModel.from_pretrained(config.pretrained_model)
        super().__init__(ernie, config.num_labels, config.dropout)
