"""Simple Classifier: Ernie + Linear"""
# from paddlenlp.transformers.ernie.modeling import ErnieForSequenceClassification, ErnieModel
from paddlenlp.transformers.bert.modeling import BertForSequenceClassification, BertModel
from src.config import Config

class SimpleConfig(Config):
    dropout: float = 0.3


class SimpleClassifier(BertForSequenceClassification):
    """Ernie For Sequence Classification"""

    def __init__(self, config: SimpleConfig):
        model = BertModel.from_pretrained(config.pretrained_model)
        super().__init__(model, config.num_labels, config.dropout)
