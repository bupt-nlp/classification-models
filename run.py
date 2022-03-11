
from trainer import Trainer
from src.processors.base_processor import ChnsenticorpDataProcessor
from src.processors.clinc150 import Clinc150DataProcessor
from src.models.simple_classifier import SimpleConfig, SimpleClassifier


if __name__ == "__main__":
    config_file = './configs/base.txt'
    config: SimpleConfig = SimpleConfig(config_file).parse_args(known_only=True)
    config.save('./configs/base.json')

    processor = ChnsenticorpDataProcessor()
    classifier = SimpleClassifier(config)
    trainer = Trainer(config=config, processor=processor, classifier=classifier)
    trainer.train()