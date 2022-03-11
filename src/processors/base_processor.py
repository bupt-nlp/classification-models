"""Base Data Processors"""
from __future__ import annotations
from typing import List

from abc import ABC
from paddlenlp.datasets import MapDataset, load_dataset
from src.data import InputExample, ExampleDataset


class DataProcessor(ABC):
    """Abstract Data Processor Class which handle the different corpus"""    
    def get_train_dataset(self) -> ExampleDataset:
        """get_train_dataset
        """
        raise NotImplementedError

    def get_test_dataset(self) -> ExampleDataset:
        raise NotImplementedError

    def get_dev_dataset(self) -> ExampleDataset:
        raise NotImplementedError
    
    def get_labels(self) -> List[str]:
        pass

    def get_train_labels(self) -> List[str]:
        return self.get_labels()
    
    def get_test_labels(self) -> List[str]:
        return self.get_labels()
        
    def get_dev_labels(self) -> List[str]:
        return self.get_labels()

    
class ChnsenticorpDataProcessor(DataProcessor):
    """Chnsenticorp Data Processor"""

    corpus_name: str = 'chnsenticorp'
    
    def __init__(self) -> None:
        super().__init__()
        self.train_labels = []
        self.dev_labels = []
        self.test_labels = []

    def _to_examples(self, dataset) -> List[InputExample]:
        examples = []
        for data_item in dataset:
            examples.append(InputExample(
                text=data_item['text'],
                label=data_item['label']
            ))
        return examples
    
    def get_train_dataset(self):
        dataset = load_dataset(self.corpus_name, splits=['train'])
        examples = self._to_examples(dataset)
        return ExampleDataset(examples)

    def get_dev_dataset(self):
        dataset = load_dataset(self.corpus_name, splits=['dev'])
        examples = self._to_examples(dataset)
        return ExampleDataset(examples)

    def get_test_dataset(self):
        dataset = load_dataset(self.corpus_name, splits=['test'])
        examples = self._to_examples(dataset)
        return ExampleDataset(examples)
    
    def get_labels(self) -> List[str]:
        return self.train_labels


class XnliCNDatProcessor(DataProcessor):
    corpus_name: str = 'xnli_cn'

    def __init__(self) -> None:
        super().__init__()
        self.train_labels = []
        self.dev_labels = []
        self.test_labels = []

    def _to_examples(self, dataset) -> List[InputExample]:
        examples = []
        for data_item in dataset:
            examples.append(InputExample(
                text=data_item['text'],
                label=data_item['label']
            ))
        return examples
    
    def get_train_dataset(self):
        dataset = load_dataset(XnliCNDatProcessor.corpus_name, splits=['train'])
        examples = self._to_examples(dataset)
        return ExampleDataset(examples)

    def get_dev_dataset(self):
        dataset = load_dataset(XnliCNDatProcessor.corpus_name, splits=['dev'])
        examples = self._to_examples(dataset)
        return ExampleDataset(examples)

    def get_test_dataset(self):
        dataset = load_dataset(XnliCNDatProcessor.corpus_name, splits=['test'])
        examples = self._to_examples(dataset)
        return ExampleDataset(examples)

    def get_labels(self) -> List[str]:
        return self.train_labels
