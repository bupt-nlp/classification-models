"""Base Data Processors"""
from __future__ import annotations
from typing import List, Any

import numpy as np
from abc import ABC
import paddle
from paddlenlp.transformers.tokenizer_utils import PretrainedTokenizer
from paddlenlp.datasets import MapDataset, load_dataset


class DataProcessor(ABC):
    """Abstract Data Processor Class which handle the different corpus"""    
    def get_train_dataset(self) -> MapDataset:
        """get_train_dataset
        """
        raise NotImplementedError

    def get_test_dataset(self) -> MapDataset:
        raise NotImplementedError

    def get_dev_dataset(self) -> MapDataset:
        raise NotImplementedError
    
    def get_labels() -> List[str]:
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
    
    def get_train_dataset(self):
        dataset = load_dataset(self.corpus_name, splits=['train'])
        self.train_labels = dataset.label_list
        return dataset

    def get_dev_dataset(self):
        dataset = load_dataset(self.corpus_name, splits=['dev'])
        self.dev_labels = dataset.label_list

        return dataset

    def get_test_dataset(self):
        dataset = load_dataset(self.corpus_name, splits=['test'])
        self.test_labels = dataset.label_list
        return dataset
    
    def get_labels(self) -> List[str]:
        return self.train_labels


class XnliCNDatProcessor(DataProcessor):
    corpus_name: str = 'xnli_cn'

    def __init__(self) -> None:
        super().__init__()
        self.train_labels = []
        self.dev_labels = []
        self.test_labels = []
    
    def get_train_dataset(self):
        dataset = load_dataset(XnliCNDatProcessor.corpus_name, splits=['train'])
        self.train_labels = dataset.label_list
        return dataset

    def get_dev_dataset(self):
        dataset = load_dataset(XnliCNDatProcessor.corpus_name, splits=['dev'])
        self.dev_labels = dataset.label_list

        return dataset

    def get_test_dataset(self):
        dataset = load_dataset(XnliCNDatProcessor.corpus_name, splits=['test'])
        self.test_labels = dataset.label_list
        return dataset

    def get_labels(self) -> List[str]:
        return self.train_labels
