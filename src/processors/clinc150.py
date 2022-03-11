"""DataProcessor for CLinc150 Corpus"""
from __future__ import annotations
import json
import os
from typing import List

from .base_processor import DataProcessor
from src.data import ExampleDataset, InputExample


class Clinc150DataProcessor(DataProcessor):
    """Clinc150 Data Processor"""
    def __init__(self, dataset_dir: str = 'data/clinc150') -> None:
        self.dataset_dir = dataset_dir

    def _read(self, split: str) -> List[InputExample]:
        with open(os.path.join(self.dataset_dir, f'{split}.json'), 'r', encoding='utf-8') as file_handler:
            data = json.load(file_handler)
            examples = data['data']
            examples = [InputExample(text=text, label=label) for text, label in examples]
        return examples
    
    def _get_dataset(self, split: str) -> ExampleDataset:
        examples = self._read(split)
        return ExampleDataset(examples)
        
    def get_train_dataset(self) -> ExampleDataset:
        return self._get_dataset('train')
    
    def get_dev_dataset(self) -> ExampleDataset:
        return self._get_dataset('dev')
    
    def get_test_dataset(self) -> ExampleDataset:
        return self._get_dataset('test')
    
    def get_labels(self) -> List[str]:
        labels = []
        with open(os.path.join(self.dataset_dir, 'data/clinc150/domain.json'), 'r', encoding='utf-8') as file_handler:
            data = json.load(file_handler)
            for domain_labels in data.values():
                labels.extend(domain_labels)
        return labels
