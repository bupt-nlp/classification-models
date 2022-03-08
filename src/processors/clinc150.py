"""DataProcessor for CLinc150 Corpus"""
from __future__ import annotations
import json
import os
from typing import List

from paddlenlp.datasets import MapDataset

from .base_processor import DataProcessor


class Clinc150DataProcessor(DataProcessor):
    def __init__(self, dataset_dir: str = 'data/clinc150') -> None:
        self.dataset_dir = dataset_dir

    def _read(self, split: str) -> List[dict]:
        examples = []
        with open(os.path.join(self.dataset_dir, f'{split}.json'), 'r', encoding='utf-8') as f:
            data = json.load(f)
            examples = data['data']
            examples = [dict(text=text, label=label) for text, label in examples]
        return examples

    def get_train_dataset(self) -> MapDataset:
        examples = self._read('train')
        labels = set([example['label'] for example in examples])
        return MapDataset(examples, label_list=labels)
    
    def get_dev_dataset(self) -> MapDataset:
        examples = self._read('dev')
        labels = set([example['label'] for example in examples])
        return MapDataset(examples, label_list=labels)
    
    def get_train_dataset(self) -> MapDataset:
        examples = self._read('test')
        labels = set([example['label'] for example in examples])
        return MapDataset(examples, label_list=labels)
    
    def get_labels(self) -> List[str]:
        labels = []
        with open(os.path.join(self.dataset_dir, 'data/clinc150/domain.json'), 'r', encoding='utf-8') as f:
            data = json.load(f)
            for domain_labels in data.values():
                labels.extend(domain_labels)
        return labels
