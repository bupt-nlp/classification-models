"""
Paddle Prompt Learning - https://github.com/wj-Mcat/paddle-prompt

Authors:    Jingjing WU (吴京京) <https://github.com/wj-Mcat>


2022-now @ Copyright wj-Mcat

Licensed under the Apache License, Version 2.0 (the 'License');
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an 'AS IS' BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from collections import OrderedDict
import numpy as np

import paddle
from paddle.io import Dataset
from paddlenlp.transformers.tokenizer_utils import PretrainedTokenizer


@dataclass
class InputExample:
    """Input Example Data Structure for training data
    """
    text: str                                   # source sentence 
    label: Union[str, List[str]]                # label field

    guid: Optional[Union[int, str]] = None      # store the union id for example
    text_pair: Optional[str] = None             # for sentence pair task
    target_text: Optional[str] = None           # for generation task
    meta: Dict[str, Any] = field(default_factory=dict)  # store the meta data of training example


class ExampleDataset(Dataset):
    def __init__(self, examples: List[InputExample]):
        super().__init__()
        self.examples: List[InputExample] = examples
        self.label2idx: Dict[str, int] = OrderedDict()
        
        for example in examples:
            if example.label not in self.label2idx:
                self.label2idx[example.label] = len(self.label2idx)

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx: int):
        return self.examples[idx]


@dataclass
class InputFeature:
    """Input Feature which should be wrapped into PLMs
    """
    input_ids: List[int]
    token_type_ids: List[int]
    attention_mask: List[int]

    label_id: int


def convert_examples_to_features(examples: List[InputExample], tokenizer: PretrainedTokenizer, max_seq_len: int, label2idx: Dict[str, int]) -> List[InputFeature]:
    if not examples:
        return []

    # 1. convert label ids and text-pairs
    label_ids, texts = [], []
    for example in examples:
        if isinstance(example.label, list):
            label_ids.append([label2idx[label] for label in example.label])
        else:
            label_ids.append(label2idx[example.label])

        if example.text_pair:
            texts.append((example.text, example.text_pair))
        else:
            texts.append(example.text)

    encoded_feature = tokenizer.batch_encode(
        batch_text_or_text_pairs=texts,
        max_seq_len=max_seq_len,
        pad_to_max_seq_len=True,
        return_token_type_ids=True,
        return_attention_mask=True,
    )
    
def extract_and_stack_by_fields(encoded_features: List[dict], fields: List[str]) -> set:
    tensors = {}
    for field in fields:
        data = [feature[field] for feature in encoded_features]
        tensors[field] = np.array(data)
    
    return [tensors[field] for field in fields]
