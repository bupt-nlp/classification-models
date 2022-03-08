from __future__ import annotations
from typing import Optional

import numpy as np
import paddle
from paddle.io import Dataset
from paddlenlp.transformers.tokenizer_utils import PretrainedTokenizer


def convert_example(example: dict,
                    tokenizer: PretrainedTokenizer,
                    max_seq_length: int = 512,
                    mode: str = 'train'
                    ):
    """convert single example to input related data

    Args:
        example (InputExample): Single Input Example object
        tokenizer (PretrainedTokenizer): pretrained tokenizer
        max_seq_length (int, optional): max sequence length. Defaults to 512.
        mode (str, optional): the mode of model. Defaults to 'train'.
    """
    if 'text_b' in example:
        text, text_pair = example['text_a'], example['text_b']
    else:
        text, text_pair = example['text'], None 

    encoded_inputs = tokenizer(
        text=text,
        text_pair=text_pair,
        max_seq_len=max_seq_length
    )
    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]
    if mode == 'test':
        return input_ids, token_type_ids

    label = np.array([example["label"]], dtype="int64")
    return input_ids, token_type_ids, label


def create_dataloader(dataset: Dataset,
                      mode: str = 'train',
                      batch_size: int = 16,
                      collate_fn = None,
                      trans_fn = None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset, batch_size=batch_size, shuffle=True)
    else:
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size)

    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        return_list=True
    )
