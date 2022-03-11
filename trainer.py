"""
refer to: https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_classification
"""
from __future__ import annotations

from functools import partial
import argparse
import os
import random
import time
import distutils.util
from collections import defaultdict
from typing import Callable, Dict, List, Optional
from matplotlib.container import ErrorbarContainer
import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.metric.metrics import Metric, Accuracy
from paddle.io import DataLoader
from paddle.optimizer import Optimizer
from paddle.optimizer.lr import LRScheduler
from paddle.nn import Layer
from paddle.amp.grad_scaler import AmpScaler


import paddlenlp as ppnlp
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset, MapDataset
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.transformers.tokenizer_utils import PretrainedTokenizer
from loguru import logger
from src.processors import convert_example, create_dataloader, processors_map
from src.processors.base_processor import DataProcessor
from src.config import Config
from tqdm import tqdm
from src.models.cnn import CNNConfig, CNNClassifier
from src.models.simple_classifier import SimpleConfig, SimpleClassifier
from src.models.rnn import RNNConfig, RNNClassifier
from src.data import InputExample, InputFeature, ExampleDataset, extract_and_stack_by_fields


def set_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


class Trainer:
    def __init__(self, config: Config, processor: DataProcessor, classifier) -> None:
        self.config = config
        self.set_device()
        self.tokenizer: PretrainedTokenizer = ppnlp.transformers.BertTokenizer.from_pretrained(
            config.pretrained_model)

        # 2. build data related objects
        self.train_dataset = processor.get_train_dataset()
        self.dev_dataset = processor.get_dev_dataset()
        self.test_dataset = processor.get_test_dataset()
        self.train_dataloader = create_dataloader(
            self.train_dataset,
            batch_size=config.batch_size,
            collate_fn=lambda examples: self.collate_fn(examples, self.train_dataset.label2idx),
        )
        self.dev_dataloader = create_dataloader(
            self.dev_dataset,
            batch_size=config.batch_size,
            collate_fn=lambda examples: self.collate_fn(examples, self.dev_dataset.label2idx),
        )
        self.test_dataloader = create_dataloader(
            self.test_dataset,
            batch_size=config.batch_size,
            collate_fn=lambda examples: self.collate_fn(examples, self.test_dataset.label2idx),
        )

        # 3. init model related
        self.model = classifier
        self.lr_scheduler: LRScheduler = LinearDecayWithWarmup(
            config.learning_rate, 
            total_steps=len(self.train_dataloader) * config.epochs,
            warmup=config.warmup_proportion
        )
        if config.init_from_ckpt and os.path.isfile(config.init_from_ckpt):
            state_dict = paddle.load(config.init_from_ckpt)
            self.model.set_dict(state_dict)
        # Generate parameter names needed to perform weight decay.
        # All bias and LayerNorm parameters are excluded.

        decay_params = [
            p.name for n, p in self.model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ]
        self.optimizer: Optimizer = paddle.optimizer.AdamW(
            learning_rate=self.lr_scheduler,
            parameters=self.model.parameters(),
            weight_decay=config.weight_decay,
            apply_decay_param_fun=lambda x: x in decay_params)

        self.criterion = paddle.nn.loss.CrossEntropyLoss()
        self.metric: Metric = paddle.metric.Accuracy()
        # self.model = paddle.DataParallel(self.model)

        self.context_data = defaultdict(int)

    def collate_fn(self, examples: List[InputExample], label2idx: Dict[str, int]):
        # 1. construct text or text pair dataset
        is_pair = examples[0].text_pair is not None
        has_label = examples[0].label is not None
        if is_pair:
            texts = [(example.text, example.text_pair) for example in examples]
        else:
            texts = [example.text for example in examples]

        encoded_features = self.tokenizer.batch_encode(
            texts,
            max_seq_len=self.config.max_seq_length,
            pad_to_max_seq_len=True,
            return_token_type_ids=True
        )
        fields = ['input_ids', 'token_type_ids']
        
        # 2. return different data based on label
        if not has_label:
            return extract_and_stack_by_fields(encoded_features, fields)
        
        label_ids = []
        is_multi_class = isinstance(examples[0].label, list)
        if not is_multi_class:
            label_ids = [label2idx[example.label] for example in examples]
        else:
            for example in examples:
                example_label_ids = [label2idx[label] for label in example.label]
                label_ids.append(example_label_ids)
        
        features = extract_and_stack_by_fields(encoded_features, fields)
        features.append(
            np.array(label_ids)
        )
        return features

    def set_device(self):
        paddle.set_device(self.config.device)
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.init_parallel_env()

    @paddle.no_grad()
    def evalute(self, dataloader: DataLoader, mode: str = 'dev'):
        logger.success(f'{mode} stage ...')

        self.model.eval()
        self.metric.reset()
        losses = []
        for batch in dataloader:
            input_ids, token_type_ids, labels = batch
            logits = self.model(input_ids, token_type_ids)
            loss = self.criterion(logits, labels)
            losses.append(loss.numpy())
            correct = self.metric.compute(logits, labels)
            self.metric.update(correct)
        accu = self.metric.accumulate()
        logger.info("eval loss: %.5f, accuracy: %.5f" % (np.mean(losses), accu))
        self.model.train()
        self.metric.reset()

    def _update_bar_info(self, fields: List[str]):
        bar_info = []
        for field in fields:
            data = self.context_data[field]
            if paddle.is_tensor(data):
                data = data.detach().cpu().numpy().item()
            bar_info.append(f'{field}: {data}')

        self.train_bar.set_description('\t'.join(bar_info))

    def on_batch_end(self):
        # 1. update global step
        self.context_data['global_steps'] += 1
        self.train_bar.update()

        # 2. compute acc on training dataset
        fields = ['loss', 'train-acc']
        train_acc = paddle.mean(self.metric.compute(self.context_data['logits'], self.context_data['labels'])).numpy().item()
        self.context_data['train-acc'] = train_acc
        self._update_bar_info(fields)

        # 3. step the grad 
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.clear_grad()

        # 4. eval on dev dataset
        if self.context_data['global_steps'] % self.config.valid_steps == 0:
            self.evalute(self.dev_dataloader)

    def on_batch_start(self):
        self.metric.reset()
    
    def train_epoch(self, epoch: int):
        logger.info(f'training epoch<{epoch}> ...')

        self.train_bar = tqdm(total=len(self.train_dataloader))

        for step, batch in enumerate(self.train_dataloader, start=1):
            input_ids, token_type_ids, labels = batch

            self.on_batch_start()

            with paddle.amp.auto_cast(
                    self.config.use_amp,
                    custom_white_list=["layer_norm", "softmax", "gelu"], ):
                logits = self.model(
                    input_ids=input_ids, 
                    token_type_ids=token_type_ids
                )

                loss = self.criterion(logits, labels)

                self.context_data['logits'] = logits
                self.context_data['loss'] = loss
                self.context_data['labels'] = labels

            loss.backward()
            self.on_batch_end()

    def train(self):
        for epoch in range(1, self.config.epochs + 1):
            self.train_epoch(epoch)
            self.evalute(self.test_dataloader, mode='test')
    
    def predict(self):
        pass

    
