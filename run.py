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
from typing import Callable, List, Optional
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

def set_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader):
    """
    Given a dataset, it evals model and computes the metric.

    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
        criterion(obj:`paddle.nn.Layer`): It can compute the loss.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
    """
    logger.info('evaluation on dataset ...')
    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader:
        input_ids, token_type_ids, labels = batch
        logits = model(input_ids, token_type_ids)
        loss = criterion(logits, labels)
        losses.append(loss.numpy())
        correct = metric.compute(logits, labels)
        metric.update(correct)
    accu = metric.accumulate()
    print("eval loss: %.5f, accuracy: %.5f" % (np.mean(losses), accu))
    model.train()
    metric.reset()

def train(
    epoch: int,
    model: Layer,
    tokenizer: PretrainedTokenizer,
    criterion: Layer,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
    metric: Metric,
    train_dataloader: DataLoader,
    dev_dataloader: DataLoader,
    config: Config,
    scaler: Optional[AmpScaler] = None
):
    logger.info(f'training epoch<{epoch}> ...')
    global_step = epoch * len(train_dataloader)
    bar = tqdm(total=len(train_dataloader))

    for step, batch in enumerate(train_dataloader, start=1):
        input_ids, token_type_ids, labels = batch
        
        bar_info = ''
        with paddle.amp.auto_cast(
                config.use_amp,
                custom_white_list=["layer_norm", "softmax", "gelu"], ):
            logits = model(input_ids, token_type_ids)
            loss = criterion(logits, labels)
            bar.update()

            bar_info = f"loss: {loss.detach().cpu().numpy().item()}"

        probs = F.softmax(logits, axis=1)
        correct = metric.compute(probs, labels)

        bar_info += f'acc: {paddle.mean(correct).detach().cpu().numpy().item()}'
        bar.set_description(bar_info)

        metric.update(correct)
        metric.accumulate()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.minimize(optimizer, loss)
        else:
            loss.backward()
            optimizer.step()
        lr_scheduler.step()
        optimizer.clear_grad()

        global_step += 1
        if global_step % config.valid_steps == 0:
            evaluate(model, criterion, metric, dev_dataloader)

        # if global_step % config.save_steps == 0:
        #     save_dir = os.path.join(
        #         config.save_dir, "model_%d" % global_step)
        #     if not os.path.exists(save_dir):
        #         os.makedirs(save_dir)
        #     model._layers.save_pretrained(save_dir)
        #     tokenizer.save_pretrained(save_dir)

        global_step += 1


def do_train():
    config: Config = Config().parse_args(known_only=True)

    paddle.set_device(config.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()
        
    set_seed(config.seed)

    processor: DataProcessor = processors_map[config.task]()
    train_ds: MapDataset = processor.get_train_dataset()
    label2idx = {label:index for index, label in enumerate(train_ds.label_list)}

    dev_ds = processor.get_dev_dataset()
    model = ppnlp.transformers.ErnieForSequenceClassification.from_pretrained(
        'ernie-1.0', num_classes=len(train_ds.label_list))
    tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained('ernie-1.0')

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
        label2idx=label2idx
        )
    collate_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
        Stack(dtype="int64")  # label
    ): [data for data in fn(samples)]
    train_data_loader = create_dataloader(
        train_ds,
        mode='train',
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        trans_fn=trans_func)
    dev_data_loader = create_dataloader(
        dev_ds,
        mode='dev',
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        trans_fn=trans_func
    )

    if config.init_from_ckpt and os.path.isfile(config.init_from_ckpt):
        state_dict = paddle.load(config.init_from_ckpt)
        model.set_dict(state_dict)
    named_parameters = model.named_parameters()

    model = paddle.DataParallel(model)

    num_training_steps = len(train_data_loader) * config.epochs

    lr_scheduler = LinearDecayWithWarmup(config.learning_rate, num_training_steps,
                                         config.warmup_proportion)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in named_parameters 
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=config.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)
        
    criterion = paddle.nn.loss.CrossEntropyLoss()
    metric = paddle.metric.Accuracy()
    scaler = None
    if config.use_amp:
        scaler = paddle.amp.GradScaler(init_loss_scaling=config.scale_loss)
    
    for epoch in range(1, config.epochs + 1):
        train(
            epoch, model, tokenizer, criterion, optimizer, lr_scheduler, metric, train_data_loader, dev_data_loader, config, scaler
        )

if __name__ == "__main__":
    do_train()
