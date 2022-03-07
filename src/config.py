from typing import Literal, Callable, List
from tap import Tap

from paddlenlp.data import Tuple

from src.data import InputExample, InputFeature


CollateFn = Callable[[List[InputExample]], List[Tuple]]
TransFn = Callable[[InputExample], InputFeature]

class Config(Tap):
    """Configuration for Training"""
    save_dir: str = './checkpoint'      # The output directory where the model checkpoints will be written.
    dataset: Literal['chnsenticorp', 'xnli_cn'] = 'chnsenticorp'       # Dataset for classfication tasks.
    max_seq_length: int = 128           # The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.
    batch_size: int = 32                # Batch size per GPU/CPU for training.
    learning_rate: float = 5e-5         # The initial learning rate for Adam.
    weight_decay: float = 0             # Weight decay if we apply some.
    epochs: int = 3                     # Total number of training epochs to perform.

    warmup_proportion: float = 0        # Linear warmup proption over the training process.

    valid_steps: int = 100              # The interval steps to evaluate model performance.
    save_steps: int = 100               # The interval steps to save checkppoints.

    logging_steps: int = 10             # The interval steps to logging.
    init_from_ckpt: str = None          # The path of checkpoint to be loaded.
    seed: int = 1000                    # random seed for initialization
    device: Literal['cpu', 'gpu', 'xpu', 'npu'] = 'gpu'     # Select which device to train model, defaults to gpu.
    use_amp: bool = False               # Enable mixed precision training.
    scale_loss: float = 2**15           # The value of scale_loss for fp16.