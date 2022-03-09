from argparse import Action
from multiprocessing.sharedctypes import Value
import paddle
from paddle.nn import Layer
import paddle.nn as nn


def get_activation(name: str) -> Layer:
    activations = dict(
        relu=nn.ReLU,
        tanh=nn.Tanh,
        leaky_relu=nn.LeakyReLU,
    )
    if name not in activations:
        names = ','.join(list(activations.keys()))
        raise ValueError(f'activation<{name}> not supported, the expected name is: <{names}>')
    
    return activations[name]()
    