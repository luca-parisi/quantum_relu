"""
This file contains utility-based or helper functions leveraged across the two quantum activation functions
QReLU and m-QReLU.
"""
from typing import Callable, Optional, Union

import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from .constants import (MAX_RANDINT, MIN_RANDINT, PREFIX_FUNC_NAME, STATEFUL,
                        SUFFIX_GRAD_NAME)


def py_func(
        func: Callable,
        inp: list[Tensor],
        t_out: Union[list[Tensor], tuple[Tensor]],
        stateful: bool = STATEFUL,
        name: str = None,
        grad: Optional[float] = None
) -> Union[list[Tensor], Tensor]:
    """
    Generate a unique name of the function to avoid duplicates.

    func: Callable
        A Python function, which accepts `ndarray` objects as arguments and
        returns a list of `ndarray` objects (or a single `ndarray`). This function
        must accept as many arguments as there are tensors in `inp`, and these
        argument types will match the corresponding `tf.Tensor` objects in `inp`.
        The returns `ndarray`s must match the number and types defined `Tout`.
        N.B.: Input and output numpy `ndarray`s of `func` are not
        guaranteed to be copies. In some cases, their underlying memory will be
        shared with the corresponding TensorFlow tensors. In-place modification
        or storing `func` input or return values in python datastructures
        without explicit (np.)copy can have non-deterministic consequences.
    inp: list[Tensor]
        A list of `Tensor` objects.
    t_out: Union[list[Tensor], tuple[Tensor]]
        A list or tuple of tensorflow data types or a single tensorflow data
        type if there is only one, indicating what `func` returns.
    stateful: bool
        By default, it is True; thus, the function is considered stateful.
        If a function is stateless, when given the same input, it will return the same
        output and have no observable side effects. Optimizations, such as common
        subexpression elimination, are only performed on stateless operations.
    name: str
        A name for the operation (optional).
    grad: float
        A gradient.

    Returns:
            A list of `Tensor` or a single `Tensor` which `func` computes.
    """

    rnd_name = f'{PREFIX_FUNC_NAME}{SUFFIX_GRAD_NAME}{str(np.random.randint(MIN_RANDINT, MAX_RANDINT))}'
    tf.RegisterGradient(rnd_name)(grad)
    g = tf.compat.v1.get_default_graph()
    with g.gradient_override_map({PREFIX_FUNC_NAME: rnd_name}):
        return tf.compat.v1.py_func(func, inp, t_out, stateful=stateful, name=name)
