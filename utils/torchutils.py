#Extension from sbi/utils/torchutils.py
#Intending to create a BoxNormal and a combination of BoxUniform/BoxNormal
from typing import Union

import numpy as np
import torch
from torch import Tensor, float32
from torch.distributions import Independent, Normal

from sbi.types import Array, OneOrMore, ScalarFloat

class BoxNormal(Independent):
    def __init__(
        self,
        loc: ScalarFloat,
        scale: ScalarFloat,
        reinterpreted_batch_ndims: int = 1,
        device: str = "cpu",
    ):
        """Multidimensional uniform distribution defined on a box.
        A `Uniform` distribution initialized with e.g. a parameter vector low or high of
         length 3 will result in a /batch/ dimension of length 3. A log_prob evaluation
         will then output three numbers, one for each of the independent Uniforms in
         the batch. Instead, a `BoxUniform` initialized in the same way has three
         /event/ dimensions, and returns a scalar log_prob corresponding to whether
         the evaluated point is in the box defined by low and high or outside.
        Refer to torch.distributions.Uniform and torch.distributions.Independent for
         further documentation.
        Args:
            low: lower range (inclusive).
            high: upper range (exclusive).
            reinterpreted_batch_ndims (int): the number of batch dims to
                                             reinterpret as event dims.
            device: device of the prior, defaults to "cpu", should match the training
                device when used in SBI.
        """

        super().__init__(
            Normal(
                loc=torch.as_tensor(loc, dtype=torch.float32, device=device),
                scale=torch.as_tensor(scale, dtype=torch.float32, device=device),
                validate_args=False,
            ),
            reinterpreted_batch_ndims,
        )


def ensure_theta_batched(theta: Tensor) -> Tensor:
    r"""
    Return parameter set theta that has a batch dimension, i.e. has shape
     (1, shape_of_single_theta)
     Args:
         theta: parameters $\theta$, of shape (n) or (1,n)
     Returns:
         Batched parameter set $\theta$
    """

    # => ensure theta has shape (1, dim_parameter)
    if theta.ndim == 1:
        theta = theta.unsqueeze(0)

    return theta


def ensure_x_batched(x: Tensor) -> Tensor:
    """
    Return simulation output x that has a batch dimension, i.e. has shape
    (1, shape_of_single_x).
    Args:
         x: simulation output of shape (n) or (1,n).
     Returns:
         Batched simulation output x.
    """

    # ensure x has shape (1, shape_of_single_x). If shape[0] > 1, we assume that
    # the batch-dimension is missing, even though ndim might be >1 (e.g. for images)
    if x.shape[0] > 1 or x.ndim == 1:
        x = x.unsqueeze(0)

    return x


def atleast_2d_many(*arys: Array) -> OneOrMore[Tensor]:
    """Return tensors with at least dimension 2.
    Tensors or arrays of dimension 0 or 1 will get additional dimension(s) prepended.
    Returns:
        Tensor or list of tensors all with dimension >= 2.
    """
    if len(arys) == 1:
        arr = arys[0]
        if isinstance(arr, np.ndarray):
            arr = torch.from_numpy(arr)
        return atleast_2d(arr)
    else:
        return [atleast_2d_many(arr) for arr in arys]


def atleast_2d(t: Tensor) -> Tensor:
    return t if t.ndim >= 2 else t.reshape(1, -1)


def maybe_add_batch_dim_to_size(s: torch.Size) -> torch.Size:
    """
    Take a torch.Size and add a batch dimension to it if dimensionality of size is 1.
    (N) -> (1,N)
    (1,N) -> (1,N)
    (N,M) -> (N,M)
    (1,N,M) -> (1,N,M)
    Args:
        s: Input size, possibly without batch dimension.
    Returns: Batch size.
    """
    return s if len(s) >= 2 else torch.Size([1]) + s


def atleast_2d_float32_tensor(arr: Union[Tensor, np.ndarray]) -> Tensor:
    return atleast_2d(torch.as_tensor(arr, dtype=float32))


def batched_first_of_batch(t: Tensor) -> Tensor:
    """
    Takes in a tensor of shape (N, M) and outputs tensor of shape (1,M).
    """
    return t[:1]


def assert_all_finite(quantity: Tensor, description: str = "tensor") -> None:
    """Raise if tensor quantity contains any NaN or Inf element."""

    msg = f"NaN/Inf present in {description}."
    assert torch.isfinite(quantity).all(), msg