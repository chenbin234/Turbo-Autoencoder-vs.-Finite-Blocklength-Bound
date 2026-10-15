"""Straight-through estimators for quantization and sign operations."""

from typing import Protocol, cast

import torch
from torch import Tensor

# PyTorch autograd functions intentionally implement framework-defined methods
# whose concrete signatures differ from the variadic base declarations.


class _AutogradContext(Protocol):
    @property
    def saved_tensors(self) -> tuple[Tensor, ...]:
        """Tensors saved during the forward pass."""

    def save_for_backward(self, *tensors: Tensor) -> None:
        """Save tensors required by the backward pass."""


class MySign(torch.autograd.Function):
    """Apply the sign function with a clipped straight-through gradient."""

    @staticmethod
    def forward(ctx: object, inputs: Tensor) -> Tensor:
        """
        Apply the element-wise sign operation.

        Args:
            ctx: Context object for saving information for backward.
            inputs: Input tensor.

        Returns:
            The signed tensor.

        """
        context = cast("_AutogradContext", ctx)
        context.save_for_backward(inputs)
        return torch.sign(inputs)

    @staticmethod
    def backward(ctx: object, *grad_outputs: object) -> Tensor:
        """
        Pass gradients through inputs in the interval ``[-1, 1]``.

        Args:
            ctx: Context object with saved tensors.
            grad_outputs: Gradients of the loss with respect to the outputs.

        Returns:
            The masked input gradient.

        """
        context = cast("_AutogradContext", ctx)
        grad_output = cast("Tensor", grad_outputs[0])

        (inputs,) = context.saved_tensors
        mask = (inputs.abs() <= 1).float()
        grad_input = grad_output * mask
        # ! optinal: test if the gradient is the key to better performance
        # grad_input = torch.clamp(grad_input, -0.25, 0.25)
        return grad_input
