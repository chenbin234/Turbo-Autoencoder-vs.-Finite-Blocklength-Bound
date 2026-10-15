"""Soft QPSK constraint that pulls encoder symbols toward the normalized QPSK constellation."""

import math
from typing import Protocol, cast

import torch
from torch import Tensor

# Magnitude of the I and Q components of a normalized QPSK point (+-1 +-1j) / sqrt(2).
_QPSK_AMPLITUDE = 1.0 / math.sqrt(2.0)

# PyTorch autograd functions intentionally implement framework-defined methods
# whose concrete signatures differ from the variadic base declarations.


class _AutogradContext(Protocol):
    @property
    def saved_tensors(self) -> tuple[Tensor, ...]:
        """Tensors saved during the forward pass."""

    def save_for_backward(self, *tensors: Tensor) -> None:
        """Save tensors required by the backward pass."""


class SoftQPSK(torch.autograd.Function):
    """
    Pull every I/Q symbol a fraction ``alpha`` of the way to its nearest normalized QPSK point.

    Each symbol ``x`` is first scaled to unit energy, ``u = x / |x|``, so that ``alpha``
    means the same thing at any encoder output scale. The nearest normalized QPSK point
    ``q`` is the quadrant of ``u``, and the output is ``u + alpha * (q - u)``:

    - ``alpha = 0`` returns ``u``, which is what the per-symbol power constraint transmits anyway.
    - ``0 < alpha < 1`` shrinks the error vector to the nearest QPSK point by ``1 - alpha``,
      so every symbol has phase error of at most ``atan((1 - alpha) / (1 + (sqrt(2) - 1) * alpha))``
      (22.5 degrees at ``alpha = 0.5``) after per-symbol normalization.
    - ``alpha = 1`` returns exact QPSK, the same symbols as ``enforce_qpsk_symbols``.

    The backward pass treats the pull ``alpha * (q - u)`` as a constant, so the gradient
    reaches ``x`` through the unit-energy normalization only. For ``alpha < 1`` this is the
    exact gradient of the forward map rescaled by ``1 / (1 - alpha)``: the direction is
    exact, not a surrogate, and the scale stays independent of ``alpha``. At ``alpha = 1``
    it becomes a straight-through estimator.
    """

    @staticmethod
    def forward(ctx: object, inputs: Tensor, alpha: float) -> Tensor:
        """
        Apply the soft QPSK pull to I/Q symbols.

        Args:
            ctx: Context object for saving information for backward.
            inputs: Real-valued symbols with the in-phase and quadrature components in the last dimension, shape (..., 2).
            alpha: Fraction of the distance to the nearest normalized QPSK point to remove, in ``[0, 1]``.

        Returns:
            The softly constrained symbols, with the same shape as ``inputs``.

        """
        context = cast("_AutogradContext", ctx)

        # normalise each symbol to unit energy; the floor keeps an all-zero symbol finite
        power = inputs.square().sum(dim=-1, keepdim=True)  # output size = (..., 1)
        eps = torch.finfo(power.dtype).eps
        inv_norm = torch.rsqrt(power.clamp_min(eps))
        unit = inputs * inv_norm  # output size = (..., 2)

        # nearest normalized QPSK point: the quadrant of the symbol, on the unit circle
        qpsk = torch.where(unit >= 0, _QPSK_AMPLITUDE, -_QPSK_AMPLITUDE).to(unit.dtype)

        context.save_for_backward(unit, inv_norm, power >= eps)
        return unit + alpha * (qpsk - unit)

    @staticmethod
    def backward(ctx: object, *grad_outputs: object) -> tuple[Tensor, None]:
        """
        Pass gradients straight through the pull and exactly through the normalization.

        Args:
            ctx: Context object with saved tensors.
            grad_outputs: Gradients of the loss with respect to the outputs.

        Returns:
            The input gradient, and ``None`` for the non-differentiable ``alpha``.

        """
        context = cast("_AutogradContext", ctx)
        grad_output = cast("Tensor", grad_outputs[0])

        (unit, inv_norm, is_unclamped) = context.saved_tensors
        # the Jacobian of x / |x| is (I - u u^T) / |x|: it removes the radial part of the gradient,
        # except below the energy floor, where the norm is a constant and the Jacobian is I / sqrt(eps)
        radial = (unit * grad_output).sum(dim=-1, keepdim=True) * is_unclamped
        grad_input = (grad_output - unit * radial) * inv_norm
        return grad_input, None
