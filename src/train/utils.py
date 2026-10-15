"""Configuration and loss utilities used by TurboAE training."""

import os
import pathlib
import random
import subprocess
from collections.abc import Callable
from typing import Protocol, cast

import numpy as np
import torch
import torch.distributed
from torch import nn
from torch.nn import functional


def get_git_metadata() -> tuple[str, str]:
    """Return the current Git branch and commit hash."""
    repo_root = pathlib.Path(__file__).resolve().parents[2]

    def run_git_command(args: list[str]) -> str:
        try:
            return subprocess.run(
                args,
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (FileNotFoundError, subprocess.CalledProcessError):
            return "unknown"

    return (
        run_git_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        run_git_command(["git", "rev-parse", "HEAD"]),
    )


class _LossConfig(Protocol):
    """Configuration values required by the loss factories."""

    sync_loss_function: str
    huber_delta: float
    smooth_l1_beta: float
    alpha: float
    beta: float
    loss_function: str


def resolve_device(local_rank: int = 0) -> torch.device:
    """
    Select the configured CUDA device, or CPU when CUDA is unavailable.

    Returns:
        The device on which training should run.

    """
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def ddp_setup(force_gloo: bool = False) -> None:
    """Set up distributed training with the selected PyTorch backend."""
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    # NCCL requires CUDA. Keep the configured backend on CUDA and use gloo for CPU DDP.
    backend = "gloo" if force_gloo or not cuda_available else "nccl"
    print(f"Using {backend} backend for distributed training")

    try:
        torch.distributed.init_process_group(backend=backend)
        print(f"Distributed training initialized with {backend} backend")
    except Exception as error:  # noqa: BLE001
        print(f"Failed to initialize with {backend}: {error}")
        if backend == "nccl":
            print("Falling back to gloo backend")
            torch.distributed.init_process_group(backend="gloo")


def setup_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible training."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def broadcast_seed(base_seed: int | None = None) -> int:
    """
    Select a seed on rank zero and broadcast it to all distributed workers.

    Returns:
        The seed agreed upon by every worker.

    Raises:
        RuntimeError: If the distributed process group has not been initialized.

    """
    if not torch.distributed.is_initialized():
        raise RuntimeError("Call torch.distributed.init_process_group first.")

    world_size = torch.distributed.get_world_size()
    print(f"world_size: {world_size}")
    if world_size <= 1:
        return base_seed if base_seed is not None else random.randint(0, 2**31 - 1)

    rank = torch.distributed.get_rank()
    device = resolve_device(torch.cuda.current_device() if torch.cuda.is_available() else 0)
    if rank == 0:
        seed = base_seed if base_seed is not None else random.randint(0, 2**31 - 1)
    else:
        seed = 0

    print(f"rank {rank} seed: {seed}")
    seed_tensor = torch.tensor([seed], dtype=torch.long, device=device)
    print(f"rank {rank} seed_tensor: {seed_tensor}")
    torch.distributed.broadcast(seed_tensor, src=0)
    print(f"rank {rank} seed_tensor after broadcast: {seed_tensor}")
    return int(seed_tensor.item())


class CustomLoss(nn.Module):
    """Combine binary and categorical cross-entropy objectives."""

    def __init__(self, alpha: float = 0.01) -> None:
        super().__init__()
        self.alpha = alpha
        self.bce_loss = nn.BCELoss()
        self.cce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        bce_prediction: torch.Tensor,
        bce_target: torch.Tensor,
        cce_prediction: torch.Tensor,
        cce_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the weighted sum of the BCE and CCE terms.

        Returns:
            A scalar loss tensor.

        """
        bce = self.bce_loss(bce_prediction, bce_target)
        cce = self.cce_loss(
            cce_prediction.to(torch.float32),
            cce_target.to(torch.float32),
        )
        return bce + self.alpha * cce


def probabilities_to_logits(probabilities: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Convert probabilities to logits while avoiding singularities at zero and one.

    Returns:
        A tensor of logits with the same shape as ``probabilities``.

    """
    probabilities = torch.clamp(probabilities, eps, 1 - eps)
    return torch.log(probabilities) - torch.log(1 - probabilities)


class ProductLoss(nn.Module):
    """Approximate block error rate using the product of bit likelihoods."""

    def forward(  # noqa: PLR6301
        self,
        probabilities: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the mean soft block-error loss.

        Returns:
            A scalar loss tensor.

        """
        logits = probabilities_to_logits(probabilities)
        term = (2 * targets.float() - 1) * logits
        return torch.mean(1.0 - torch.prod(torch.sigmoid(term), dim=1))


class MaxLoss(nn.Module):
    """Use the largest per-bit binary cross-entropy in each block."""

    def forward(  # noqa: PLR6301
        self,
        probabilities: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the mean maximum bit loss.

        Returns:
            A scalar loss tensor.

        """
        bce = functional.binary_cross_entropy_with_logits(
            probabilities,
            targets.float(),
            reduction="none",
        )
        return torch.mean(torch.max(bce, dim=1).values)


class SmoothMaxLoss(nn.Module):
    """Use an exponentially weighted smooth maximum of per-bit losses."""

    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the mean smooth-maximum bit loss.

        Returns:
            A scalar loss tensor.

        """
        bce = functional.binary_cross_entropy_with_logits(
            probabilities,
            targets.float(),
            reduction="none",
        )
        weights = torch.exp(self.alpha * bce)
        return torch.mean(torch.sum(bce * weights, dim=1) / torch.sum(weights, dim=1))


class LogSumExpLoss(nn.Module):
    """Aggregate per-bit losses with a log-sum-exp approximation."""

    def forward(  # noqa: PLR6301
        self,
        probabilities: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the mean log-sum-exp block loss.

        Returns:
            A scalar loss tensor.

        """
        bce = functional.binary_cross_entropy_with_logits(
            probabilities,
            targets.float(),
            reduction="none",
        )
        block_length = bce.shape[1]
        return torch.mean(
            torch.log(torch.sum(torch.exp(bce), dim=1) - (block_length - 1)),
        )


class PNormLoss(nn.Module):
    """Aggregate per-bit losses with a p-norm."""

    def __init__(self, p: float = 2, gamma: float = 1e-8) -> None:
        super().__init__()
        self.p = p
        self.gamma = gamma

    def forward(self, probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the mean p-norm block loss.

        Returns:
            A scalar loss tensor.

        """
        bce = functional.binary_cross_entropy_with_logits(
            probabilities,
            targets.float(),
            reduction="none",
        )
        return torch.mean(
            (torch.sum(bce**self.p, dim=1) + self.gamma) ** (1.0 / self.p),
        )


class HybridBCESoftBLERLoss(nn.Module):
    """Combine binary cross-entropy with a soft block-error objective."""

    def __init__(self, beta: float = 1.0, eps: float = 1e-12) -> None:
        super().__init__()
        self.beta = beta
        self.eps = eps

    def forward(self, probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the hybrid BCE and soft-BLER loss.

        Returns:
            A scalar loss tensor.

        """
        probabilities = probabilities.squeeze(-1)
        targets = targets.squeeze(-1)

        bce = functional.binary_cross_entropy(probabilities, targets)
        bit_likelihoods = probabilities * targets + (1 - probabilities) * (1 - targets)
        joint_probability = torch.prod(bit_likelihoods, dim=1)
        soft_bler = -torch.log(joint_probability + self.eps).mean()
        return bce + self.beta * soft_bler


class HybridBCEProductLoss(nn.Module):
    """Combine binary cross-entropy with :class:`ProductLoss`."""

    def __init__(self, beta: float = 1.0) -> None:
        super().__init__()
        self.beta = beta
        self.bce_loss = nn.BCELoss()
        self.product_loss = ProductLoss()

    def forward(self, probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the hybrid BCE and product loss.

        Returns:
            A scalar loss tensor.

        """
        return self.bce_loss(probabilities, targets) + self.beta * self.product_loss(
            probabilities,
            targets,
        )


class HybridBCEMaxLoss(nn.Module):
    """Combine binary cross-entropy with :class:`MaxLoss`."""

    def __init__(self, beta: float = 1.0) -> None:
        super().__init__()
        self.beta = beta
        self.bce_loss = nn.BCELoss()
        self.max_loss = MaxLoss()

    def forward(self, probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the hybrid BCE and maximum loss.

        Returns:
            A scalar loss tensor.

        """
        return self.bce_loss(probabilities, targets) + self.beta * self.max_loss(
            probabilities,
            targets,
        )


class HybridBCESmoothMaxLoss(nn.Module):
    """Combine binary cross-entropy with :class:`SmoothMaxLoss`."""

    def __init__(self, beta: float = 1.0) -> None:
        super().__init__()
        self.beta = beta
        self.bce_loss = nn.BCELoss()
        self.smooth_max_loss = SmoothMaxLoss()

    def forward(self, probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the hybrid BCE and smooth-maximum loss.

        Returns:
            A scalar loss tensor.

        """
        return self.bce_loss(probabilities, targets) + self.beta * self.smooth_max_loss(
            probabilities,
            targets,
        )


class HybridBCELogSumExpLoss(nn.Module):
    """Combine binary cross-entropy with :class:`LogSumExpLoss`."""

    def __init__(self, beta: float = 1.0) -> None:
        super().__init__()
        self.beta = beta
        self.bce_loss = nn.BCELoss()
        self.log_sum_exp_loss = LogSumExpLoss()

    def forward(self, probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the hybrid BCE and log-sum-exp loss.

        Returns:
            A scalar loss tensor.

        """
        return self.bce_loss(probabilities, targets) + self.beta * self.log_sum_exp_loss(
            probabilities,
            targets,
        )


class HybridBCEPNormLoss(nn.Module):
    """Combine binary cross-entropy with :class:`PNormLoss`."""

    def __init__(self, p: float, beta: float = 1.0) -> None:
        super().__init__()
        self.beta = beta
        self.bce_loss = nn.BCELoss()
        self.p_norm_loss = PNormLoss(p=p)

    def forward(self, probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the hybrid BCE and p-norm loss.

        Returns:
            A scalar loss tensor.

        """
        return self.bce_loss(probabilities, targets) + self.beta * self.p_norm_loss(
            probabilities,
            targets,
        )


def build_sync_loss_fn(config: object) -> nn.Module:
    """
    Build the configured synchronization-loss module.

    Returns:
        The selected synchronization-loss module.

    Raises:
        ValueError: If the configured loss name is unsupported.

    """
    loss_config = cast("_LossConfig", config)
    configured_loss = loss_config.sync_loss_function
    loss_name = str(configured_loss).strip().lower().replace(" ", "_")

    if loss_name in {"mse", "mse_loss", "mseloss"}:
        return nn.MSELoss()
    if loss_name in {"mae", "mae_loss", "maeloss", "l1", "l1_loss"}:
        return nn.L1Loss()
    if loss_name in {"huber", "huber_loss", "huberloss"}:
        return nn.HuberLoss(delta=loss_config.huber_delta)
    if loss_name in {"smooth_l1", "smooth_l1_loss", "smoothl1"}:
        return nn.SmoothL1Loss(beta=loss_config.smooth_l1_beta)

    supported = "MSE, MAE, Huber, smooth_l1"
    raise ValueError(
        f"Invalid synchronization loss {configured_loss!r}; expected one of: {supported}."
    )


def build_loss_fn(config: object) -> nn.Module:
    """
    Build the configured training-loss module.

    Returns:
        The selected training-loss module.

    Raises:
        ValueError: If the configured loss name is unsupported.

    """
    loss_config = cast("_LossConfig", config)
    loss_builders: dict[str, Callable[[], nn.Module]] = {
        "BCE": nn.BCELoss,
        "MSE": nn.MSELoss,
        "ProductLoss": ProductLoss,
        "MaxLoss": MaxLoss,
        "SmoothMaxLoss": lambda: SmoothMaxLoss(alpha=loss_config.alpha),
        "LogSumExpLoss": LogSumExpLoss,
        "PNormLoss": lambda: PNormLoss(p=2),
        "HybridBCESoftBLERLoss": lambda: HybridBCESoftBLERLoss(beta=loss_config.beta),
        "HybridBCEProductLoss": lambda: HybridBCEProductLoss(beta=loss_config.beta),
        "HybridBCEMaxLoss": lambda: HybridBCEMaxLoss(beta=loss_config.beta),
        "HybridBCESmoothMaxLoss": lambda: HybridBCESmoothMaxLoss(beta=loss_config.beta),
        "HybridBCELogSumExpLoss": lambda: HybridBCELogSumExpLoss(beta=loss_config.beta),
        "HybridBCEPNormLoss": lambda: HybridBCEPNormLoss(p=2, beta=loss_config.beta),
    }
    loss_name = loss_config.loss_function

    try:
        return loss_builders[str(loss_name)]()
    except KeyError as error:
        supported = ", ".join(loss_builders)
        raise ValueError(
            f"Invalid training loss {loss_name!r}; expected one of: {supported}.",
        ) from error
