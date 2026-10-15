"""Blocks for synchronization models."""

import torch
import torch.nn.functional as F
from torch import nn


def make_activation(name: str) -> nn.Module:
    """Return the activation function corresponding to the given name."""
    activation = name.lower().replace("-", "_")
    if activation == "elu":
        return nn.ELU()
    if activation == "relu":
        return nn.ReLU()
    if activation == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1)
    if activation == "gelu":
        return nn.GELU()
    if activation == "silu":
        return nn.SiLU()
    if activation == "selu":
        return nn.SELU()
    if activation == "tanh":
        return nn.Tanh()
    if activation == "identity":
        return nn.Identity()
    raise ValueError(f"Unsupported activation '{name}'.")


class ShiftResidualConvBlock(nn.Module):
    """Residual temporal block that compares a feature stream with its shifts.

    The center stream and its two directional finite differences contain the
    same linear information as ``(x, x_left, x_right, x - x_left,
    x - x_right)`` without presenting the convolution with five linearly
    dependent copies of every channel.
    """

    def __init__(self, channels: int, kernel_size: int = 5, activation: str = "elu") -> None:
        super().__init__()
        self.activation = make_activation(activation)
        self.shift_conv = nn.Conv1d(in_channels=channels * 3, out_channels=channels, kernel_size=kernel_size, padding="same")
        self.norm1 = nn.BatchNorm1d(channels)
        self.conv = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding="same")
        self.norm2 = nn.BatchNorm1d(channels)
        self.gate = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def shift_left(x: torch.Tensor) -> torch.Tensor:
        """Shift the input tensor to the left along the last dimension, padding with zeros on the right."""
        return F.pad(x[..., 1:], pad=(0, 1), mode="constant", value=0)

    @staticmethod
    def shift_right(x: torch.Tensor) -> torch.Tensor:
        """Shift the input tensor to the right along the last dimension, padding with zeros on the left."""
        return F.pad(x[..., :-1], pad=(1, 0), mode="constant", value=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gated shift-residual convolutional update."""
        x_left = self.shift_left(x)
        x_right = self.shift_right(x)

        # Raw shifted streams are recoverable from x and these differences, so
        # including all five tensors would add parameters but no information.
        shifted_features = torch.cat(
            (x, x - x_left, x - x_right),
            dim=1,
        )

        update = self.shift_conv(shifted_features)
        update = self.activation(self.norm1(update))
        update = self.conv(update)
        update = self.norm2(update)
        return x + torch.sigmoid(self.gate) * update


class Attention(nn.Module):
    """Pointwise temporal attention pooling over sequence features."""

    def __init__(self, channels: int, activation: str) -> None:
        super().__init__()
        self.attention_pool = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            make_activation(activation),
            nn.Conv1d(channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the attention-weighted pooled sequence features."""
        attention = torch.softmax(self.attention_pool(x), dim=-1)
        attention_pooled = torch.sum(x * attention, dim=-1)
        return attention_pooled


class DelayEstimator(nn.Module):
    """Estimate continuous delay d = q * ts + eps from a delayed triangular pulse."""

    CANDIDATE_CORRELATION_INDEX = 6

    def __init__(
        self,
        hidden_channels: int = 64,
        num_layers: int = 2,
        kernel_size: int = 5,
        eps_margin: float = 1e-3,
        activation: str = "elu",
        fuse_blocks: bool = False,
        dmax: float = 12.0,
        ts: float = 1.0,
        separate_q_eps: bool = True,
        num_input_features: int = 2,
        num_candidate_features: int = 10,
    ) -> None:
        super().__init__()

        self.hidden_channels = hidden_channels
        self.dmax = dmax
        self.ts = ts
        self.delay_margin = float(eps_margin)
        self.eps_margin = float(eps_margin)
        self.fuse_blocks = fuse_blocks
        self.separate_q_eps = separate_q_eps
        self.num_input_features = num_input_features
        self.num_candidate_features = num_candidate_features
        self.output_input_dim = hidden_channels
        self.num_q_classes = int(self.dmax / self.ts) + 1

        self.input_projection = nn.Sequential(
            nn.Conv1d(in_channels=self.num_input_features, out_channels=hidden_channels, kernel_size=1),
            nn.BatchNorm1d(hidden_channels),
            make_activation(activation),
        )

        self.shift_residual_conv_blocks = nn.Sequential(*[
            ShiftResidualConvBlock(channels=hidden_channels, kernel_size=kernel_size, activation=activation) for _ in range(num_layers)
        ])

        self.attention_pool = Attention(hidden_channels, activation)

        self.candidate_projection = nn.Sequential(
            nn.Linear(self.num_candidate_features, hidden_channels),
            make_activation(activation),
            nn.Linear(hidden_channels, hidden_channels),
            make_activation(activation),
        )

        self.q_head = nn.Linear(hidden_channels, 1)
        self.q_prior_scale = nn.Parameter(torch.tensor(10.0))
        self.q_correlation_prior_scale = nn.Parameter(torch.tensor(1.0))
        self.candidate_eps_head = nn.Linear(hidden_channels, 1)
        self.fused_eps_output = nn.Sequential(
            nn.Linear(self.num_q_classes, hidden_channels),
            make_activation(activation),
            nn.Linear(hidden_channels, 1),
        )

    @staticmethod
    def _extract_feature_mean(x: torch.Tensor, start: int, end: int) -> torch.Tensor:
        if end <= start:
            return torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
        return x[:, start:end].mean(dim=-1, keepdim=True)

    @staticmethod
    def _extract_feature_std(x: torch.Tensor, start: int, end: int) -> torch.Tensor:
        if end <= start:
            return torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
        return x[:, start:end].std(dim=-1, keepdim=True, unbiased=False)

    @staticmethod
    def _outside_feature_mean(x: torch.Tensor, support_start: int, support_end: int) -> torch.Tensor:
        """Average every sample outside a proposed support window."""
        before_count = support_start
        after_count = x.shape[-1] - support_end
        outside_count = before_count + after_count
        if outside_count <= 0:
            return torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)

        before_sum = x[:, :support_start].sum(dim=-1, keepdim=True)
        after_sum = x[:, support_end:].sum(dim=-1, keepdim=True)
        return (before_sum + after_sum) / outside_count

    def _build_candidate_features(
        self,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Describe each integer-delay hypothesis with distinct evidence."""
        if y.dim() != 3:
            raise ValueError("Expected y with shape (batch, nb, length).")

        # y shape (batch, nb, length), torch.complex64
        batch_size, _, received_length = y.shape
        signal_length = received_length - self.num_q_classes + 1  # 224-97+1=128
        q_positions = torch.arange(self.num_q_classes, device=y.device, dtype=y.real.dtype)  # [0, 1, 2, 3, ..., 96]

        # Compute power related features.
        power = torch.abs(y) ** 2  # (batch, nb, length), torch.float32
        raw_power_mean = power.mean(dim=1)  # (batch, length), torch.float32
        block_mean_power = power.mean(dim=-1, keepdim=True).clamp_min(1e-12)  # (batch, nb, 1), torch.float32
        normalized_power = (
            power / block_mean_power
        )  # (batch, nb, length), in case training has variable SNR, this cancel the effect of SNR on the features
        power_mean = normalized_power.mean(dim=1)  # (batch, length), torch.float32
        power_std = normalized_power.std(dim=1, unbiased=False)  # (batch, length), torch.float32

        # Compute correlation features, y[t] * conj(y[t + 1])
        adjacent_corr = y[..., 1:] * torch.conj(y[..., :-1])  # (batch, nb, length-1), torch.complex64
        normalized_corr = adjacent_corr / block_mean_power  # (batch, nb, length-1), torch.complex64
        corr_real_mean = normalized_corr.real.mean(dim=1)  # (batch, length-1), torch.float32
        corr_imag_mean = normalized_corr.imag.mean(dim=1)  # (batch, length-1), torch.float32
        corr_abs_mean = torch.abs(normalized_corr).mean(dim=1)  # (batch, length-1), torch.float32

        # initialize candidate features for each integer-delay q.
        candidate_features = []
        q_prior_features = []
        e_priors = []

        # for each hypothesis of q, extract the signal and calculate the features
        for q_index in range(self.num_q_classes):
            support_start = q_index
            right_boundary = support_start + signal_length
            support_end = min(right_boundary + 1, received_length)

            # Calculate the normalized q value features
            q_value = q_positions[q_index].expand(batch_size, 1)  # (batch, 1), torch.float32
            q_normalized = q_value / max(self.num_q_classes - 1, 1)  # (batch, 1), torch.float32, normalized q value in [0, 12]
            centered_q_squared = (q_normalized - self.dmax / 2) ** 2

            # Extract the mean and std of the power within the support window, as well as the mean outside the support window.
            extract_power_mean = self._extract_feature_mean(power_mean, support_start, support_end)  # (batch, 1), torch.float32
            extract_power_std = self._extract_feature_std(power_mean, support_start, support_end)  # (batch, 1), torch.float32
            outside_power_mean = self._outside_feature_mean(power_mean, support_start, support_end)  # (batch, 1), torch.float32
            diff_extract_outside_power_mean = extract_power_mean - outside_power_mean

            # Calculate the left and right bundary energy
            left_boundary_power = power_mean[:, support_start : support_start + 1]  # (batch, 1), torch.float32
            if right_boundary < received_length:
                right_boundary_power = power_mean[:, right_boundary : right_boundary + 1]  # (batch, 1), torch.float32
            else:
                right_boundary_power = torch.zeros_like(left_boundary_power)

            # Calculate the bundary energy as the sum of the differences between the left and right boundary power and the outside power mean.
            diff_left_boundary_outside_mean = (left_boundary_power - outside_power_mean).clamp_min(0.0)  # (batch, 1), torch.float32
            diff_right_boundary_outside_mean = (right_boundary_power - outside_power_mean).clamp_min(0.0)  # (batch, 1), torch.float32
            boundary_energy = diff_left_boundary_outside_mean + diff_right_boundary_outside_mean  # (batch, 1), torch.float32

            # Calculate the eps prior based on the left and right boundary energy.
            # X_d[n] = (1-alpha)* X_q[n] + alpha * X_q[n+1], alpha = eps / ts
            # E_L = |h|^2 * |s[0]|^2 * (1-alpha)^2, E_R = |h|^2 * |s[L-1]|^2 * alpha^2
            # alpha = sqrt(E_R) / (sqrt(E_L) + sqrt(E_R))
            # eps = alpha * ts
            sqrt_floor = 1e-6
            floor_root = sqrt_floor**0.5  # Numerical stabilization
            left_amplitude = torch.sqrt(diff_left_boundary_outside_mean + sqrt_floor) - floor_root  # (batch, 1), torch.float32
            right_amplitude = torch.sqrt(diff_right_boundary_outside_mean + sqrt_floor) - floor_root  # (batch, 1), torch.float32
            alpha = right_amplitude / (left_amplitude + right_amplitude + 1e-8)  # (batch, 1), torch.float32
            e_prior = alpha * self.ts  # (batch, 1), torch.float32

            # Calculate the mean of the correlation within the support window
            corr_end = support_end - 1
            corr_real_mean_extract = self._extract_feature_mean(corr_real_mean, support_start, corr_end)  # (batch, 1), torch.float32
            corr_imag_mean_extract = self._extract_feature_mean(corr_imag_mean, support_start, corr_end)  # (batch, 1), torch.float32
            corr_abs_mean_extract = self._extract_feature_mean(corr_abs_mean, support_start, corr_end)  # (batch, 1), torch.float32

            # Calculate the q prior based on the mean of the power within the support window and the mean of the power outside the support window.
            # raw_power_mean of shape (batch, length), torch.float32
            raw_extract_power_mean = self._extract_feature_mean(raw_power_mean, support_start, support_end)  # (batch, 1), torch.float32
            raw_outside_feature_mean = self._outside_feature_mean(raw_power_mean, support_start, support_end)  # (batch, 1), torch.float32
            q_prior_feature = raw_extract_power_mean - raw_outside_feature_mean  # (batch, 1), torch.float32

            candidate_features.append(
                torch.cat(
                    (
                        q_normalized,
                        centered_q_squared,
                        extract_power_mean,
                        extract_power_std,
                        diff_extract_outside_power_mean,
                        boundary_energy,
                        alpha,
                        corr_real_mean_extract,
                        corr_imag_mean_extract,
                        corr_abs_mean_extract,
                    ),
                    dim=1,
                )
            )
            q_prior_features.append(q_prior_feature)
            e_priors.append(e_prior)

        return (
            torch.stack(candidate_features, dim=1),  # (batch, num_q_classes, num_candidate_features), torch.float32
            torch.cat(q_prior_features, dim=1),  # (batch, num_q_classes), torch.float32
            torch.cat(e_priors, dim=1).clamp(min=self.eps_margin, max=self.ts - self.eps_margin),  # (batch, num_q_classes), torch.float32
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate one shared delay from a batch of fading-block waveforms."""
        # batch_size, nb, sequence_length = x.shape
        # x_flat = x.reshape(batch_size * nb, sequence_length)  # (batch_size * nb, sequence_length), torch.complex64

        # # generate features.
        # sequence_features = self.input_projection(
        #     torch.stack((x_flat.real, x_flat.imag), dim=1)
        # )  # (batch_size * nb, hidden_channels, sequence_length), torch.float32
        # sequence_features = self.shift_residual_conv_blocks(sequence_features)  # (batch_size * nb, hidden_channels, sequence_length), torch.float32
        # # reshape
        # joint_sequence = (
        #     sequence_features
        #     .reshape(batch_size, nb, self.hidden_channels, sequence_length)
        #     .permute(0, 2, 1, 3)
        #     .reshape(batch_size, self.hidden_channels, nb * sequence_length)
        # )  # (batch_size, hidden_channels, nb * sequence_length), torch.float32

        # fused_features = self.attention_pool(joint_sequence)  # (batch_size, hidden_channels), torch.float32

        candidate_features, q_prior_features, e_prior = self._build_candidate_features(
            x
        )  # (batch_size, num_q_classes, num_candidate_features), (batch_size, num_q_classes), (batch_size, num_q_classes)
        candidate_embeddings = self.candidate_projection(candidate_features)  # (batch_size, num_q_classes, hidden_channels), torch.float32
        learned_q_logits = self.q_head(candidate_embeddings).squeeze(-1)  # (batch_size, num_q_classes), torch.float32

        q_logits = learned_q_logits + self.q_prior_scale * torch.tanh(q_prior_features)
        q_probabilities = torch.softmax(q_logits, dim=-1)

        q_values = (
            torch.arange(
                self.num_q_classes,
                device=x.device,
                dtype=q_probabilities.dtype,
            ).unsqueeze(0)
            * self.ts
        )

        global_eps_residual = torch.tanh(self.fused_eps_output(e_prior)) * self.ts
        candidate_eps = e_prior + torch.tanh(self.candidate_eps_head(candidate_embeddings).squeeze(-1)) * self.ts
        candidate_eps = (candidate_eps + global_eps_residual).clamp(
            min=self.eps_margin,
            max=self.ts - self.eps_margin,
        )

        candidate_delays = q_values + candidate_eps
        return torch.sum(
            q_probabilities * candidate_delays,
            dim=-1,
            keepdim=True,
        ).clamp(min=self.delay_margin, max=self.dmax - self.delay_margin)
