"""Rayleigh block-fading channel and signal-processing helpers."""

import math
from collections.abc import Mapping
from typing import TypeAlias, cast

import torch
import torch.nn.functional as F

ReceiverOption: TypeAlias = str | float | torch.Tensor
ForwardMetadata: TypeAlias = dict[str, torch.Tensor]


class RayleighBlockFadingChannel:
    """Simulate a Rayleigh block-fading channel with optional timing delay."""

    def __init__(
        self,
        nb: int,
        tp: float,
        ts: float,
        N: int,
        dmax: float,
        is_delay: bool,
        receiver_mmse_regularization: float,
        *,
        normalize_transmit_blocks: bool = True,
    ) -> None:
        self.nb = nb
        self.tp = tp
        self.ts = ts
        self.N = N
        self.dmax = dmax
        self.is_delay = is_delay
        self.receiver_mmse_regularization = receiver_mmse_regularization
        # Disable when the transmitter already enforces batch-average power,
        # so the channel preserves differences in symbol and block energies.
        self.normalize_transmit_blocks = normalize_transmit_blocks

    def receiver_resample(
        self,
        received_signal: torch.Tensor,
        predicted_delay: torch.Tensor,
        **method_kwargs: ReceiverOption,
    ) -> torch.Tensor:
        """
        Recover delay-corrected samples using only receiver-side information.

        Args:
            received_signal: Upsampled received signal with shape
                (batch_size, nb, M), complex dtype.
            predicted_delay: Estimated delay, shape (batch_size, 1),
                (batch_size, nb, 1), or the sync component dict.
            method_kwargs: Receiver-local options. Supported keys are
                ``method`` (``"lmmse"``, ``"symbol_lmmse"``, ``"linear"``,
                or ``"nearest"``), ``mmse_regularization`` for regularized
                inverse methods, and ``sample_phase`` for interpolation methods.

        Returns:
            Complex tensor with shape (batch_size, nb, ns * N). No new noise
            is added because ``received_signal`` already contains the observed
            channel noise.

        Raises:
            TypeError: If the resampling method is not a string.
            ValueError: If an option or input shape is invalid.

        """
        method = method_kwargs.pop("method", "lmmse")
        mmse_regularization = method_kwargs.pop("mmse_regularization", self.receiver_mmse_regularization)
        sample_phase = method_kwargs.pop("sample_phase", 0.5)
        if method_kwargs:
            unknown = ", ".join(sorted(method_kwargs))
            raise ValueError(f"Unsupported receiver_resample method kwargs: {unknown}.")

        if not isinstance(method, str):
            raise TypeError("receiver_resample method must be a string.")

        if isinstance(predicted_delay, Mapping):
            delay_mapping = cast("Mapping[str, torch.Tensor]", predicted_delay)
            predicted_delay = delay_mapping["delay"]

        if received_signal.dim() != 3 or received_signal.shape[1] != self.nb or not torch.is_complex(received_signal):
            raise ValueError("received_signal must have shape (batch_size, nb, M) and complex dtype.")

        method = method.lower()
        if method not in {"lmmse", "symbol_lmmse", "linear", "nearest"}:
            raise ValueError("method must be one of: 'lmmse', 'symbol_lmmse', 'linear', 'nearest'.")

        _, _, received_length = received_signal.shape

        if predicted_delay.dim() == 3 and predicted_delay.shape[-1] == 1:
            predicted_delay = predicted_delay.squeeze(-1)

        if predicted_delay.shape[1] == 1:
            predicted_delay = predicted_delay.expand(-1, self.nb)

        signal_length = received_length - int(self.dmax / self.ts)
        if signal_length <= 0:
            raise ValueError("received_signal is too short for the configured dmax/ts.")
        if not math.isclose(self.N * self.ts, self.tp, rel_tol=1e-5, abs_tol=1e-8):
            raise ValueError("receiver_resample requires a consistent sample grid: N * ts must equal tp. For example, use tp=1, ts=0.25, N=4.")
        if signal_length % self.N != 0:
            raise ValueError("The recoverable signal length must be divisible by N. Use a consistent configuration such as ts = tp / N.")
        ns = signal_length // self.N

        if method == "lmmse":
            return self._receiver_lmmse_resample(
                received_signal=received_signal,
                predicted_delay=predicted_delay,
                ns=ns,
                mmse_regularization=mmse_regularization,
            )
        if method == "symbol_lmmse":
            return self._receiver_symbol_lmmse_resample(
                received_signal=received_signal,
                predicted_delay=predicted_delay,
                ns=ns,
                mmse_regularization=mmse_regularization,
            )

        return self._receiver_interpolation_resample(
            received_signal=received_signal,
            predicted_delay=predicted_delay,
            ns=ns,
            method=method,
            sample_phase=sample_phase,
        )

    def _receiver_interpolation_resample(
        self,
        received_signal: torch.Tensor,
        predicted_delay: torch.Tensor,
        ns: int,
        method: str,
        sample_phase: ReceiverOption,
    ) -> torch.Tensor:
        _, _, received_length = received_signal.shape
        device = received_signal.device
        real_dtype = received_signal.real.dtype

        output_length = ns * self.N
        sample_offsets = torch.arange(output_length, device=device, dtype=real_dtype).view(1, 1, output_length)
        sample_positions = predicted_delay.unsqueeze(-1) / self.ts + float(sample_phase) + sample_offsets
        sample_positions = sample_positions.clamp(0.0, received_length - 1.0)

        if method == "nearest":
            indices = torch.round(sample_positions).long().clamp(0, received_length - 1)
            output = torch.gather(received_signal, dim=-1, index=indices)
            return output

        left_indices = torch.floor(sample_positions).long().clamp(0, received_length - 1)
        right_indices = (left_indices + 1).clamp(0, received_length - 1)
        weights = sample_positions - left_indices.to(real_dtype)

        left = torch.gather(received_signal, dim=-1, index=left_indices)
        right = torch.gather(received_signal, dim=-1, index=right_indices)
        output = (1.0 - weights) * left + weights * right
        return output

    def _receiver_lmmse_resample(
        self,
        received_signal: torch.Tensor,
        predicted_delay: torch.Tensor,
        ns: int,
        mmse_regularization: ReceiverOption,
    ) -> torch.Tensor:
        batch_size, nb, received_length = received_signal.shape
        device = received_signal.device
        real_dtype = received_signal.real.dtype
        flat_blocks = batch_size * nb
        output_length = ns * self.N

        y = received_signal.reshape(flat_blocks, received_length)
        delay = predicted_delay.reshape(flat_blocks)
        delay_span_samples = int(self.dmax / self.ts)
        q = torch.floor((delay + 1e-10) / self.ts).long()
        q = q.clamp(0, max(delay_span_samples - 1, 0))
        fractional = ((delay - q.to(real_dtype) * self.ts) / self.ts).clamp(0.0, 1.0)

        regularization = self._receiver_mmse_regularization(
            mmse_regularization=mmse_regularization,
            batch_size=batch_size,
            nb=nb,
            device=device,
            real_dtype=real_dtype,
        ).reshape(flat_blocks)

        sample_offsets = torch.arange(output_length, device=device, dtype=torch.long).view(1, output_length)
        left_indices = q.view(flat_blocks, 1) + sample_offsets
        right_indices = left_indices + 1
        left_indices = left_indices.clamp(0, received_length - 1)
        right_indices = right_indices.clamp(0, received_length - 1)

        weight_left = (1.0 - fractional).view(flat_blocks, 1)
        weight_right = fractional.view(flat_blocks, 1)
        y_left = torch.gather(y, dim=-1, index=left_indices)
        y_right = torch.gather(y, dim=-1, index=right_indices)
        rhs = weight_left.to(y.dtype) * y_left + weight_right.to(y.dtype) * y_right

        weight_left = weight_left.squeeze(-1)
        weight_right = weight_right.squeeze(-1)
        diagonal = weight_left.square() + weight_right.square() + regularization + torch.tensor(1e-8, dtype=real_dtype, device=device)
        off_diagonal = weight_left * weight_right

        output = self._solve_batched_toeplitz_tridiagonal(
            diagonal=diagonal,
            off_diagonal=off_diagonal,
            rhs=rhs,
        )
        return output.reshape(batch_size, nb, output_length)

    def _receiver_symbol_lmmse_resample(
        self,
        received_signal: torch.Tensor,
        predicted_delay: torch.Tensor,
        ns: int,
        mmse_regularization: ReceiverOption,
    ) -> torch.Tensor:
        batch_size, nb, received_length = received_signal.shape
        device = received_signal.device
        real_dtype = received_signal.real.dtype
        flat_blocks = batch_size * nb
        output_length = ns * self.N

        y = received_signal.reshape(flat_blocks, received_length)
        delay = predicted_delay.reshape(flat_blocks)
        delay_span_samples = int(self.dmax / self.ts)
        q = torch.floor((delay + 1e-10) / self.ts).long()
        q = q.clamp(0, max(delay_span_samples - 1, 0))
        fractional = ((delay - q.to(real_dtype) * self.ts) / self.ts).clamp(0.0, 1.0)  # normalized fractional delay in [0, 1]

        regularization = self._receiver_mmse_regularization(
            mmse_regularization=mmse_regularization,
            batch_size=batch_size,
            nb=nb,
            device=device,
            real_dtype=real_dtype,
        ).reshape(flat_blocks)

        segment_offsets = torch.arange(output_length + 1, device=device, dtype=torch.long).view(1, output_length + 1)
        segment_indices = q.view(flat_blocks, 1) + segment_offsets
        segment_indices = segment_indices.clamp(0, received_length - 1)
        y_segment = torch.gather(y, dim=-1, index=segment_indices)

        symbol_starts = torch.arange(ns, device=device, dtype=torch.long) * self.N
        weight_left = 1.0 - fractional
        weight_right = fractional
        rhs = weight_left.view(flat_blocks, 1).to(y.dtype) * y_segment[:, symbol_starts]
        rhs = rhs + weight_right.view(flat_blocks, 1).to(y.dtype) * y_segment[:, symbol_starts + self.N]

        if self.N > 1:
            interior_offsets = torch.arange(1, self.N, device=device, dtype=torch.long)
            interior_indices = symbol_starts.view(ns, 1) + interior_offsets.view(1, self.N - 1)
            interior = y_segment[:, interior_indices.reshape(-1)].reshape(flat_blocks, ns, self.N - 1)
            rhs = rhs + interior.sum(dim=-1)

        diagonal = (self.N - 1) + weight_left.square() + weight_right.square() + regularization + torch.tensor(1e-8, dtype=real_dtype, device=device)
        off_diagonal = weight_left * weight_right

        symbol_estimate = self._solve_batched_toeplitz_tridiagonal(
            diagonal=diagonal,
            off_diagonal=off_diagonal,
            rhs=rhs,
        )
        output = symbol_estimate.repeat_interleave(self.N, dim=1)
        return output.reshape(batch_size, nb, output_length)

    @staticmethod
    def _solve_batched_toeplitz_tridiagonal(
        diagonal: torch.Tensor,
        off_diagonal: torch.Tensor,
        rhs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Solve batched tridiagonal systems with constant per-row bands.

        Returns:
            The solution tensor for every tridiagonal system in the batch.

        """
        length = rhs.shape[-1]
        diagonal_complex = diagonal.to(rhs.dtype)
        off_diagonal_complex = off_diagonal.to(rhs.dtype)

        if length == 1:
            return rhs / diagonal_complex.unsqueeze(-1)

        c_primes = []
        d_primes = []

        denom = diagonal
        c_prime = off_diagonal / denom
        d_prime = rhs[:, 0] / diagonal_complex
        c_primes.append(c_prime)
        d_primes.append(d_prime)

        for index in range(1, length):
            denom = diagonal - off_diagonal * c_primes[index - 1]
            denom_complex = denom.to(rhs.dtype)
            d_prime = (rhs[:, index] - off_diagonal_complex * d_primes[index - 1]) / denom_complex
            d_primes.append(d_prime)

            if index < length - 1:
                c_primes.append(off_diagonal / denom)

        solutions = [d_primes[-1]]
        for index in range(length - 2, -1, -1):
            solutions.append(d_primes[index] - c_primes[index].to(rhs.dtype) * solutions[-1])
        solutions.reverse()

        return torch.stack(solutions, dim=1)

    @staticmethod
    def _receiver_mmse_regularization(
        mmse_regularization: ReceiverOption,
        batch_size: int,
        nb: int,
        device: torch.device,
        real_dtype: torch.dtype,
    ) -> torch.Tensor:
        regularization = torch.as_tensor(mmse_regularization, dtype=real_dtype, device=device)

        while regularization.dim() > 2 and regularization.shape[-1] == 1:
            regularization = regularization.squeeze(-1)

        if regularization.dim() == 0 or regularization.numel() == 1:
            regularization = regularization.reshape(1).expand(batch_size, nb)
        elif regularization.dim() == 1:
            if regularization.numel() == batch_size:
                regularization = regularization.view(batch_size, 1).expand(-1, nb)
            elif regularization.numel() == batch_size * nb:
                regularization = regularization.view(batch_size, nb)
            else:
                raise ValueError("mmse_regularization must be scalar, per batch, or per fading block.")
        elif regularization.dim() == 2:
            if regularization.shape == (batch_size, 1):
                regularization = regularization.expand(-1, nb)
            elif regularization.shape != (batch_size, nb):
                raise ValueError(f"mmse_regularization must have shape ({batch_size}, 1) or ({batch_size}, {nb}).")
        else:
            raise ValueError("mmse_regularization must be scalar, per batch, or per fading block.")

        return regularization.clamp_min(0.0)

    def resample(
        self,
        x: torch.Tensor,
        h_true: torch.Tensor | ForwardMetadata,
        SNR_db_low: float,
        SNR_db_high: float,
    ) -> torch.Tensor:
        """
        Upsample transmitted blocks for the perfect-sync decoder path.

        This path assumes the true timing offset has already been removed, so it
        only constructs the upsampled per-block waveform, applies the known
        fading coefficient, and adds channel noise.
        Transmit blocks are normalized only when ``normalize_transmit_blocks``
        is enabled; otherwise the input symbol amplitudes are preserved.

        Args:
            x (torch.Tensor): transmitted signal, shape (batch_size, n),
                complex dtype.
            h_true (torch.Tensor | ForwardMetadata): fading coefficients with shape
                (batch_size, self.nb, 1), (batch_size, self.nb), or broadcastable
                scalar/per-batch variants.
            SNR_db_low: lower SNR bound for sampled noise.
            SNR_db_high: upper SNR bound for sampled noise.

        Returns:
            torch.Tensor: complex tensor with shape
                (batch_size, self.nb, (n // self.nb) * self.N).

        Raises:
            ValueError: If the input shape or configured sample grid is invalid.

        """
        if x.dim() != 2 or not torch.is_complex(x):
            raise ValueError("x must have shape (batch_size, n) and complex dtype.")

        batch_size, n = x.shape
        if n % self.nb != 0:
            raise ValueError("x length must be divisible by the number of fading blocks.")

        device = x.device
        real_dtype = x.real.dtype
        complex_dtype = x.dtype
        ns = n // self.nb
        if not math.isclose(self.N * self.ts, self.tp, rel_tol=1e-5, abs_tol=1e-8):
            raise ValueError("resample requires a consistent sample grid: N * ts must equal tp.")

        # pylint: disable-next=too-many-branches
        def _as_block_complex(
            value: torch.Tensor | complex,
            name: str,
        ) -> torch.Tensor:
            # Normalize h_true to (batch_size, self.nb) while keeping the
            # complex dtype, so each fading block can be processed uniformly.
            if isinstance(value, torch.Tensor):
                value = value.to(device=device, dtype=complex_dtype)
            else:
                value = torch.tensor(value, device=device, dtype=complex_dtype)

            while value.dim() > 2 and value.shape[-1] == 1:
                value = value.squeeze(-1)

            if value.dim() == 0:
                value = value.view(1, 1).expand(batch_size, self.nb)
            elif value.dim() == 1:
                if value.numel() == batch_size:
                    value = value.view(batch_size, 1).expand(-1, self.nb)
                elif value.numel() == batch_size * self.nb:
                    value = value.view(batch_size, self.nb)
                else:
                    raise ValueError(f"{name} must have batch size {batch_size} or shape ({batch_size}, {self.nb}).")
            elif value.dim() == 2:
                if value.shape[0] != batch_size:
                    raise ValueError(f"{name} batch dimension must be {batch_size}.")
                if value.shape[1] == 1:
                    value = value.expand(-1, self.nb)
                elif value.shape[1] != self.nb:
                    raise ValueError(f"{name} block dimension must be 1 or {self.nb}.")
            else:
                raise ValueError(f"{name} must have shape (batch,), (batch, 1), or (batch, {self.nb}).")

            return value

        h_true = _as_block_complex(h_true, "h_true").unsqueeze(-1)

        output_length = ns * self.N
        x_resampled = torch.zeros(
            (batch_size, self.nb, output_length),
            dtype=complex_dtype,
            device=device,
        )

        q_zero = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        insert_indices = torch.arange(0, output_length, self.N, device=device)
        for block_idx in range(self.nb):
            start_idx = block_idx * ns
            end_idx = (block_idx + 1) * ns
            x_block = x[:, start_idx:end_idx]

            if self.normalize_transmit_blocks:
                energies = torch.mean(torch.abs(x_block) ** 2, dim=1, keepdim=True)
                energies = torch.clamp(energies, min=1e-12)
                x_block = x_block / torch.sqrt(energies)

            x_upsampled = torch.zeros((batch_size, output_length), dtype=complex_dtype, device=device)
            for repeat_idx in range(self.N):
                x_upsampled[:, insert_indices + repeat_idx] = x_block

            x_inserted = torch.zeros((batch_size, output_length), dtype=complex_dtype, device=device)
            x_inserted = insert_tensor_in_place(
                x_inserted,
                q_zero,
                x_upsampled,
                self.N,
                device,
            )
            x_resampled[:, block_idx, :] = h_true[:, block_idx, :] * x_inserted

        sigma_low = _snr_db_to_complex_noise_std(SNR_db_low, device=device, dtype=real_dtype)
        sigma_high = _snr_db_to_complex_noise_std(SNR_db_high, device=device, dtype=real_dtype)
        noise_shape = x_resampled.shape
        this_sigma = torch.rand(noise_shape, dtype=real_dtype, device=device) * (sigma_low - sigma_high) + sigma_high
        noise = torch.complex(
            this_sigma * torch.randn(noise_shape, dtype=real_dtype, device=device),
            this_sigma * torch.randn(noise_shape, dtype=real_dtype, device=device),
        )

        return x_resampled + noise

    def forward(
        self,
        X: torch.Tensor,
        delay_true: torch.Tensor,
        SNR_db_low: float,
        SNR_db_high: float,
        return_metadata: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | ForwardMetadata]:
        """
        Apply fading, timing delay, and additive noise to a signal.

        Transmit blocks are normalized only when ``normalize_transmit_blocks``
        is enabled; otherwise the input symbol amplitudes are preserved.

        Returns:
            The received signal and either fading coefficients or channel metadata.

        Raises:
            ValueError: If the signal length or delay shape is invalid.

        """
        # X shape: (batch_size, n), dtype: torch.complex64
        batch_size, n = X.shape
        device = X.device

        if n % self.nb != 0:
            raise ValueError("X length must be divisible by the number of fading blocks.")

        real_dtype = X.real.dtype

        # Normalize delay_true to one delay per fading block. Inputs with shape
        # (batch_size, 1) keep the old shared-delay behavior by broadcasting the
        # same delay to all blocks. Inputs with shape (batch_size, nb, 1) give
        # independent delays to each block.

        # delay_true = delay_true.to(device=device, dtype=real_dtype)
        while delay_true.dim() > 2 and delay_true.shape[-1] == 1:
            delay_true = delay_true.squeeze(-1)

        if delay_true.dim() == 1:
            if delay_true.numel() != batch_size:
                raise ValueError(f"delay_true batch dimension must be {batch_size}.")
            delay_true = delay_true.view(batch_size, 1).expand(-1, self.nb)
        elif delay_true.dim() == 2:
            if delay_true.shape[0] != batch_size:
                raise ValueError(f"delay_true batch dimension must be {batch_size}.")
            if delay_true.shape[1] == 1:
                delay_true = delay_true.expand(-1, self.nb)
            elif delay_true.shape[1] != self.nb:
                raise ValueError(f"delay_true block dimension must be 1 or {self.nb}.")
        else:
            raise ValueError(f"delay_true must have shape (batch_size,), (batch_size, 1), or (batch_size, {self.nb}, 1).")

        # Calculate integer and fractional delay per block.
        eps = delay_true % self.ts  # Shape: (batch_size, nb)
        q = torch.floor((delay_true + 1e-10) / self.ts).long()
        num_delay_classes = int(self.dmax / self.ts) + 1
        true_starting_point_onehot = F.one_hot(q, num_classes=num_delay_classes).to(device)

        # Calculate the block length - ns
        ns = n // self.nb

        # total length of the received sequence (per block)
        M = int(self.dmax / self.ts + (ns * self.tp) / (self.ts))
        # Initialize a tensor array to store received signals for each block (upsampled)
        received_signal = torch.zeros((batch_size, self.nb, M), dtype=torch.complex64, device=device)
        h = torch.randn((batch_size, self.nb, 1), dtype=torch.complex64, device=device)
        sigma_low = _snr_db_to_complex_noise_std(SNR_db_low, device=device, dtype=real_dtype)
        sigma_high = _snr_db_to_complex_noise_std(SNR_db_high, device=device, dtype=real_dtype)
        noise_shape = (batch_size, self.nb, M)
        this_sigma = torch.rand(noise_shape, dtype=real_dtype, device=device) * (sigma_low - sigma_high) + sigma_high
        noise = torch.complex(
            this_sigma * torch.randn(noise_shape, dtype=real_dtype, device=device),
            this_sigma * torch.randn(noise_shape, dtype=real_dtype, device=device),
        )

        for block_idx in range(self.nb):
            q_l = q[:, block_idx : block_idx + 1]
            eps_l = eps[:, block_idx : block_idx + 1]

            # Extract this fading block of transmitted symbols for all batches.
            start_idx = block_idx * ns
            end_idx = (block_idx + 1) * ns
            X_block = X[:, start_idx:end_idx]  # Shape: (batch_size, block_len)

            # Preserve the transmitter's power allocation when block
            # normalization is disabled (e.g. batch-average power).
            if self.normalize_transmit_blocks:
                energies = torch.mean(torch.abs(X_block) ** 2, dim=1, keepdim=True)
                energies = torch.clamp(energies, min=1e-12)
                X_block = X_block / torch.sqrt(energies)

            # Upsample the signal by factor N by repeating each symbol N times.
            upsampled_length = ns * self.N
            X_upsampled = torch.zeros((batch_size, upsampled_length), dtype=torch.complex64, device=device)
            indices = torch.arange(0, upsampled_length, self.N, device=device)
            for repeat_idx in range(self.N):
                X_upsampled[:, indices + repeat_idx] = X_block

            # Insert this block at its own integer delay q_l and at q_l + 1.
            # The fractional delay eps_l linearly combines these two shifted
            # versions, matching the existing triangular/rectangular pulse model.
            X_N_Q_l = torch.zeros((batch_size, M), dtype=torch.complex64, device=device)
            X_N_Q_l = insert_tensor_in_place(X_N_Q_l, q_l, X_upsampled, self.N, device)

            X_N_Q_l_plus_1 = torch.zeros((batch_size, M), dtype=torch.complex64, device=device)
            X_N_Q_l_plus_1 = insert_tensor_in_place(X_N_Q_l_plus_1, q_l + 1, X_upsampled, self.N, device)

            X_interference = X_N_Q_l * (1 - eps_l / self.ts) + X_N_Q_l_plus_1 * (eps_l / self.ts)

            received_signal[:, block_idx, :] = h[:, block_idx, :] * X_interference + noise[:, block_idx, :]

        output_signal = received_signal

        if return_metadata:
            metadata = {
                "delay_true": delay_true,
                "eps": eps,
                "q": q,
                "true_starting_point_onehot": true_starting_point_onehot,
                "h": h,
            }
            return output_signal, metadata

        return output_signal, h


def _snr_db_to_complex_noise_std(
    snr_db: torch.Tensor | complex,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    snr_db = torch.as_tensor(snr_db, dtype=dtype, device=device)
    base = torch.tensor(10.0, dtype=dtype, device=device)
    return torch.sqrt(0.5 * torch.pow(base, -snr_db / 10.0))


def starting_point_onehot(q: torch.Tensor, dmax: float, ts: float) -> torch.Tensor:
    """
    Function to generate the one-hot encoded starting point for each batch.

    Args:
        q (torch.Tensor): (batch_size, ).
        dmax (float): The maximum delay.
        ts (float): The sampling period.

    Returns:
        torch.Tensor: The one-hot encoded starting point for each batch.

    """
    device = q.device

    # Number of classes must be specified for one-hot encoding
    num_classes = int(dmax / ts) + 1
    # Perform one-hot encoding, one hot tensor of shape (batch_size, num_classes)
    one_hot = F.one_hot(q, num_classes=num_classes)

    return one_hot.squeeze(1).to(device)


def insert_tensor_in_place(
    x_n_q_l: torch.Tensor,
    q: torch.Tensor,
    x_upsampled: torch.Tensor,
    N: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Function to insert the upsampled signal into the correct position of the tensor.

    Args:
        x_n_q_l: The tensor to insert the upsampled signal into.
        q: The delay value for each batch element.
        x_upsampled: The upsampled signal.
        N: The upsampling factor.
        device: Device containing the tensors.

    Returns:
        torch.Tensor: The tensor with the upsampled signal inserted.

    """
    batch_size, insert_length = x_upsampled.size()
    # Create a batch index
    batch_indices = torch.arange(batch_size).unsqueeze(1).to(device)
    # Create column indices using q
    column_indices = q + torch.arange(insert_length).to(device)
    # Insert X into X_N_Q_l using advanced indexing
    scale = torch.sqrt(torch.tensor(1 / N, dtype=x_upsampled.real.dtype, device=device))
    x_n_q_l[batch_indices, column_indices] = x_upsampled * scale

    return x_n_q_l
