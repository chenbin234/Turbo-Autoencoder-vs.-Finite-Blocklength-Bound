"""Channel-coded four-block Rayleigh baseline for Sionna 2.0.1/PyTorch.

This is the imperfect-synchronization/imperfect-CSI baseline. Timing and the
per-block fading coefficients are estimated from an upsampled Zadoff-Chu
preamble with the maximum-likelihood method described by Oguz et al.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

import numpy as np
import torch
from loguru import logger
from sionna import phy
from sionna.phy import Block
from sionna.phy.channel import AWGN
from sionna.phy.fec.ldpc import LDPC5GDecoder, LDPC5GEncoder
from sionna.phy.fec.polar import Polar5GDecoder, Polar5GEncoder
from sionna.phy.mapping import BinarySource, Constellation, Demapper, Mapper
from sionna.phy.utils import ebnodb2no, sim_ber

import wandb

np.set_printoptions(
    linewidth=250,
    threshold=1200,
    edgeitems=10,
    suppress=False,
    precision=6,
)

logger.remove()
logger.add(sys.stderr, level=os.environ.get("LOGURU_LEVEL", "ERROR"))
logger.add("./src/baseline/log_file.log", level="ERROR", mode="w")


class RayleighBlockFadingChannel:
    """Rectangular-pulse Rayleigh block-fading channel with random delay.

    By default, one independent complex Rayleigh coefficient is drawn for each
    fading block. Debug mode uses unit-magnitude blocks and a final deep fade.
    A propagation delay may either be shared by all blocks in a packet or drawn
    independently for each block.
    """

    def __init__(
        self,
        nb: int,
        tp: float,
        ts: float,
        n_up: int,
        preamble_length: int,
        payload_length: int,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        self.nb = nb
        self.tp = tp
        self.ts = ts
        self.n_up = n_up
        self.preamble_length = preamble_length
        self.payload_length = payload_length
        self.device = torch.device(device or phy.config.device)
        self.awgn_noise = AWGN(device=str(self.device))
        assert int(tp / ts) == n_up, "Incompatible parameters tp, ts, and n_up."

    @staticmethod
    def insert_tensor_in_place(
        output: torch.Tensor,
        q: torch.Tensor,
        x_upsampled: torch.Tensor,
        n_up: int,
    ) -> torch.Tensor:
        """Return output with every row of x_upsampled inserted at q.

        Despite the historical method name, this implementation does not mutate
        its input. torch.scatter keeps this operation device-native and works on
        both CPU and GPU.
        """
        batch_size, insert_length = x_upsampled.shape
        q = torch.as_tensor(q, dtype=torch.int64, device=x_upsampled.device).reshape(-1)
        if q.numel() != batch_size:
            raise ValueError("q must contain one insertion index per batch sample.")

        columns = (
            q[:, None]
            + torch.arange(
                insert_length,
                dtype=torch.int64,
                device=x_upsampled.device,
            )[None, :]
        )
        if not torch.compiler.is_compiling() and (bool(torch.any(columns < 0).item()) or bool(torch.any(columns >= output.shape[1]).item())):
            raise ValueError("The delayed signal does not fit in the output tensor.")

        scale = torch.sqrt(
            torch.tensor(1.0 / n_up, dtype=x_upsampled.real.dtype, device=x_upsampled.device),
        ).to(x_upsampled.dtype)
        return output.scatter(1, columns, x_upsampled * scale)

    @staticmethod
    def create_delayed_upsampled_signals(
        x_block: torch.Tensor,
        delay: torch.Tensor,
        m: int,
        n_up: int,
        ts: float,
        block_idx: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create the two adjacent sampled representations of a delayed block."""
        batch_size = x_block.shape[0]
        delay = torch.as_tensor(delay, dtype=x_block.real.dtype, device=x_block.device)
        eps = torch.remainder(delay, ts)
        q = torch.floor((delay + 1e-10) / ts).to(torch.int64)
        x_upsampled = torch.repeat_interleave(x_block, n_up, dim=1)

        if block_idx is not None:
            logger.debug("Upsampled block {} shape: {}", block_idx, x_upsampled.shape)

        shape = (batch_size, m)
        x_n_q_l = RayleighBlockFadingChannel.insert_tensor_in_place(
            torch.zeros(shape, dtype=torch.complex64, device=x_block.device),
            q,
            x_upsampled,
            n_up,
        )
        x_n_q_l_plus_1 = RayleighBlockFadingChannel.insert_tensor_in_place(
            torch.zeros(shape, dtype=torch.complex64, device=x_block.device),
            q + 1,
            x_upsampled,
            n_up,
        )

        x_concat = torch.stack((x_n_q_l, x_n_q_l_plus_1), dim=2)
        x_concat_h = x_concat.transpose(1, 2).conj()
        eps_matrix_row = torch.stack(
            (1.0 - eps[:, None] / ts, eps[:, None] / ts),
            dim=2,
        ).to(torch.complex64)
        eps_matrix_column = eps_matrix_row.transpose(1, 2)
        return x_concat, x_concat_h, eps_matrix_row, eps_matrix_column

    @staticmethod
    def upsample_preamble_zeropad_payload(
        x_block: torch.Tensor,
        n_up: int,
        preamble_len: int,
        payload_len: int,
    ) -> torch.Tensor:
        """Repeat preamble samples and zero-stuff payload samples."""
        batch_size = x_block.shape[0]
        preamble = x_block[:, :preamble_len]
        payload = x_block[:, preamble_len:]
        if payload.shape[1] != payload_len:
            raise ValueError("payload_len does not match x_block.")

        preamble_upsampled = torch.repeat_interleave(preamble, n_up, dim=1)
        payload_reshaped = payload.reshape(batch_size, payload_len, 1)
        payload_with_zeros = torch.cat(
            (
                payload_reshaped,
                torch.zeros(
                    (batch_size, payload_len, n_up - 1),
                    dtype=x_block.dtype,
                    device=x_block.device,
                ),
            ),
            dim=2,
        )
        payload_upsampled = payload_with_zeros.reshape(batch_size, payload_len * n_up)
        return torch.cat((preamble_upsampled, payload_upsampled), dim=1)

    def as_block_real(
        self,
        value: float | torch.Tensor,
        name: str,
        batch_size: int,
        real_dtype: torch.dtype,
        device: str | torch.device,
    ) -> torch.Tensor:
        """Normalize scalar, packet, or block real values to [batch, nb]."""
        value = torch.as_tensor(value, dtype=real_dtype, device=device)
        while value.ndim > 2 and value.shape[-1] == 1:
            value = value.squeeze(-1)

        if value.ndim == 0:
            return value.reshape(1, 1).expand(batch_size, self.nb)
        if value.ndim == 1:
            if value.shape[0] != batch_size:
                raise ValueError(f"{name} batch dimension must match x.")
            return value[:, None].expand(-1, self.nb)
        if value.ndim == 2:
            if value.shape[0] != batch_size:
                raise ValueError(f"{name} batch dimension must match x.")
            if value.shape[1] == 1:
                return value.expand(-1, self.nb)
            if value.shape[1] != self.nb:
                raise ValueError(f"{name} block dimension must be 1 or {self.nb}.")
            return value
        raise ValueError(f"{name} must have shape [batch], [batch, 1], or [batch, nb].")

    def _as_block_complex(
        self,
        value: complex | torch.Tensor,
        name: str,
        batch_size: int,
        complex_dtype: torch.dtype,
        device: str | torch.device,
    ) -> torch.Tensor:
        """Normalize scalar, packet, or block complex values to [batch, nb]."""
        value = torch.as_tensor(value, dtype=complex_dtype, device=device)
        while value.ndim > 2 and value.shape[-1] == 1:
            value = value.squeeze(-1)

        if value.ndim == 0:
            return value.reshape(1, 1).expand(batch_size, self.nb)
        if value.ndim == 1:
            if value.shape[0] != batch_size:
                raise ValueError(f"{name} batch dimension must match x.")
            return value[:, None].expand(-1, self.nb)
        if value.ndim == 2:
            if value.shape[0] != batch_size:
                raise ValueError(f"{name} batch dimension must match x.")
            if value.shape[1] == 1:
                return value.expand(-1, self.nb)
            if value.shape[1] != self.nb:
                raise ValueError(f"{name} block dimension must be 1 or {self.nb}.")
            return value
        raise ValueError(f"{name} must have shape [batch], [batch, 1], or [batch, nb].")

    def resample(
        self,
        x: torch.Tensor,
        predicted_delay: torch.Tensor,
        delay_true: torch.Tensor,
        h_true: torch.Tensor,
        no: float | torch.Tensor | None = None,
        *,
        add_noise: bool = True,
    ) -> torch.Tensor:
        """Resample the faded rectangular-pulse signal at predicted delays."""
        if isinstance(predicted_delay, dict):
            predicted_delay = predicted_delay["delay"]

        x = torch.as_tensor(x, dtype=torch.complex64, device=self.device)
        if x.ndim == 3:
            if x.shape[1] != self.nb:
                raise ValueError(f"x block dimension must match nb={self.nb}.")
            x_blocks = x
        elif x.ndim == 2:
            if x.shape[1] % self.nb:
                raise ValueError("x length must be divisible by nb.")
            x_blocks = x.reshape(x.shape[0], self.nb, x.shape[1] // self.nb)
        else:
            raise ValueError("x must have shape [batch, n] or [batch, nb, symbols_per_block].")

        batch_size, _, ns = x_blocks.shape
        real_dtype = x.real.dtype
        complex_dtype = x.dtype
        device = x.device
        predicted_delay = self.as_block_real(predicted_delay, "predicted_delay", batch_size, real_dtype, device)
        delay_true = self.as_block_real(delay_true, "delay_true", batch_size, real_dtype, device)
        h_true = self._as_block_complex(h_true, "h_true", batch_size, complex_dtype, device)[:, :, None]

        tp = torch.tensor(self.tp, dtype=real_dtype, device=device)
        faded_blocks = x_blocks * h_true
        prefix_integral = torch.cat(
            (
                torch.zeros((batch_size, self.nb, 1), dtype=complex_dtype, device=device),
                torch.cumsum(faded_blocks, dim=-1) * tp.to(complex_dtype),
            ),
            dim=-1,
        )

        def integral_at(sample_times: torch.Tensor) -> torch.Tensor:
            sample_times = torch.clamp(sample_times, min=0.0, max=ns * self.tp)
            positions = sample_times / tp
            left_indices = torch.floor(positions).to(torch.int64).clamp(0, ns)
            fractions = (positions - left_indices.to(real_dtype)).clamp(0.0, 1.0)
            base = torch.gather(prefix_integral, 2, left_indices)
            symbols = torch.gather(faded_blocks, 2, left_indices.clamp(0, ns - 1))
            return base + (fractions * tp).to(complex_dtype) * symbols

        block_offsets = predicted_delay - delay_true
        sample_grid = torch.arange(ns, dtype=real_dtype, device=device) * tp
        window_start = block_offsets[:, :, None] + sample_grid[None, None, :]
        window_end = window_start + tp
        x_resampled = (integral_at(window_end) - integral_at(window_start)) / torch.sqrt(tp).to(complex_dtype)
        x_resampled = x_resampled.reshape(batch_size, self.nb * ns)

        if add_noise and no is not None:
            x_resampled = self.awgn_noise(x_resampled, no)
        return x_resampled

    def __call__(
        self,
        x: torch.Tensor,
        no: float | torch.Tensor,
        n_up: int,
        max_delay: float = 12.0,
        shared_delay: bool = True,
        debug_fading: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply random delay, per-block fading, and AWGN.

        With debug_fading=True, the first nb-1 coefficients have magnitude one
        and independent random phases. The final coefficient is a zero-mean
        complex Gaussian with variance E[|h|^2] = 1e-5 (5e-6 per component).
        """
        x = torch.as_tensor(x, dtype=torch.complex64, device=self.device)
        if x.ndim == 3:
            if x.shape[1] != self.nb:
                raise ValueError(f"x block dimension must match nb={self.nb}.")
            x_blocks = x
        elif x.ndim == 2:
            if x.shape[1] % self.nb:
                raise ValueError("x length must be divisible by nb.")
            x_blocks = x.reshape(x.shape[0], self.nb, x.shape[1] // self.nb)
        else:
            raise ValueError("x must have shape [batch, n] or [batch, nb, symbols_per_block].")

        batch_size, _, ns = x_blocks.shape
        m = int(max_delay / self.ts + (ns * self.tp) / self.ts + 1)
        generator = None
        if not torch.compiler.is_compiling():
            generator = phy.config.torch_rng(str(x.device))

        delay_columns = 1 if shared_delay else self.nb
        delays = (
            torch.rand(
                (batch_size, delay_columns),
                dtype=torch.float32,
                device=x.device,
                generator=generator,
            )
            * max_delay
        )
        if shared_delay:
            delays = delays.expand(-1, self.nb)
        eps = torch.remainder(delays, self.ts)
        q = torch.floor((delays + 1e-10) / self.ts).to(torch.int64)

        received_signal_blocks = []
        for block_idx in range(self.nb):
            x_block = x_blocks[:, block_idx, :]
            energy = torch.mean(torch.abs(x_block).square(), dim=1, keepdim=True).clamp_min(1e-12)
            x_block = x_block / torch.sqrt(energy).to(torch.complex64)
            x_upsampled = torch.repeat_interleave(x_block, n_up, dim=1)

            q_l = q[:, block_idx]
            eps_l = eps[:, block_idx]
            x_n_q_l = self.insert_tensor_in_place(
                torch.zeros((batch_size, m), dtype=torch.complex64, device=x.device),
                q_l,
                x_upsampled,
                n_up,
            )
            x_n_q_l_plus_1 = self.insert_tensor_in_place(
                torch.zeros((batch_size, m), dtype=torch.complex64, device=x.device),
                q_l + 1,
                x_upsampled,
                n_up,
            )
            alpha = (1.0 - eps_l / self.ts).to(torch.complex64)[:, None]
            beta = (eps_l / self.ts).to(torch.complex64)[:, None]
            received_signal_blocks.append(x_n_q_l * alpha + x_n_q_l_plus_1 * beta)

        received_signal = torch.stack(received_signal_blocks, dim=1)
        h_real = torch.randn(
            (batch_size, self.nb, 1),
            dtype=torch.float32,
            device=x.device,
            generator=generator,
        ) / np.sqrt(2.0)
        h_imag = torch.randn(
            (batch_size, self.nb, 1),
            dtype=torch.float32,
            device=x.device,
            generator=generator,
        ) / np.sqrt(2.0)
        h_complex = torch.complex(h_real, h_imag)
        if debug_fading:
            good_channels = torch.polar(
                torch.ones_like(h_real[:, :-1, :]),
                torch.angle(h_complex[:, :-1, :]),
            )
            deep_fading_channel = h_complex[:, -1:, :] * np.sqrt(1e-5)
            h_complex = torch.cat((good_channels, deep_fading_channel), dim=1)

        y = self.awgn_noise(received_signal * h_complex, no)
        y = y.reshape(batch_size, self.nb * m)
        if self.nb == 1:
            q = q.squeeze(1)
            eps = eps.squeeze(1)
        return y, h_complex, q, eps


class CodedSystemBlockFading(Block):
    """Sionna 2.x channel-coded QAM/8PSK/16PSK/32PSK/64PSK model with estimated timing and CSI."""

    def __init__(
        self,
        num_bits_per_symbol: int,
        n: int,
        channel_coderate: float | None = None,
        *,
        modulation: str = "qam",
        code_picked: str = "polar_scl8",
        perfect_sync: bool = False,
        perfect_csi: bool = False,
        preamble_length: int = 12,
        preamble_length_per_block: int = 3,
        payload_length: int = 52,
        payload_length_per_block: int = 13,
        num_taps: int = 1,
        nb: int = 1,
        tp: float = 1.0,
        ts: float = 1.0,
        n_up: int = 1,
        max_delay: float = 12.0,
        shared_delay: bool = True,
        debug_fading: bool = False,
        device: str | torch.device | None = None,
        Channel_CODERATE: float | None = None,
        N_up: int | None = None,
    ) -> None:
        super().__init__(device=str(device) if device is not None else None)
        if channel_coderate is None:
            if Channel_CODERATE is None:
                raise TypeError("channel_coderate is required.")
            channel_coderate = Channel_CODERATE
        elif Channel_CODERATE is not None and not np.isclose(channel_coderate, Channel_CODERATE):
            raise ValueError("channel_coderate and legacy Channel_CODERATE disagree.")
        if N_up is not None:
            if n_up not in {1, N_up}:
                raise ValueError("n_up and legacy N_up disagree.")
            n_up = N_up

        if perfect_sync or perfect_csi:
            raise ValueError("This module requires imperfect synchronization and imperfect CSI (perfect_sync=False, perfect_csi=False).")
        if payload_length != nb * payload_length_per_block:
            raise ValueError("payload_length must equal nb * payload_length_per_block.")
        if preamble_length != nb * preamble_length_per_block:
            raise ValueError("preamble_length must equal nb * preamble_length_per_block.")
        if preamble_length_per_block <= 0:
            raise ValueError("Imperfect synchronization requires a non-empty preamble.")

        modulation = modulation.lower()
        if modulation not in {"qam", "8psk", "16psk", "32psk", "64psk"}:
            raise ValueError("modulation must be 'qam' or '8psk' or '16psk' or '32psk' or '64psk'.")
        if modulation == "8psk" and num_bits_per_symbol != 3:
            raise ValueError("8PSK requires num_bits_per_symbol=3.")
        if modulation == "16psk" and num_bits_per_symbol != 4:
            raise ValueError("16PSK requires num_bits_per_symbol=4.")
        if modulation == "32psk" and num_bits_per_symbol != 5:
            raise ValueError("32PSK requires num_bits_per_symbol=5.")
        if modulation == "64psk" and num_bits_per_symbol != 6:
            raise ValueError("64PSK requires num_bits_per_symbol=6.")

        self.modulation = modulation
        self.code_picked = code_picked
        self.num_bits_per_symbol = num_bits_per_symbol
        self.n = n
        self.k = int(n * channel_coderate)
        self.channel_coderate = channel_coderate
        self.Channel_CODERATE = channel_coderate
        self.perfect_sync = perfect_sync
        self.perfect_csi = perfect_csi
        self.preamble_length = preamble_length
        self.preamble_length_per_block = preamble_length_per_block
        self.payload_length = payload_length
        self.payload_length_per_block = payload_length_per_block
        self.num_taps = num_taps
        self.nb = nb
        self.tp = tp
        self.ts = ts
        self.n_up = n_up
        self.N_up = n_up
        self.max_delay = max_delay
        self.shared_delay = shared_delay
        self.debug_fading = debug_fading

        block_device = str(self.device)
        if self.modulation == "8psk":
            # Sionna uses each point's array index as its binary bit label.
            # Counterclockwise labels: 000, 001, 011, 010, 110, 111, 101, 100.
            phase_indices = np.array([0, 1, 3, 2, 7, 6, 4, 5])
            # Generate the 8PSK constellation points.
            # [000, 001, 010, 011, 100, 101, 110, 111] -> [0, pi/4, 3pi/4, pi/2, 7pi/4, 3pi/2, pi, 5pi/4]
            points = np.exp(1j * (np.pi / 4) * phase_indices).astype(np.complex64)
            self.constellation = Constellation(
                "custom",
                3,
                points=points,
                normalize=True,
                device=block_device,
            )
        elif self.modulation == "16psk":
            # Sionna uses each point's array index as its binary bit label.
            # Counterclockwise labels: 0000, 0001, 0011, 0010, 0110, 0111, 0101, 0100, 1100, 1101, 1111, 1110, 1010, 1011, 1001, 1000.
            phase_indices = np.array([0, 1, 3, 2, 7, 6, 4, 5, 15, 14, 12, 13, 8, 9, 11, 10])
            # Generate the 16PSK constellation points.
            points = np.exp(1j * (np.pi / 8) * phase_indices).astype(np.complex64)
            self.constellation = Constellation(
                "custom",
                4,
                points=points,
                normalize=True,
                device=block_device,
            )
        elif self.modulation == "32psk":
            # Sionna uses each point's array index as its binary bit label.
            # Counterclockwise labels follow the five-bit binary-reflected Gray code.
            phase_indices = np.array([
                0,
                1,
                3,
                2,
                7,
                6,
                4,
                5,
                15,
                14,
                12,
                13,
                8,
                9,
                11,
                10,
                31,
                30,
                28,
                29,
                24,
                25,
                27,
                26,
                16,
                17,
                19,
                18,
                23,
                22,
                20,
                21,
            ])
            # Generate the 32PSK constellation points.
            points = np.exp(1j * (np.pi / 16) * phase_indices).astype(np.complex64)
            self.constellation = Constellation(
                "custom",
                5,
                points=points,
                normalize=True,
                device=block_device,
            )
        elif self.modulation == "64psk":
            # Sionna uses each point's array index as its binary bit label.
            # Counterclockwise labels follow the six-bit binary-reflected Gray code.
            phase_indices = np.array([
                0,
                1,
                3,
                2,
                7,
                6,
                4,
                5,
                15,
                14,
                12,
                13,
                8,
                9,
                11,
                10,
                31,
                30,
                28,
                29,
                24,
                25,
                27,
                26,
                16,
                17,
                19,
                18,
                23,
                22,
                20,
                21,
                63,
                62,
                60,
                61,
                56,
                57,
                59,
                58,
                48,
                49,
                51,
                50,
                55,
                54,
                52,
                53,
                32,
                33,
                35,
                34,
                39,
                38,
                36,
                37,
                47,
                46,
                44,
                45,
                40,
                41,
                43,
                42,
            ])
            # Generate the 64PSK constellation points.
            points = np.exp(1j * (np.pi / 32) * phase_indices).astype(np.complex64)
            self.constellation = Constellation(
                "custom",
                6,
                points=points,
                normalize=True,
                device=block_device,
            )
        else:
            self.constellation = Constellation("qam", self.num_bits_per_symbol, device=block_device)
        self.binary_source = BinarySource(device=block_device)
        if code_picked == "ldpc":
            self.encoder = LDPC5GEncoder(self.k, self.n, device=block_device)
            self.decoder = LDPC5GDecoder(self.encoder, num_iter=20, device=block_device)
        elif code_picked == "polar_scl8":
            self.encoder = Polar5GEncoder(self.k, self.n, device=block_device)
            self.decoder = Polar5GDecoder(
                self.encoder,
                dec_type="SCL",
                list_size=8,
                device=block_device,
            )
        elif code_picked == "polar_sc":
            self.encoder = Polar5GEncoder(self.k, self.n, device=block_device)
            self.decoder = Polar5GDecoder(self.encoder, dec_type="SC", device=block_device)
        else:
            raise ValueError("code_picked must be 'ldpc' or 'polar_scl8' or 'polar_sc'.")
        self.mapper = Mapper(constellation=self.constellation, device=block_device)
        self.demapper = Demapper("app", constellation=self.constellation, device=block_device)
        self.rayleigh_fading_channel = RayleighBlockFadingChannel(
            nb=self.nb,
            tp=self.tp,
            ts=self.ts,
            n_up=self.n_up,
            preamble_length=self.preamble_length_per_block,
            payload_length=self.payload_length,
            device=self.device,
        )
        self._zc_seq_base = self.generate_zadoff_chu_pilot(self.preamble_length_per_block, device=self.device)

    def get_zc_seq(self, batch_size: int) -> torch.Tensor:
        """Return the Zadoff-Chu preamble as [batch, nb, preamble_len]."""
        return self._zc_seq_base.reshape(1, 1, self.preamble_length_per_block).expand(batch_size, self.nb, -1)

    @staticmethod
    def generate_zadoff_chu_pilot(
        length: int,
        root: int = 1,
        *,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """Generate a unit-power Zadoff-Chu sequence."""
        if length < 0:
            raise ValueError("The Zadoff-Chu length cannot be negative.")
        if length == 0:
            return torch.empty(0, dtype=torch.complex64, device=device)

        def gcd(a: int, b: int) -> int:
            while b:
                a, b = b, a % b
            return a

        if gcd(root, length) != 1:
            for candidate in range(2, length):
                if gcd(candidate, length) == 1:
                    root = candidate
                    break

        index = torch.arange(length, dtype=torch.float32, device=device)
        phase = -torch.pi * root * index * (index + 1.0) / length if length % 2 == 0 else -torch.pi * root * index.square() / length
        pilot = torch.exp(torch.complex(torch.zeros_like(phase), phase))
        return pilot / torch.sqrt(torch.mean(torch.abs(pilot).square())).to(torch.complex64)

    generateZadoffChuPilot = generate_zadoff_chu_pilot

    def synchronize_and_estimate_channel(
        self,
        y_p: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Estimate integer/fractional delay and per-block fading from pilots."""
        m = int(self.max_delay / self.ts + (self.preamble_length_per_block * self.tp) / self.ts + 1)
        y_p = torch.as_tensor(y_p, dtype=torch.complex64, device=self.device)
        batch_size = y_p.shape[0]
        device = y_p.device

        if y_p.ndim == 3:
            if y_p.shape[1:] != (self.nb, m):
                raise ValueError(f"y_p must have shape [batch, {self.nb}, {m}].")
            y_blocks = y_p
        elif y_p.ndim == 2:
            expected_width = self.nb * m
            if y_p.shape[1] != expected_width:
                raise ValueError(f"y_p width must be nb * M = {expected_width}.")
            y_blocks = y_p.reshape(batch_size, self.nb, m)
        else:
            raise ValueError("y_p must have shape [batch, nb*M] or [batch, nb, M].")

        q_max = int(self.max_delay / self.ts)
        x_blocks = self.get_zc_seq(batch_size)
        y_block_list = [y_blocks[:, block_idx, :].unsqueeze(2) for block_idx in range(self.nb)]
        y_h_block_list = [y_block.transpose(1, 2).conj() for y_block in y_block_list]
        x_block_list = [x_blocks[:, block_idx, :] for block_idx in range(self.nb)]

        def delayed_pilot(
            x_block: torch.Tensor,
            delay: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            return RayleighBlockFadingChannel.create_delayed_upsampled_signals(
                x_block=x_block,
                delay=delay,
                m=m,
                n_up=self.n_up,
                ts=self.ts,
            )

        def build_delay_set(
            x_block: torch.Tensor,
            y_list: list[torch.Tensor],
            y_h_list: list[torch.Tensor],
        ) -> list[torch.Tensor]:
            """Build Oguz delay candidates for one or more fading blocks."""
            delay_set: list[torch.Tensor] = []

            def valid_fractional_delay(
                root: torch.Tensor,
                q_delay: float,
            ) -> torch.Tensor:
                root = root.real
                is_valid = torch.isfinite(root) & (root >= 0.0) & (root <= self.ts) & (q_delay + root <= self.max_delay)
                return torch.where(is_valid, root, torch.zeros_like(root))

            for q_value in range(q_max + 1):
                q_delay = q_value * self.ts
                delay_specific = torch.full(
                    (batch_size,),
                    q_delay,
                    dtype=torch.float32,
                    device=device,
                )
                delay_set.append(delay_specific)

                x_concat, x_concat_h, _, _ = delayed_pilot(
                    x_block,
                    delay_specific,
                )
                product = torch.zeros(
                    (batch_size, 2, 2),
                    dtype=torch.complex64,
                    device=device,
                )
                for y, y_h in zip(y_list, y_h_list, strict=True):
                    x_h_y = torch.matmul(x_concat_h, y)
                    y_h_x = torch.matmul(y_h, x_concat)
                    product = product + torch.matmul(x_h_y, y_h_x)

                a = product[:, 0, 0]
                b = product[:, 1, 0]
                c = product[:, 0, 1]
                d = product[:, 1, 1]
                w3 = a
                w2 = (b + c - 2.0 * a) / self.ts
                w1 = (d - b - c + a) / self.ts**2

                x_h_x = torch.matmul(x_concat_h, x_concat)
                a = x_h_x[:, 0, 0]
                b = x_h_x[:, 1, 0]
                c = x_h_x[:, 0, 1]
                d = x_h_x[:, 1, 1]
                x3 = a
                x2 = (b + c - 2.0 * a) / self.ts
                x1 = (d - b - c + a) / self.ts**2

                z1 = w1 * x2 - w2 * x1
                z2 = 2.0 * w1 * x3 - 2.0 * w3 * x1
                z3 = w2 * x3 - w3 * x2
                sqrt_discriminant = torch.sqrt(z2.square() - 4.0 * z1 * z3)
                e1 = (-z2 + sqrt_discriminant) / (2.0 * z1)
                e2 = (-z2 - sqrt_discriminant) / (2.0 * z1)

                delay_set.extend((q_delay + valid_fractional_delay(e1, q_delay), q_delay + valid_fractional_delay(e2, q_delay)))
            return delay_set

        def objective_function(
            delay: torch.Tensor,
            x: torch.Tensor,
            y: torch.Tensor,
            y_h: torch.Tensor,
        ) -> torch.Tensor:
            """Evaluate the Oguz objective for one block and candidate delay."""
            x_concat, x_concat_h, eps_row, eps_column = delayed_pilot(x, delay)
            x_h_y = torch.matmul(x_concat_h, y)
            y_h_x = torch.matmul(y_h, x_concat)
            product = torch.matmul(x_h_y, y_h_x)
            numerator = (
                torch
                .matmul(
                    torch.matmul(eps_row, product),
                    eps_column,
                )
                .reshape(batch_size)
                .real
            )

            x_h_x = torch.matmul(x_concat_h, x_concat)
            denominator = (
                torch
                .matmul(
                    torch.matmul(eps_row, x_h_x),
                    eps_column,
                )
                .reshape(batch_size)
                .real
            )
            return numerator / denominator

        def objective_matrix(
            delay_set: list[torch.Tensor],
            block_indices: Sequence[int],
        ) -> torch.Tensor:
            columns = []
            for delay in delay_set:
                column = torch.zeros(
                    batch_size,
                    dtype=torch.float32,
                    device=device,
                )
                for block_idx in block_indices:
                    column = column + objective_function(
                        delay,
                        x_block_list[block_idx],
                        y_block_list[block_idx],
                        y_h_block_list[block_idx],
                    )
                columns.append(column)
            return torch.stack(columns, dim=1)

        def best_delay(
            delay_set: list[torch.Tensor],
            objectives: torch.Tensor,
        ) -> torch.Tensor:
            indices = torch.argmax(objectives, dim=1, keepdim=True)
            delays = torch.stack(delay_set, dim=1)
            return torch.gather(delays, 1, indices).squeeze(1)

        def estimate_q_e(
            delay: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            q = torch.floor(delay / self.ts).to(torch.int64)
            e = delay - q.to(torch.float32) * self.ts
            return q, e

        def estimate_h(
            delay: torch.Tensor,
            x: torch.Tensor,
            y: torch.Tensor,
        ) -> torch.Tensor:
            x_concat, x_concat_h, eps_row, eps_column = delayed_pilot(x, delay)
            x_h_y = torch.matmul(x_concat_h, y)
            v_h_y = torch.matmul(eps_row, x_h_y).reshape(batch_size)
            x_h_x = torch.matmul(x_concat_h, x_concat)
            denominator = torch.matmul(
                torch.matmul(eps_row, x_h_x),
                eps_column,
            ).reshape(batch_size)
            return v_h_y / denominator

        if self.shared_delay:
            delay_set = build_delay_set(
                x_block_list[0],
                y_block_list,
                y_h_block_list,
            )
            delay_hat = best_delay(
                delay_set,
                objective_matrix(delay_set, range(self.nb)),
            )
            q_shared, e_shared = estimate_q_e(delay_hat)
            q = q_shared[:, None].expand(-1, self.nb)
            e = e_shared[:, None].expand(-1, self.nb)
            h_hat = torch.stack(
                [
                    estimate_h(
                        delay_hat,
                        x_block_list[block_idx],
                        y_block_list[block_idx],
                    )
                    for block_idx in range(self.nb)
                ],
                dim=1,
            )
        else:
            q_blocks = []
            e_blocks = []
            h_blocks = []
            for block_idx in range(self.nb):
                delay_set = build_delay_set(
                    x_block_list[block_idx],
                    [y_block_list[block_idx]],
                    [y_h_block_list[block_idx]],
                )
                delay_hat = best_delay(
                    delay_set,
                    objective_matrix(delay_set, [block_idx]),
                )
                q_block, e_block = estimate_q_e(delay_hat)
                q_blocks.append(q_block)
                e_blocks.append(e_block)
                h_blocks.append(
                    estimate_h(
                        delay_hat,
                        x_block_list[block_idx],
                        y_block_list[block_idx],
                    )
                )

            q = torch.stack(q_blocks, dim=1)
            e = torch.stack(e_blocks, dim=1)
            h_hat = torch.stack(h_blocks, dim=1)

        if self.nb == 1:
            q = q.squeeze(1)
            e = e.squeeze(1)
        return q, e, h_hat

    def forward(
        self,
        batch_size: int,
        ebno_db: float | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Simulate one Monte-Carlo batch and return true and decoded bits."""
        bits = self.binary_source([batch_size, self.k])
        codewords = self.encoder(bits)
        x_data = self.mapper(codewords).reshape(
            batch_size,
            self.nb,
            self.payload_length_per_block,
        )
        zc_seq = self.get_zc_seq(batch_size)
        x = torch.cat((zc_seq, x_data), dim=-1)

        overall_rate = self.k / ((self.preamble_length + self.payload_length) * self.num_bits_per_symbol)
        # ! Even through we set ebno_db, it actually represents Es/N0, not Eb/N0
        no = ebnodb2no(
            ebno_db,
            num_bits_per_symbol=1,
            coderate=1.0,
        )

        y_pilot, h_true, q_true, e_true = self.rayleigh_fading_channel(
            zc_seq,
            no,
            n_up=self.n_up,
            max_delay=self.max_delay,
            shared_delay=self.shared_delay,
            debug_fading=self.debug_fading,
        )
        delay_true = q_true.to(torch.float32) * self.ts + e_true.to(torch.float32)
        q_hat, e_hat, h_hat = self.synchronize_and_estimate_channel(y_pilot)
        delay_hat = q_hat.to(torch.float32) * self.ts + e_hat.to(torch.float32)

        if not torch.compiler.is_compiling():
            delay_hat_blocks = self.rayleigh_fading_channel.as_block_real(
                delay_hat,
                "delay_hat",
                batch_size,
                torch.float32,
                self.device,
            )
            delay_true_blocks = self.rayleigh_fading_channel.as_block_real(
                delay_true,
                "delay_true",
                batch_size,
                torch.float32,
                self.device,
            )
            delta = delay_hat_blocks - delay_true_blocks
            logger.info(
                "Mean absolute delay-estimation error: {:.6f}",
                torch.mean(torch.abs(delta)).item(),
            )
            if bool(torch.any(torch.abs(delta) > self.tp).item()):
                logger.warning(
                    "Some absolute delay-estimation errors exceed tp={}",
                    self.tp,
                )

        h_true = h_true.squeeze(-1)
        y_message = self.rayleigh_fading_channel.resample(
            x=x,
            predicted_delay=delay_hat,
            delay_true=delay_true,
            h_true=h_true,
            no=no,
        )
        block_symbol_len = self.preamble_length_per_block + self.payload_length_per_block
        y_blocks = y_message.reshape(
            batch_size,
            self.nb,
            block_symbol_len,
        )
        y_payload = y_blocks[:, :, self.preamble_length_per_block :].reshape(
            batch_size,
            self.payload_length,
        )
        h_payload = torch.repeat_interleave(
            h_hat,
            self.payload_length_per_block,
            dim=1,
        ).reshape(batch_size, self.payload_length)

        x_hat = y_payload.to(torch.complex64) / h_payload
        no_scaled = no / torch.abs(h_payload).square()
        llr = self.demapper(x_hat, no_scaled)
        bits_hat = self.decoder(llr)
        return bits, bits_hat


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=("Run the Sionna 2.0.1/PyTorch four-block channel-coded baseline with estimated timing and CSI."))
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument(
        "--modulation",
        type=str.lower,
        choices=("qam", "8psk", "16psk", "32psk", "64psk"),
        default="32psk",
        help=(
            "Payload modulation: qam (16-QAM, default), Gray-labeled 8psk (3 bits/symbol), Gray-labeled 16psk (4 bits/symbol), Gray-labeled 32psk (5 bits/symbol), or Gray-labeled 64psk (6 bits/symbol)."
        ),
    )
    parser.add_argument(
        "--snr-db",
        type=float,
        nargs="+",
        default=np.linspace(10, 24, num=8).tolist(),
        metavar="DB",
        help="Es/N0 points in dB (default: 24, 24.33, 24.67, 25).",
    )
    parser.add_argument("--num-target-block-errors", type=int, default=500)
    parser.add_argument("--max-mc-iter", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--debug-fading",
        action="store_true",
        help="Use unit-magnitude fading blocks and a final deep fade with complex variance 1e-5.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Computation device. Under Slurm, cuda maps to the allocated GPU.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.environ.get("WANDB_MODE", "online"),
    )
    parser.add_argument(
        "--wandb-project",
        default="Joint-detection-and-decoding-in-short-packet-communications",
    )
    return parser


def _select_device(requested: str) -> str:
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
        return "cuda:0"
    return "cpu"


def run_simulation(
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Configure Sionna, run BER/BLER simulation, and log the results."""
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.max_mc_iter <= 0:
        raise ValueError("--max-mc-iter must be positive.")
    if args.num_target_block_errors <= 0:
        raise ValueError("--num-target-block-errors must be positive.")

    device = _select_device(args.device)
    phy.config.device = device
    phy.config.seed = args.seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.modulation == "8psk":
        num_bits_per_symbol = 3
    elif args.modulation == "16psk":
        num_bits_per_symbol = 4
    elif args.modulation == "32psk":
        num_bits_per_symbol = 5
    elif args.modulation == "64psk":
        num_bits_per_symbol = 6
    else:
        num_bits_per_symbol = 6
    num_blocks = 4
    preamble_length_per_block = 2
    preamble_length = preamble_length_per_block * num_blocks
    payload_length = 64 - preamble_length
    payload_length_per_block = payload_length // num_blocks
    n = payload_length * num_bits_per_symbol
    k = 64
    channel_coderate = k / n
    code_picked = "ldpc"  # Choose "ldpc", "polar_scl8", or "polar_sc".
    if args.modulation == "64psk":
        # The 64/336 rate is below Sionna's minimum LDPC rate of 1/5.
        code_picked = "polar_scl8"
    # debug_fading = getattr(args, "debug_fading", True)  # Enable with --debug-fading or set to True here.
    debug_fading = False  # Enable with --debug-fading or set to True here.

    model = CodedSystemBlockFading(
        num_bits_per_symbol=num_bits_per_symbol,
        n=n,
        channel_coderate=channel_coderate,
        modulation=args.modulation,
        code_picked=code_picked,
        perfect_sync=False,
        perfect_csi=False,
        preamble_length=preamble_length,
        preamble_length_per_block=preamble_length_per_block,
        payload_length=payload_length,
        payload_length_per_block=payload_length_per_block,
        num_taps=1,
        nb=num_blocks,
        tp=1.0,
        ts=0.125,
        n_up=8,
        max_delay=4.0,
        shared_delay=True,
        debug_fading=debug_fading,
        device=device,
    )
    model.eval()
    snr_db = torch.tensor(args.snr_db, dtype=torch.float32, device=device)

    logger.info(
        "Running imperfect-sync/imperfect-CSI baseline on {} at Es/N0 {}",
        device,
        args.snr_db,
    )
    ber, bler = sim_ber(
        model,
        ebno_dbs=snr_db,
        batch_size=args.batch_size,
        num_target_block_errors=args.num_target_block_errors,
        max_mc_iter=args.max_mc_iter,
        device=device,
    )

    snr_values = snr_db.detach().cpu().tolist()
    ber_values = ber.detach().cpu().tolist()
    bler_values = bler.detach().cpu().tolist()
    wandb.log({
        "BER_BlockFadingChannel": wandb.plot.line(
            wandb.Table(
                data=list(zip(snr_values, ber_values)),
                columns=["SNR", "BER"],
            ),
            "SNR",
            "BER",
            title="BER with estimated synchronization and CSI",
        ),
        "BLER_BlockFadingChannel": wandb.plot.line(
            wandb.Table(
                data=list(zip(snr_values, bler_values)),
                columns=["SNR", "BLER"],
            ),
            "SNR",
            "BLER",
            title="BLER with estimated synchronization and CSI",
        ),
    })

    print("\nEb/N0 [dB]:", snr_values)
    print("BER:        ", ber_values)
    print("BLER:       ", bler_values)
    return ber, bler


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_argument_parser().parse_args(argv)
    with wandb.init(
        project=args.wandb_project,
        mode=args.wandb_mode,
        config=vars(args),
        save_code=True,
    ):
        run_simulation(args)
    print("Finished!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
