"""Serial turbo autoencoder model and its encoder/decoder building blocks."""

from typing import Literal, cast

import torch
from numpy import arange
from numpy.random import mtrand
from torch import nn
from torch.nn import functional

import wandb
from src.channel import RayleighBlockFadingChannel
from src.models import DelayEstimator, MySign, SoftQPSK
from src.models.components.crc import crc_padding
from src.models.components.sionna_interleaver import interleaver_polar


def generate_permutation_array(k: int, seed: int = 0, interleaver_type: str = "random") -> torch.Tensor:
    """
    Return the selected interleaver permutation for a block of length ``k``.

    Args:
        k: Number of positions to interleave, including CRC bits.
        seed: Seed for the random interleaver; ignored by ``sionna_polar``.
        interleaver_type: Either ``random`` or ``sionna_polar``.

    Returns:
        A one-dimensional tensor containing the permuted indices.

    Raises:
        ValueError: If ``interleaver_type`` is unsupported.

    """
    if interleaver_type == "sionna_polar":
        return interleaver_polar(k)
    if interleaver_type != "random":
        raise ValueError(f"interleaver_type must be 'random' or 'sionna_polar', got {interleaver_type!r}.")

    if seed == 100:
        p_array = torch.tensor([
            8,
            48,
            56,
            29,
            36,
            22,
            15,
            4,
            63,
            44,
            54,
            31,
            23,
            16,
            9,
            0,
            42,
            61,
            33,
            24,
            51,
            14,
            6,
            41,
            58,
            34,
            27,
            50,
            20,
            7,
            43,
            62,
            35,
            28,
            55,
            17,
            2,
            47,
            10,
            40,
            32,
            25,
            59,
            3,
            52,
            11,
            45,
            18,
            38,
            60,
            1,
            53,
            26,
            46,
            19,
            12,
            37,
            5,
            57,
            30,
            49,
            21,
            13,
            39,
        ])
    else:
        rand_gen = mtrand.RandomState(seed=seed)
        p_array = rand_gen.permutation(arange(k))

        p_array = torch.LongTensor(p_array).view(len(p_array))

    return p_array


def add_padding(input_tensor: torch.Tensor, n_m: int) -> torch.Tensor:
    """
    Pads the input tensor with zeros on the right along the last dimension.

    Args:
        input_tensor (torch.Tensor): Input tensor of shape (batch_size, F, n) with dtype torch.float32.
        n_m (int): Number of zeros to pad on the right side of the last dimension.

    Returns:
        torch.Tensor: Padded tensor of shape (batch_size, F, n + n_m).

    """
    # Padding for the last dimension is specified as (pad_left, pad_right).
    padded_tensor = functional.pad(
        input_tensor,
        pad=(0, n_m),
        mode="constant",
        value=0,
    )

    return padded_tensor


def remove_padding(padded_tensor: torch.Tensor, n_m: int) -> torch.Tensor:
    """
    Removes the padding (n_m zeros) from the right along the last dimension.

    Args:
        padded_tensor (torch.Tensor): Padded tensor of shape (batch_size, F, n + n_m).
        n_m (int): Number of padded zeros on the right side of the last dimension.

    Returns:
        torch.Tensor: Tensor of shape (batch_size, F, n) after removing padding.

    """
    # Remove the last n_m elements; assumes padded_tensor.shape[-1] >= n_m.

    if n_m > 0:
        return padded_tensor[..., :-n_m]
    return padded_tensor


class CNNBlock(nn.Module):
    """Stack of convolutional layers used by the encoder and decoder."""

    def __init__(self, in_channels: int, out_channels: int, padding: str = "same") -> None:
        super().__init__()

        # Define the sequence of convolutional layers
        # input size = (batch_size, in_channels, k), output size = (batch_size, n, k)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=100, kernel_size=5, padding=padding),
            nn.ELU(),
            nn.Conv1d(in_channels=100, out_channels=100, kernel_size=5, padding=padding),
            nn.ELU(),
            nn.Conv1d(in_channels=100, out_channels=100, kernel_size=5, padding=padding),
            nn.ELU(),
            nn.Conv1d(in_channels=100, out_channels=100, kernel_size=5, padding=padding),
            nn.ELU(),
            nn.Conv1d(in_channels=100, out_channels=100, kernel_size=5, padding=padding),
            nn.ELU(),
        )

        # Define the output layer
        # input size = (batch_size, n, k), output size = (batch_size, F, k)
        self.output_layer = nn.Conv1d(in_channels=100, out_channels=out_channels, kernel_size=1)
        # self.output_layer = nn.Linear(in_features=100, out_features=F)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the convolutional stack to ``x``.

        Returns:
            The transformed feature tensor.

        """
        # Pass the input through the convolutional layers
        x = self.conv_layers(x)

        # convert the second and third dimension of the tensor
        # x = x.permute(0, 2, 1)

        # Pass the result through the output layer
        x = self.output_layer(x)

        return x


class Encoder(nn.Module):
    """Encode binary messages into complex-valued channel symbols."""

    def __init__(
        self,
        num_features: int,
        p_array_seed: int,
        CRC: int = 0,
        enforce_qpsk_symbols: bool = False,
        is_soft_qpsk: bool = False,
        soft_qpsk_alpha: float = 0.5,
        interleaver_type: str = "random",
    ) -> None:
        super().__init__()
        if enforce_qpsk_symbols and is_soft_qpsk:
            raise ValueError("enforce_qpsk_symbols and is_soft_qpsk are mutually exclusive; enable at most one.")
        if is_soft_qpsk and not 0.0 <= soft_qpsk_alpha <= 1.0:
            raise ValueError(f"soft_qpsk_alpha must lie in [0, 1], got {soft_qpsk_alpha}.")
        self.p_array_seed = p_array_seed
        self.interleaver_type = interleaver_type
        self.add_parity_check_bit = False
        self.CRC = CRC
        self.enforce_qpsk_symbols = enforce_qpsk_symbols
        self.is_soft_qpsk = is_soft_qpsk
        self.soft_qpsk_alpha = soft_qpsk_alpha

        self.outer_encoder = CNNBlock(in_channels=1, out_channels=num_features)

        # The outer encoder's activations reach MySign at a scale of roughly
        # 0.05. sign() is scale invariant, so the forward pass is unaffected,
        # but the straight-through gradient reaching the outer encoder is
        # correspondingly tiny and training idles at BER 0.5 until the
        # activation scale slowly inflates. Normalizing here costs no
        # parameters and removes that dead phase, which grows with num_features.
        self.pre_sign_norm = nn.BatchNorm1d(num_features, affine=False)

        self.inner_encoder = CNNBlock(in_channels=num_features, out_channels=2)

    @staticmethod
    def set_interleaver(inputs: torch.Tensor, p_array: torch.Tensor) -> torch.Tensor:
        """
        Permute the symbol dimension using ``p_array``.

        Returns:
            The permuted feature tensor.

        """
        # inputs has the shape (batch_size, k, 1)
        inputs = inputs.permute(2, 0, 1)  # output size = (n, batch_size, F)
        res = inputs[p_array]
        res = res.permute(1, 2, 0)  # output size = (batch_size, F, n)

        return res

    @staticmethod
    def power_constraint(x: torch.Tensor) -> torch.Tensor:
        """
        Apply power constraint to the input tensor x.

        This function normalizes each input symbol to have power 1.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n) with dtype torch.complex64.

        Returns:
            torch.Tensor: Power-constrained tensor of shape (batch_size, n) with dtype torch.complex64.

        """
        if torch.is_complex(x):
            power = x.real.square() + x.imag.square()
        elif x.dim() > 2 and x.size(-1) == 2:
            power = x.square().sum(dim=-1, keepdim=True)
        else:
            power = x.square()

        eps = torch.finfo(power.dtype).eps
        return x * torch.rsqrt(power.clamp_min(eps))

    @staticmethod
    def add_even_parity_bit(x: torch.Tensor) -> torch.Tensor:
        """
        Adds an even parity check bit along dimension 1 of a binary tensor.

        Args:
            x (torch.Tensor): Binary tensor of shape (batch, k, 1) with values 0 or 1.

        Returns:
            torch.Tensor: Tensor of shape (batch, k+1, 1) with even parity bit appended.

        Raises:
            ValueError: If ``x`` contains values other than zero and one.

        """
        # Ensure it's binary 0/1
        if not torch.all((x == 0) | (x == 1)):
            raise ValueError("Input tensor must contain only 0s and 1s.")

        # Sum along dimension 1 (keepdim=True to maintain shape for broadcasting)
        ones_count = x.sum(dim=1)  # shape: (batch, 1)

        # Even parity bit: 1 if sum is odd, else 0
        parity_bit = (ones_count % 2).to(x.dtype)  # 1 if odd, 0 if even

        # Append parity bit along dim=1
        parity_bit = parity_bit.unsqueeze(1)  # shape: (batch, 1, 1)
        return torch.cat([x, parity_bit], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of binary messages.

        Returns:
            Encoded real-valued in-phase and quadrature components.

        """
        if self.add_parity_check_bit:
            x = Encoder.add_even_parity_bit(x)

        block_length = x.shape[1]  # k+CRC
        p_array = generate_permutation_array(block_length, seed=self.p_array_seed, interleaver_type=self.interleaver_type)

        x = 2.0 * x - 1.0

        # switch the second and third dimension of the tensor
        x = torch.transpose(x, 1, 2)  # output size = (batch_size, 1, k+CRC)

        # outer encoder input size = (batch_size, 1, k+CRC), output size = (batch_size, F, k+CRC)
        x = self.outer_encoder(x)

        # normalize before quantization so the straight-through gradient is
        # not scaled down by the small activation magnitude (see __init__)
        x = self.pre_sign_norm(x)

        # quantization to 2 levels
        # x = STEQuantize.apply(x, 1.0, 2)
        x = MySign.apply(x)

        # interleaver
        x = self.set_interleaver(x, p_array)  # output size = (batch_size, F, k+CRC)

        # inner encoder
        x = self.inner_encoder(x)  # output size = (batch_size, 2, k+CRC)

        # switch the second and third dimension of the tensor
        x = torch.transpose(x, 1, 2)  # output size = (batch_size, k+CRC, 2)

        if self.CRC:
            # input size = (batch_size, k+CRC, 2), output size = (batch_size, k, 2), k=n
            # simply remove the last element along dimension 1
            x = x[:, : -self.CRC, :]

        if self.enforce_qpsk_symbols:
            x = MySign.apply(x)
        elif self.is_soft_qpsk:
            # pull each symbol toward, but not onto, its nearest normalized QPSK point
            x = SoftQPSK.apply(x, self.soft_qpsk_alpha)

        return x


class EqualizerCNNBlock(nn.Module):
    """Convolutional equalizer block."""

    def __init__(self, in_channels: int, num_features: int) -> None:
        super().__init__()

        self.cnn_block1 = CNNBlock(
            in_channels=in_channels,
            out_channels=num_features,
        )  # output size = (batch_size, F, n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply equalization to ``x``.

        Returns:
            The equalized feature tensor.

        """
        # input size = (batch_size, n, n), output size = (batch_size, F, n)
        x = self.cnn_block1(x)

        return x


class DecoderCNNBlock(nn.Module):
    """Intermediate convolutional decoder block."""

    def __init__(self, in_channels: int, num_features: int) -> None:
        super().__init__()

        self.cnn_block1 = CNNBlock(
            in_channels=in_channels,
            out_channels=num_features,
        )  # output size = (batch_size, F, n)
        self.cnn_block2 = nn.Conv1d(
            in_channels=num_features,
            out_channels=num_features,
            kernel_size=1,
            padding="same",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode one iteration of feature messages.

        Returns:
            Updated decoder features.

        """
        # input size = (batch_size, F, n), output size = (batch_size, F, n)
        x = self.cnn_block1(x)

        # input size = (batch_size, n, n), output size = (batch_size, F, n)
        x = self.cnn_block2(x)

        return x


class DecoderCNNLastBlock(nn.Module):
    """Final convolutional decoder block producing bit probabilities."""

    def __init__(
        self,
        in_channels: int,
        k: int,
        num_features: int,
        output_size: int,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.k = k
        self.F = num_features
        self.output_size = output_size

        self.cnn_block1 = CNNBlock(
            in_channels=in_channels,
            out_channels=num_features,
        )  # output size = (batch_size, F, n)

        # input size = (batch_size, F, n), output size = (batch_size, F*n)
        self.flatten = nn.Flatten()

        self.decoder_ouput = nn.Linear(
            in_features=num_features * k,
            out_features=output_size,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return decoded bit probabilities for ``x``.

        Returns:
            A tensor of decoded bit probabilities.

        """
        # input size = (batch_size, F, n), output size = (batch_size, M1, n)
        x = self.cnn_block1(x)

        # flattten layer
        # input size = (batch_size, M1, n), output size = (batch_size, M1*n)
        x = self.flatten(x)

        # final decoder output
        # input size = (batch_size, M1*n), output size = (batch_size, k)
        x = self.decoder_ouput(x)

        # add sigmoid activation function
        x = self.sigmoid(x)

        # add one dimension to the output (second dimension) to make sure the output has the shape (batch_size, 1, k)
        x = x.unsqueeze(1)

        return x


class Decoder(nn.Module):
    """Iteratively equalize and decode received channel samples."""

    def __init__(
        self,
        k: int,
        n: int,
        num_features: int,
        n_m: int,
        upsampling_factor: int,
        num_iteration: int,
        p_array_seed: int,
        CRC: int = 0,
        interleaver_type: str = "random",
        padding_input: str = "padding_zeros",  # options: "padding_zeros", "nn_linear", "nn_conv1d"
    ) -> None:
        super().__init__()

        self.k = k
        self.n = n
        self.F = num_features
        self.n_M = n_m
        self.N_up = upsampling_factor
        self.num_iteration = num_iteration
        self.p_array_seed = p_array_seed
        self.interleaver_type = interleaver_type
        self.CRC = CRC
        self.padding_input = padding_input

        if self.CRC:
            # self.n = n + self.CRC  # if add CRC, then n = n + CRC
            # input size = (batch_size, 2*N_up, n), output size = (batch_size, n, n)
            self.dec_input = nn.Sequential(
                nn.Conv1d(in_channels=2 * upsampling_factor, out_channels=n, kernel_size=1, stride=1),
                nn.BatchNorm1d(n),
                nn.ELU(),
            )
            # enlarge the input (batch_size, n, n) to (batch_size, n+CRC, n) for the EQ-CNN block
            if self.padding_input == "padding_zeros":
                pass
            elif self.padding_input == "nn_linear":
                # use linear layer
                self.enlarge_input_crc = nn.Sequential(
                    nn.Linear(in_features=n, out_features=n + self.CRC),
                    # nn.BatchNorm1d(n + self.CRC),
                    nn.ELU(),
                )
            elif self.padding_input == "nn_conv1d":
                # use conv1D
                self.enlarge_input_crc = nn.Sequential(
                    nn.Conv1d(in_channels=n, out_channels=n + self.CRC, kernel_size=1, stride=1),
                    nn.BatchNorm1d(n + self.CRC),
                    nn.ELU(),
                )

            self.eq_block_list = nn.ModuleList()
            self.dec_block_list = nn.ModuleList()
            for i in range(num_iteration):
                self.eq_block_list.append(
                    EqualizerCNNBlock(
                        in_channels=n + num_features,
                        num_features=num_features,
                    )
                )

                if i != num_iteration - 1:
                    self.dec_block_list.append(
                        DecoderCNNBlock(
                            in_channels=num_features,
                            num_features=num_features,
                        )
                    )

            # input size = (batch_size, F, n+1), output size = (batch_size, 1, k+1)
            self.dec_last_block = DecoderCNNLastBlock(
                in_channels=num_features,
                k=k + self.CRC,
                num_features=num_features,
                output_size=k + self.CRC,
            )
        else:
            # input size = (batch_size, 2, n*N_up), output size = (batch_size, 2, n)
            self.dec_input = nn.Sequential(
                nn.Conv1d(
                    in_channels=2 * upsampling_factor,
                    out_channels=n,
                    kernel_size=1,
                    stride=1,
                ),
                nn.BatchNorm1d(n),
                nn.ELU(),
            )

            # # input size = (batch_size, n+F, n), output size = (batch_size, F, n)
            # self.eq_block = EQ_CNN_block(in_channels=n + F, k=n, n=n, F=F)

            # # input size = (batch_size, F, n), output size = (batch_size, F, n)
            # self.dec_block = DEC_CNN_block(in_channels=F, k=n, n=n, F=F)
            self.eq_block_list = nn.ModuleList()
            self.dec_block_list = nn.ModuleList()
            for i in range(num_iteration):
                self.eq_block_list.append(
                    EqualizerCNNBlock(
                        in_channels=n + num_features,
                        num_features=num_features,
                    )
                )

                if i != num_iteration - 1:
                    self.dec_block_list.append(
                        DecoderCNNBlock(
                            in_channels=num_features,
                            num_features=num_features,
                        )
                    )

            # input size = (batch_size, F, n), output size = (batch_size, 1, k)
            self.dec_last_block = DecoderCNNLastBlock(
                in_channels=num_features,
                k=k,
                num_features=num_features,
                output_size=k,
            )

    @staticmethod
    def interleaver(inputs: torch.Tensor, p_array: torch.Tensor) -> torch.Tensor:
        """
        Apply an interleaver permutation to feature messages.

        Returns:
            The permuted feature tensor.

        """
        # inputs has the shape (batch_size, F, n)

        inputs = inputs.permute(2, 0, 1)  # output size = (n, batch_size, F)
        res = inputs[p_array]
        res = res.permute(1, 2, 0)  # output size = (batch_size, F, n)

        return res

    @staticmethod
    def deinterleaver(inputs: torch.Tensor, p_array: torch.Tensor) -> torch.Tensor:
        """
        Invert an interleaver permutation for feature messages.

        Returns:
            The restored feature tensor.

        """
        # inputs has the shape (batch_size, F, n)

        reverse_p_array = torch.argsort(p_array)

        inputs = inputs.permute(2, 0, 1)  # input size = (n, batch_size, F)
        res = inputs[reverse_p_array]
        res = res.permute(1, 2, 0)  # output size = (batch_size, F, n)

        return res

    # pylint: disable-next=too-many-locals
    def forward(self, y_delay_removed: torch.Tensor) -> torch.Tensor:
        """
        Decode a received, delay-corrected signal into bit probabilities.

        Returns:
            A tensor containing the decoded bit probabilities.

        Raises:
            ValueError: If the received tensor has an incompatible shape.

        """
        device = y_delay_removed.device

        if y_delay_removed.dim() == 4:
            batch_size, nb, block_length, complex_dim = y_delay_removed.shape
            if complex_dim != 2:
                raise ValueError(f"Expected complex dimension 2, got {complex_dim}.")
            if self.n % nb != 0:
                raise ValueError(f"n={self.n} must be divisible by nb={nb}.")
            expected_block_length = (self.n // nb + 2 * self.n_M) * self.N_up
            if block_length < expected_block_length:
                raise ValueError(f"Received block is too short for polyphase decoder: got {block_length}, need at least {expected_block_length}.")
            y_delay_removed = y_delay_removed[:, :, :expected_block_length, :]
            y_delay_removed = y_delay_removed.reshape(batch_size, -1, 2)
        elif y_delay_removed.dim() != 3:
            raise ValueError("Expected received signal with shape (batch, length, 2) or (batch, nb, block_length, 2).")

        batch_size, received_length, complex_dim = y_delay_removed.shape
        if complex_dim != 2:
            raise ValueError(f"Expected complex dimension 2, got {complex_dim}.")
        if received_length % self.N_up != 0:
            raise ValueError(f"Received length {received_length} is not divisible by N_up={self.N_up}.")
        symbol_length = received_length // self.N_up  # 512/8 = 64
        expected_symbol_length = self.n + 2 * self.n_M
        if symbol_length != expected_symbol_length:
            raise ValueError(f"Unexpected symbol-rate length after polyphase reshape: got {symbol_length}, expected {expected_symbol_length}.")

        y_delay_removed = y_delay_removed.reshape(batch_size, symbol_length, self.N_up, 2)
        y_delay_removed = y_delay_removed.permute(0, 3, 2, 1).reshape(batch_size, 2 * self.N_up, symbol_length)

        # input size = (batch_size, 2*N_up, n), output size = (batch_size, n, n)
        y = self.dec_input(y_delay_removed)
        if self.CRC:
            if self.padding_input == "padding_zeros":
                # pad the input with zeros from the right to make it of shape (batch_size, n+CRC, n)
                y = add_padding(y, self.CRC)  # output size = (batch_size, n+CRC, n)
            elif self.padding_input == "nn_linear":
                # input size = (batch_size, n, n), output size = (batch_size, n+CRC, n)
                y = self.enlarge_input_crc(y)
            elif self.padding_input == "nn_conv1d":
                # transpose the second and third dimension of the tensor, output size = (batch_size, n, n)
                y = y.permute(0, 2, 1)
                # input size = (batch_size, n, n), output size = (batch_size, n+CRC, n)
                y = self.enlarge_input_crc(y)
                # transpose the second and third dimension of the tensor, output size = (batch_size, n, n+CRC)
                y = y.permute(0, 2, 1)

        block_length = self.n + self.CRC  # n+CRC
        p_array = generate_permutation_array(block_length, seed=self.p_array_seed, interleaver_type=self.interleaver_type)

        # initiate i_c, i_c_pri is all zeros with the shape (batch_size, F, n)
        i_c_pri = torch.zeros((y.shape[0], self.F, self.n + self.CRC), dtype=torch.float32, device=device)

        # iterative EQ-DEC block
        for i in range(self.num_iteration - 1):
            # pad i_c_pri with zeros from the right to make it of shape (batch_size, F, n+CRC)
            i_c_pri_padding = add_padding(i_c_pri, self.n_M)  # output size = (batch_size, F, n+CRC)

            # concatenate y with i_c_pri
            # y is of shape (batch_size, n, n+CRC), i_c_pri shape (batch_size, F, n+CRC)
            y_concatenate = torch.cat((y, i_c_pri_padding), dim=1)  # output size = (batch_size, n+F, n+CRC)

            # EQ-CNN block
            # input size = (batch_size, n+F, n+CRC), output size = (batch_size, F, n+CRC)
            i_c = self.eq_block_list[i](y_concatenate)

            # remove padding
            # input size = (batch_size, F, n+CRC), output size = (batch_size, F, n+CRC)
            i_c = remove_padding(i_c, self.n_M)

            # update i_c
            i_c = i_c - i_c_pri

            # deinterleaver
            i_c = self.deinterleaver(i_c, p_array)

            # DEC-CNN block
            # input size = (batch_size, F, n+CRC), output size = (batch_size, F, n+CRC)
            i_b = self.dec_block_list[i](i_c)

            # update i_c_pri
            i_c_pri = i_b - i_c

            # interleaver
            i_c_pri = self.interleaver(i_c_pri, p_array)

        # ! last iteration
        # EQ-CNN last block

        # pad i_c_pri with zeros from the left and the right to make it of shape (batch_size, F, n+CRC)
        i_c_pri_padding = add_padding(i_c_pri, self.n_M)  # output size = (batch_size, F, n+CRC)

        # concatenate y with i_c_pri
        # y is of shape (batch_size, n, n+CRC), i_c_pri shape (batch_size, F, n+CRC)
        y_concatenate = torch.cat((y, i_c_pri_padding), dim=1)  # output size = (batch_size, n+F, n+CRC)
        # input size = (batch_size, n+F, n+CRC), output size = (batch_size, F, n+CRC)
        i_c = self.eq_block_list[self.num_iteration - 1](y_concatenate)

        # remove padding
        # input size = (batch_size, F, n+CRC), output size = (batch_size, F, n+CRC)
        i_c = remove_padding(i_c, self.n_M)

        # update i_c
        i_c = i_c - i_c_pri

        # deinterleaver
        i_c = self.deinterleaver(i_c, p_array)

        # DEC-CNN last block
        # input size = (batch_size, F, n+CRC), output size = (batch_size, 1, k+CRC)
        y_decoded = self.dec_last_block(i_c)

        # switch the second and third dimension of the tensor, output size = (batch_size, 1, k+CRC)
        y_decoded = y_decoded.permute(0, 2, 1)

        return y_decoded


class TurboAutoencoderSync(nn.Module):
    """End-to-end serial turbo autoencoder with synchronization block."""

    def __init__(self, autoencoder_config: wandb.sdk.wandb_config.Config) -> None:
        super().__init__()

        self.k = cast("int", autoencoder_config["k"])
        self.CRC = cast("int", autoencoder_config.get("CRC", 0))
        self.n = cast("int", autoencoder_config["n"])
        self.n_M = cast("int", autoencoder_config["n_M"])
        self.F = cast("int", autoencoder_config["F"])
        self.channel = cast("str", autoencoder_config["channel"])
        self.is_delay = cast("bool", autoencoder_config["is_delay"])
        self.num_iteration = cast("int", autoencoder_config["num_iteration"])
        self.p_array_seed = cast("int", autoencoder_config["p_array_seed"])
        self.interleaver_type = cast("str", autoencoder_config.get("interleaver_type", "random"))
        self.is_train_sync_block = cast("bool", autoencoder_config["is_train_sync_block"])
        self.independent_delay_per_block = cast("bool", autoencoder_config["independent_delay_per_block"])
        self.nb = cast("int", autoencoder_config["nb"])
        self.tp = cast("int", autoencoder_config["tp"])
        self.ts = cast("int", autoencoder_config["ts"])
        self.N = cast("int", autoencoder_config["N_up"])
        self.dmax = cast("int", autoencoder_config["dmax"])
        self.ns = int(self.n / self.nb)
        self.add_parity_check_bit = cast("bool", autoencoder_config["add_parity_check_bit"])
        self.enforce_qpsk_symbols = cast("bool", autoencoder_config["enforce_qpsk_symbols"])
        self.is_soft_qpsk = cast("bool", autoencoder_config["is_soft_qpsk"])
        self.soft_qpsk_alpha = cast("float", autoencoder_config["soft_qpsk_alpha"])
        self.encoder_output_power_constraint = cast("str", autoencoder_config["encoder_output_power_constraint"])
        self.receiver_resample_method = cast("str", autoencoder_config["receiver_resample_method"])
        self.receiver_resample_sample_phase = cast("float", autoencoder_config["receiver_resample_sample_phase"])
        self.receiver_mmse_regularization = cast("float", autoencoder_config["receiver_mmse_regularization"])
        self.sync_hidden_channels = cast("int", autoencoder_config["sync_hidden_channels"])
        self.sync_num_layers = cast("int", autoencoder_config["sync_num_layers"])
        self.sync_kernel_size = cast("int", autoencoder_config["sync_kernel_size"])
        self.sync_activation = cast("str", autoencoder_config["sync_activation"])
        self.sync_fuse_blocks = cast("bool", autoencoder_config["sync_fuse_blocks"])
        self.sync_separate_q_eps = cast("bool", autoencoder_config["sync_separate_q_eps"])

        self.encoder = Encoder(
            num_features=self.F,
            p_array_seed=self.p_array_seed,
            CRC=self.CRC,
            enforce_qpsk_symbols=self.enforce_qpsk_symbols,
            is_soft_qpsk=self.is_soft_qpsk,
            soft_qpsk_alpha=self.soft_qpsk_alpha,
            interleaver_type=self.interleaver_type,
        )
        if self.is_train_sync_block:
            self.blockfadingchannel = RayleighBlockFadingChannel(
                self.nb,
                self.tp,
                self.ts,
                self.N,
                self.dmax,
                self.is_delay,
                self.receiver_mmse_regularization,
                normalize_transmit_blocks=self.encoder_output_power_constraint != "average",
            )
            self.sync_block = DelayEstimator(
                hidden_channels=self.sync_hidden_channels,
                num_layers=self.sync_num_layers,
                kernel_size=self.sync_kernel_size,
                activation=self.sync_activation,
                fuse_blocks=self.sync_fuse_blocks,
                dmax=self.dmax,
                ts=self.ts,
                separate_q_eps=self.sync_separate_q_eps,
            )

        self.decoder = Decoder(
            k=self.k,
            n=self.n,
            num_features=self.F,
            n_m=self.n_M,
            upsampling_factor=self.N,
            num_iteration=self.num_iteration,
            p_array_seed=self.p_array_seed,
            CRC=self.CRC,
            interleaver_type=self.interleaver_type,
        )

    def forward(
        self,
        x: torch.Tensor,
        delay_true: torch.Tensor,
        SNR_db_low: float,
        SNR_db_high: float,
        mode: Literal[
            "encoder",
            "decoder",
            "sync_encoder",
            "sync_block",
            "joint_encoder",
            "joint_decoder",
        ] = "encoder",
        return_sync_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Run the encoder, channel, and the receiver stages selected by ``mode``.

        Returns:
            Decoded bits, predicted delay, or both as selected by ``mode``.

        Raises:
            NotImplementedError: If the configured channel is unsupported.
            ValueError: If ``mode`` is not supported.

        """
        # Retained for API compatibility; joint modes already return both components.
        _ = return_sync_components

        # Transmitter part

        # add CRC check bits to the input message x before feeding into the encoder,
        # according to the CRC: int in the config file. The CRC check bits should be added to the end of the input message x,
        # and the length of the input message x should be increased accordingly.
        if self.CRC:
            x = crc_padding(x, self.CRC)

        # input size = (batch_size, k + CRC, 1), output size = (batch_size, n, 2)
        x = self.encoder(x)
        # Make sure the tensor is contiguous in memory before view_as_complex
        x = x.contiguous()
        # convert x from real to complex tensor, output size = (batch_size, n)
        x = torch.view_as_complex(x)  # output size = (batch_size, n)

        if self.encoder_output_power_constraint == "per_symbol":
            x_transmitted = self.encoder.power_constraint(x)
        elif self.encoder_output_power_constraint == "per_block":
            x_transmitted = normalize_per_fading_block(x, self.nb, self.ns)
        elif self.encoder_output_power_constraint == "average":
            x_transmitted = normalize_average_power(x)
        else:
            x_transmitted = x

        if self.channel == "block_fading":
            x, h_true = self.blockfadingchannel.forward(x_transmitted, delay_true, SNR_db_low, SNR_db_high)  # output size = (batch_size, nb, M)

            # normalised each block of the received signal to have unit power before feeding into the sync block
            # energies = torch.mean(torch.abs(x) ** 2, dim=2, keepdim=True)
            # energies = torch.clamp(energies, min=1e-12)
            # x = x / torch.sqrt(energies)  # shape (batch_size, nb, M)

            if mode in {"encoder", "decoder"}:
                x_resampled = self.blockfadingchannel.resample(
                    x_transmitted,
                    h_true,
                    SNR_db_low,
                    SNR_db_high,
                )  # output size = (batch_size, nb, (n / nb) * N)
                # normalised each block of the sampled signal to have unit power before feeding into the decoder block
                energies = torch.mean(torch.abs(x_resampled) ** 2, dim=2, keepdim=True)
                energies = torch.clamp(energies, min=1e-12)
                x_resampled = x_resampled / torch.sqrt(energies)  # shape (batch_size, nb, (n / nb) * N)
                # convert x_resampled from complex to real tensor, output size = (batch_size, nb, (n / nb) * N, 2)
                x_resampled = torch.view_as_real(x_resampled)  # output size = (batch_size, nb, (n / nb) * N, 2)

                # feed the resampled signal to the decoder
                predicted_bits = self.decoder(x_resampled)
                return predicted_bits
            if mode in {"sync_encoder", "sync_block"}:
                # Feed to the sync block. If block fusion is disabled, reduce
                # per-block estimates back to the shared-delay shape expected by
                # the trainer: (batch_size, 1).
                predicted_delay = self.sync_block(x)
                return predicted_delay
            if mode in {"joint_encoder", "joint_decoder"}:
                # Feed to the sync block. If block fusion is disabled, reduce
                # per-block estimates back to the shared-delay shape expected by
                # the trainer: (batch_size, 1).
                predicted_delay = self.sync_block(x)
                # Receiver-side timing correction must use only the sampled
                # receive waveform and the estimated delay.
                x_resampled = self.blockfadingchannel.receiver_resample(
                    x,
                    predicted_delay,
                    method=self.receiver_resample_method,
                    sample_phase=self.receiver_resample_sample_phase,
                )  # output size = (batch_size, nb, (n / nb) * N)
                # normalised each block of the sampled signal to have unit power before feeding into the decoder block
                energies = torch.mean(torch.abs(x_resampled) ** 2, dim=2, keepdim=True)
                energies = torch.clamp(energies, min=1e-12)
                x_resampled = x_resampled / torch.sqrt(energies)  # shape (batch_size, nb, (n / nb) * N)

                # convert x_resampled from complex to real tensor, output size = (batch_size, nb, (n / nb) * N, 2)
                x_resampled = torch.view_as_real(x_resampled)  # output size = (batch_size, nb, (n / nb) * N, 2)

                # feed the resampled signal to the decoder
                predicted_bits = self.decoder(x_resampled)

                return predicted_delay, predicted_bits
            raise ValueError(f"Unsupported forward mode: {mode}.")
        raise NotImplementedError


def normalize_average_power(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize mean complex-symbol power across the entire forward batch.

    A single scale is shared by all packets and symbols, preserving relative
    amplitudes and phases across fading blocks. This applies during both
    training and evaluation; each DDP worker normalizes its own local batch.
    All-zero inputs remain zero, and a floor keeps near-zero inputs finite.

    Args:
        x: Complex transmitted symbols with shape (batch_size, n).

    Returns:
        The signal with the same shape and unit mean squared magnitude when
        its input mean power is at least 1e-12.

    """
    average_power = (x.real.square() + x.imag.square()).mean()
    return x / torch.sqrt(average_power.clamp_min(1e-12))


def normalize_per_fading_block(x: torch.Tensor, nb: int, ns: int) -> torch.Tensor:
    """
    Normalize each fading block in ``x`` to unit average power.

    Returns:
        The normalized signal with its original two-dimensional shape.

    """
    x_blocks = x.reshape(x.shape[0], nb, ns)
    energies = torch.mean(torch.abs(x_blocks) ** 2, dim=2, keepdim=True)
    energies = torch.clamp(energies, min=1e-12)
    return (x_blocks / torch.sqrt(energies)).reshape(x.shape[0], -1)
