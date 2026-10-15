"""Simulate modern channel codes with QAM or PSK over an AWGN channel."""

try:
    import sionna.phy
except ImportError as e:
    import os
    import sys

    if "google.colab" in sys.modules:
        print("Installing Sionna and restarting the runtime. Please run the cell again.")
        os.system("pip install sionna")  # ty: ignore[deprecated]
        os.kill(os.getpid(), 5)
    else:
        raise e  # ruff: ignore[verbose-raise] - Preserve the original traceback behavior.

import numpy as np
import torch
import torch.nn.functional as F
from sionna.phy import Block
from sionna.phy.channel import AWGN
from sionna.phy.fec.ldpc import LDPC5GDecoder, LDPC5GEncoder
from sionna.phy.fec.polar import Polar5GDecoder, Polar5GEncoder
from sionna.phy.mapping import BinarySource, Constellation, Demapper, Mapper
from sionna.phy.utils import ebnodb2no
from sionna.phy.utils.misc import sim_ber

import wandb

sionna.phy.config.seed = 42


class SystemModel(Block):
    """Simulate a coded QAM or 8PSK transmission over an AWGN channel."""

    def __init__(
        self,
        k: int,
        n: int,
        num_bits_per_symbol: int,
        encoder: Block,
        decoder: Block,
        demapping_method: str = "app",
        sim_esno: bool = False,
        cw_estimates: bool = False,
        modulation: str | None = None,
    ) -> None:
        super().__init__()

        bits_per_modulation = {"4qam": 2, "8psk": 3, "16qam": 4}
        if modulation is None:
            # Preserve callers that already select QAM by bits per symbol.
            modulation = f"{1 << num_bits_per_symbol}qam"
        else:
            modulation = modulation.lower()
            if modulation not in bits_per_modulation:
                raise ValueError("modulation must be '4qam', '8psk', or '16qam'.")
            if num_bits_per_symbol != bits_per_modulation[modulation]:
                raise ValueError(f"{modulation} requires num_bits_per_symbol={bits_per_modulation[modulation]}.")

        self.k = k
        self.n = n
        self.sim_esno = sim_esno
        self.cw_estimates = cw_estimates
        self.modulation = modulation
        self.num_bits_per_symbol = num_bits_per_symbol

        self.source = BinarySource()
        if modulation == "8psk":
            # Array indices are binary bit labels. Around the circle these
            # labels are 000, 001, 011, 010, 110, 111, 101, 100.
            phase_indices = np.array([0, 1, 3, 2, 7, 6, 4, 5])
            points = np.exp(1j * (np.pi / 4) * phase_indices).astype(np.complex64)
            self.constellation = Constellation("custom", num_bits_per_symbol=3, points=points, normalize=True)
        else:
            self.constellation = Constellation("qam", num_bits_per_symbol=num_bits_per_symbol)
        self.mapper = Mapper(constellation=self.constellation)
        self.demapper = Demapper(demapping_method, constellation=self.constellation)
        self.channel = AWGN()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, batch_size: int, ebno_db: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Transmit and decode a batch at the given signal-to-noise ratio."""
        if self.sim_esno:
            # ! the input ebno_dbs is actually Es/N0 in dB, so we set num_bits_per_symbol=1 and coderate=1 to get the correct noise variance
            # ! no = energy_per_symbol / (ebno * coderate * num_bits_per_symbol), energy_per_symbol is 1 for normalized constellations, so no = 1 / (ebno * coderate * num_bits_per_symbol) = 1 / (Es/N0 * 1 * 1) = 1 / (Es/N0)
            no = ebnodb2no(ebno_db, num_bits_per_symbol=1, coderate=1)
        else:
            no = ebnodb2no(
                ebno_db,
                num_bits_per_symbol=self.num_bits_per_symbol,
                coderate=self.k / (self.n * self.num_bits_per_symbol),
            )

        u = self.source([batch_size, self.k])
        c = self.encoder(u)
        x = self.mapper(c)
        y = self.channel(x, no)
        llr_ch = self.demapper(y, no)
        u_hat = self.decoder(llr_ch)

        if self.cw_estimates:
            return c, u_hat
        return u, u_hat


def main() -> None:
    """Run the selected code simulation and log its error rates."""
    k = 64  # number of information bits
    n = 64  # blocklength (complex symbols)
    batch_size = 5000
    modulation = "4qam"  # Choose "4qam", "8psk", or "16qam".
    num_bits_per_symbol = {"4qam": 2, "8psk": 3, "16qam": 4}[modulation]
    Es_snr_db = torch.arange(0, 5, 0.5, dtype=torch.float64)

    # Choose "ldpc" or "polar" before running this script.
    code_picked = "polar_scl8"

    if code_picked == "ldpc":
        enc = LDPC5GEncoder(k=k, n=n * num_bits_per_symbol)
        dec = LDPC5GDecoder(enc, num_iter=20)
    elif code_picked == "polar_scl8":
        enc = Polar5GEncoder(k=k, n=n * num_bits_per_symbol)
        dec = Polar5GDecoder(enc, dec_type="SCL", list_size=8)
    elif code_picked == "polar_sc":
        enc = Polar5GEncoder(k=k, n=n * num_bits_per_symbol)
        dec = Polar5GDecoder(enc, dec_type="SC")
    else:
        raise ValueError("code_picked must be 'ldpc' or 'polar_scl8' or 'polar_sc'.")

    model = SystemModel(
        k=k,
        n=n,
        num_bits_per_symbol=num_bits_per_symbol,
        encoder=enc,
        decoder=dec,
        sim_esno=True,
        modulation=modulation,
    )

    wandb.login()
    with wandb.init(
        project="Joint-detection-and-decoding-in-short-packet-communications",
        config={"code_picked": code_picked, "modulation": modulation, "k": k, "n": n},
    ):
        ber, bler = sim_ber(
            model,
            Es_snr_db,
            batch_size=batch_size,
            num_target_block_errors=5000,
            max_mc_iter=100,
        )

        data_ber = [[Es_snr_db[i], ber[i]] for i in range(len(ber))]
        data_bler = [[Es_snr_db[i], bler[i]] for i in range(len(bler))]
        table_ber = wandb.Table(data=data_ber, columns=["SNR", "BER"])
        table_bler = wandb.Table(data=data_bler, columns=["SNR", "BLER"])

        ber_plot = wandb.plot_table(
            vega_spec_name="BER_new",
            data_table=table_ber,
            fields={"x": "SNR", "y": "BER"},
        )
        bler_plot = wandb.plot_table(
            vega_spec_name="BLER_new",
            data_table=table_bler,
            fields={"x": "SNR", "y": "BLER"},
        )
        wandb.log({"BER": ber_plot, "BLER": bler_plot})


if __name__ == "__main__":
    main()
