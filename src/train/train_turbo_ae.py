"""Script to train the Turbo AutoEncoder model with the block fading channel and delay."""

# import libraries
import argparse
import os
import pathlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import torch
from torch import nn, optim
from torch.distributed import barrier, broadcast_object_list, destroy_process_group, get_rank
from torch.utils.data import DataLoader

import wandb
from src.features import InfobitDataset
from src.inference.inference_turboae import make_inference_dataloader
from src.models import TurboAutoencoderSync, generate_permutation_array
from src.train.trainer import Trainer
from src.train.utils import (
    broadcast_seed,
    build_loss_fn,
    build_sync_loss_fn,
    ddp_setup,
    get_git_metadata,
    resolve_device,
    setup_seed,
)


@dataclass
class TurboAEConfig:
    """TurboAE hyperparameters with their default values."""

    model_type: str = "Turbo_AutoEncoder"
    run_name: str | None = "unknown"
    git_branch: str = "unknown"
    git_commit_hash: str = "unknown"
    trainable_parameters: int = 0
    epochs: int = 700
    epochs_integrate_sync: int = 500
    num_samples: int = 5000
    batch_size: int = 5000
    start_epoch: int = 1
    print_every: int | None = None
    save_every: int = 700
    in_channels: int = 1
    input_size: int = 64
    k: int = 64  # number of information bits
    CRC: int = 3  # number of CRC bits, option: {0, 1, 2, 3, 6, 11, 16}
    interleaver_type: str = "sionna_polar"  # "random" or "sionna_polar"
    candidate_list_size: int = 8  # number of candidates to keep after CRC check
    n: int = 64  # code length (complex)
    n_M: int = 0
    F: int = 30
    num_iteration: int = 6
    num_train_enc: int = 100  # Number of iterations for training the encoder in each epoch
    num_train_dec: int = 500  # Number of iterations for training the decoder in each epoch
    num_train_sync_enc: int = 50  # Sync-encoder steps per epoch.
    num_train_sync_block: int = 50  # Sync-block steps per epoch.
    num_train_joint_enc: int = 50  # Joint encoder/sync steps per epoch.
    num_train_joint_dec: int = 100  # Joint decoder/sync steps per epoch.
    dec_num_unit: int = 100
    dec_num_layer: int = 5
    add_parity_check_bit: bool = False  # Add a parity bit at the encoder input.
    fixed_pilot_length: int = 0  # Fixed pilot length for each block (0 means no fixed pilot)
    train_enc_snr_high: float = 15.0  # dB
    train_enc_snr_low: float = 15.0  # dB
    train_dec_snr_high: float = 15.0  # dB
    train_dec_snr_low: float = 15.0  # dB
    train_sync_snr_high: float = 15.0  # dB
    train_sync_snr_low: float = 15.0  # dB
    test_snr_high: float = 19  # dB
    test_snr_low: float = 18  # dB
    test_snr_interval: float = 2.0  # dB
    test_data_samples: int = 100_000  # Total test samples per SNR across all workers.
    channel: str = "block_fading"
    num_taps: int = 1  # Number of taps for the Proakis channel
    is_delay: bool = True  # Whether to include delay in the channel
    independent_delay_per_block: bool = False  # Sample delay independently per block.
    nb: int = 4
    tp: int = 1
    ts: float = 0.125
    N_up: int = 8
    dmax: int = 4
    alpha: float = 0.5
    beta: float = 1
    pretrained: str | None = (
        "./trained_models/Turbo_AutoEncoder_k64n64_CRC3_20261012_08_17_54/Turbo_AutoEncoder_20261012_08_17_54_epoch700.pth"  # Path to the pretrained model: "./trained_models/Turbo_AutoEncoder_20241205_10_38_41/Turbo_AutoEncoder_20241205_10_38_41_epoch10.pth"
    )
    is_inference: bool = True  # Skip training entirely and only run inference on "pretrained".
    loss_function: str = "BCE"  # "BCE", "MSE", "ProductLoss", "MaxLoss", "SmoothMaxLoss", "LogSumExpLoss", "PNormLoss", "HybridBCESoftBLERLoss", "HybridBCEProductLoss", "HybridBCEMaxLoss", "HybridBCESmoothMaxLoss", "HybridBCELogSumExpLoss", "HybridBCEPNormLoss"
    bce_epsilon: float = 1e-7  # Avoid exact 0/1 probabilities in BCELoss.
    grad_clip_norm: float = 1.0  # Prevent rare exploding gradients from poisoning weights.
    skip_nonfinite_gradients: bool = True
    learning_rate: float = 1e-4
    is_train_sync_block: bool = True
    sync_loss_function: str = "smooth_l1"  # Options: "MSE", "MAE", "Huber", "smooth_l1", "log_cosh", or "relative_mse""
    sync_learning_rate: float = 1e-4
    sync_hidden_channels: int = 64
    sync_num_layers: int = 4
    sync_kernel_size: int = 5
    sync_activation: str = "gelu"
    sync_fuse_blocks: bool = True  # if true, output delay shape (batch, 1, 1), if false, output delay shape (batch, nb, 1)
    sync_separate_q_eps: bool = True
    smooth_l1_beta: float = 0.2
    save_model_name: str = ""
    snapshot_path: str = ""
    use_ddp: bool = False
    workstation_GPU_index: int = 0
    max_stuck_counter: int = 39
    encoder_output_power_constraint: str = "per_symbol"  # "per_symbol", "per_block", "average", or "none"
    # "average" sets mean complex-symbol power to 1 across the whole batch, allowing unequal symbol/block powers.
    receiver_resample_method: str = "symbol_lmmse"  # "symbol_lmmse", "lmmse", "linear", or "nearest"
    receiver_resample_sample_phase: float = 0.5  # used by "linear" and "nearest"
    receiver_mmse_regularization: float = 0.2  # used by "symbol_lmmse" and "lmmse"
    enforce_qpsk_symbols: bool = (
        False  # if true, enforce the encoder output to be in the QPSK constellation; if false, allow the encoder to learn any constellation
    )
    is_soft_qpsk: bool = False  # if true, pull the encoder output toward normalized QPSK without forcing it; excludes enforce_qpsk_symbols
    soft_qpsk_alpha: float = 0.8  # used by is_soft_qpsk, in [0, 1]: 0 leaves unit-energy symbols unchanged, 1 snaps them onto QPSK
    p_array_seed: int = 3
    seed: int = 0


def make(
    config: wandb.sdk.wandb_config.Config,
) -> tuple[
    TurboAutoencoderSync,
    nn.Module,
    nn.Module,
    optim.Optimizer,
    optim.Optimizer,
    optim.Optimizer,
    DataLoader,
]:
    """
    Build the model, losses, optimizers, and test data loader.

    Returns:
        The model and all components required for training and evaluation.

    Raises:
        ValueError: If the channel configuration is unsupported, or if an
            inference-only run has no pretrained checkpoint.

    """
    # An inference-only run never updates the weights, so it is only meaningful
    # when a trained checkpoint is supplied.
    if config.is_inference and not config.pretrained:
        raise ValueError("Inference-only runs (is_inference=True) require 'pretrained' to point to a checkpoint.")

    # test dataset
    test_dataset = InfobitDataset(num_samples=config.test_data_samples, k=config.input_size)
    test_dataloader = make_inference_dataloader(test_dataset, batch_size=config.batch_size, use_ddp=config.use_ddp)

    # Initialize the model
    model = TurboAutoencoderSync(autoencoder_config=config)

    print("interleaver_type:", model.interleaver_type)
    print("p_array_seed:", config.p_array_seed)
    print(
        "p_array:",
        generate_permutation_array(k=config.k + config.CRC, seed=config.p_array_seed, interleaver_type=model.interleaver_type),
    )

    if config.pretrained:
        device = resolve_device(int(os.environ.get("LOCAL_RANK", config.workstation_GPU_index)))
        checkpoint = torch.load(config.pretrained, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Pretrained model loaded")

    loss_fn = build_loss_fn(config)

    # Make the loss and optimizer
    if config.channel == "block_fading" and config.is_delay:
        optimizer_encoder = optim.Adam(model.encoder.parameters(), lr=config.learning_rate)
        optimizer_decoder = optim.Adam(
            filter(lambda p: p.requires_grad, model.decoder.parameters()),
            lr=config.learning_rate,
        )
        optimizer_sync = optim.Adam(model.sync_block.parameters(), lr=config.sync_learning_rate)
        sync_loss_function = build_sync_loss_fn(config)
    else:
        raise ValueError("Invalid channel type")

    return (
        model,
        loss_fn,
        sync_loss_function,
        optimizer_encoder,
        optimizer_decoder,
        optimizer_sync,
        test_dataloader,
    )


def model_pipeline(hyperparameters: TurboAEConfig) -> TurboAutoencoderSync:
    """
    Train and evaluate a model with the supplied hyperparameters.

    When ``is_inference`` is set, training is skipped and only the pretrained
    model is evaluated.

    Returns:
        The trained model, or the pretrained model for an inference-only run.

    Raises:
        ValueError: If the channel configuration is unsupported.

    """
    # Only rank zero creates an online W&B run. Disabled runs still provide a
    # local wandb.config object to the other DDP workers.
    wandb_mode = "disabled" if hyperparameters.use_ddp and get_rank() != 0 else "online"
    with wandb.init(
        project="Joint-detection-and-decoding-in-short-packet-communications",
        config=asdict(hyperparameters),
        name=hyperparameters.run_name,
        group=hyperparameters.model_type,
        mode=wandb_mode,
        save_code=True,
    ):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # make the model, data, and optimization problem
        (
            model,
            loss_fn,
            sync_loss_function,
            optimizer_encoder,
            optimizer_decoder,
            optimizer_sync,
            test_dataloader,
        ) = make(config)
        print(model)

        # and use them to train the model
        trainer = Trainer(
            config=config,
            model=model,
            loss_fn=loss_fn,
            sync_loss_function=sync_loss_function,
            optimizer_encoder=optimizer_encoder,
            optimizer_decoder=optimizer_decoder,
            optimizer_sync=optimizer_sync,
        )
        # Inference-only runs evaluate the pretrained weights loaded in make().
        if config.is_inference:
            print("Inference-only run: skipping training.")
        elif config.channel == "block_fading":
            trainer.train_TurboAE_RayleighBlockFading()
        else:
            raise ValueError("Invalid channel")

        if config.use_ddp:
            # Every worker evaluates its test-data shard. Inference synchronizes
            # model state once and combines statistics at the end of each SNR.
            barrier()
            trainer.inference(test_dataloader=test_dataloader)
            barrier()
            destroy_process_group()
        else:
            trainer.inference(test_dataloader=test_dataloader)

        # finish logging to wandb
        wandb.finish()

    return model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training run."""
    parser = argparse.ArgumentParser(description="Train the Turbo AutoEncoder")
    parser.add_argument(
        "--p-array-seed",
        type=int,
        default=3,
        help="Seed used to generate the interleaver permutation array.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2,
        help="Global random seed used for this training run.",
    )
    # With store_true, argparse sets the attribute to False by default and to True when the flag appears.
    parser.add_argument(
        "--use-ddp",
        action="store_true",
        help="Enable DistributedDataParallel; launch with torchrun when using this option.",
    )
    return parser.parse_args()


def main() -> None:
    """Configure and run Turbo AutoEncoder training."""
    args = parse_args()

    # update the configuration for the model
    git_branch, git_commit_hash = get_git_metadata()

    config = TurboAEConfig(
        run_name=os.environ.get("SLURM_JOB_NAME"),
        git_branch=git_branch,
        git_commit_hash=git_commit_hash,
        p_array_seed=args.p_array_seed,
        seed=args.seed,
        use_ddp=args.use_ddp,
    )

    if config.use_ddp:
        ddp_setup()

    # Only the process that owns the W&B run needs credentials. Offline and
    # disabled modes intentionally do not contact the W&B service.
    is_primary = not config.use_ddp or get_rank() == 0
    # fetch the WANDB_MODE from the environment, defaulting to "online" if not set.
    wandb_mode = os.environ.get("WANDB_MODE", "online").lower()
    if is_primary and wandb_mode not in {"disabled", "offline"}:
        wandb.login()

    # get the current time for the model name
    time = datetime.now(tz=timezone.utc).astimezone().strftime("%Y%m%d_%H_%M_%S")
    if config.use_ddp:
        synchronized_time = [time if get_rank() == 0 else ""]
        broadcast_object_list(synchronized_time, src=0)
        time = synchronized_time[0]
    # defining model save location
    save_model_name = "Turbo_AutoEncoder_" + time
    # save the model name in the config for later use
    config.save_model_name = save_model_name
    config.snapshot_path = f"./trained_models/{save_model_name}/{save_model_name}_epoch"

    # create the folder if it doesn't exist
    if config.use_ddp:
        # agree on a base-seed
        base_seed = broadcast_seed(config.seed)
        setup_seed(base_seed)
        if get_rank() == 0 and not config.is_inference:
            pathlib.Path(f"./trained_models/{save_model_name}").mkdir(exist_ok=True, parents=True)
        barrier()
    else:
        setup_seed(config.seed)
        # create the folder if it doesn't exist; an inference-only run saves nothing
        if not config.is_inference and not pathlib.Path(f"./trained_models/{save_model_name}").exists():
            pathlib.Path(f"./trained_models/{save_model_name}").mkdir(exist_ok=True, parents=True)

    # Build, train and analyze the model with the pipeline
    model_pipeline(config)


if __name__ == "__main__":
    main()
