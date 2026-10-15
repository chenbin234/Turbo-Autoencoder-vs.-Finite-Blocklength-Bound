"""Run TurboAE inference and calculate error statistics."""

from collections import Counter
from collections.abc import Iterable
from itertools import chain
from typing import cast

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Subset

import wandb
from src.models.components.crc import candidate_list_decoding, crc_padding


def make_inference_dataloader(dataset: Dataset[torch.Tensor], batch_size: int, use_ddp: bool = False) -> DataLoader[torch.Tensor]:
    """
    Split test samples across ranks without padding or dropping any samples.

    Unlike a training DistributedSampler, this permits unequal shard lengths,
    including empty shards when there are fewer samples than workers.

    Returns:
        A loader for this rank's samples, or the full dataset without DDP.

    """
    if use_ddp:
        indices = range(dist.get_rank(), len(dataset), dist.get_world_size())
        dataset = Subset(dataset, indices)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def inference_loop(
    config: wandb.sdk.wandb_config.Config,
    model: torch.nn.Module,
    test_dataloader: Iterable[object],
    device: torch.device,
    is_delay: bool,
    CRC: int,
    candidate_list_size: int,
) -> torch.nn.Module:
    """
    Evaluate local test samples and combine statistics across all DDP workers.

    DDP callers must supply disjoint data shards and call this on every rank.
    Forward passes use the unwrapped model, so unequal batch counts are safe.
    Only rank zero prints and logs the combined results.

    Returns:
        The evaluated, unwrapped model.

    Raises:
        ValueError: If the channel is unsupported or the global dataset is empty.

    """
    if config.channel != "block_fading" or not is_delay:
        raise ValueError("Inference requires the block-fading channel with delay.")

    use_ddp = bool(getattr(config, "use_ddp", False))
    rank = dist.get_rank() if use_ddp else 0
    world_size = dist.get_world_size() if use_ddp else 1
    if isinstance(model, DistributedDataParallel):
        model = model.module
    model.eval()

    if use_ddp:
        # Match rank zero's saved model, including BatchNorm running statistics.
        # No DDP forward-time broadcasts are needed after this synchronization.
        with torch.no_grad():
            for tensor in chain(model.parameters(), model.buffers()):
                dist.broadcast(tensor, src=0)
        # Channel noise and delays must use distinct streams on different ranks.
        torch.manual_seed(int(config.seed) + rank)

    inference_snr_db = np.arange(config.test_snr_low, config.test_snr_high + 0.5, config.test_snr_interval)
    bit_error = int(config.input_size)
    infer_ber_list: list[float] = []
    infer_bler_list: list[float] = []
    if rank == 0:
        print(f"Starting inference on {world_size} worker(s) ...", flush=True)
        wandb.define_metric("custom_step")
        wandb.define_metric("BER v.s. SNR (AWGN)", step_metric="custom_step")
        wandb.define_metric("BLER v.s. SNR (AWGN)", step_metric="custom_step")

    with torch.no_grad():
        for snr_db in inference_snr_db:
            # These accumulators have fixed size regardless of the sample count.
            error_histogram = torch.zeros(bit_error + 1, dtype=torch.int64, device=device)
            # Sum of squared delay errors, absolute delay errors, and their count.
            sync_totals = torch.zeros(3, dtype=torch.float64, device=device)
            floor_diff_counts: Counter[int] = Counter()

            for batch_index, data in enumerate(test_dataloader, 1):
                X = cast("torch.Tensor", data).to(device)
                batch_size = X.shape[0]
                delay_true = torch.rand(batch_size, 1, device=device) * getattr(config, "dmax", 12)
                sync_predictions, predicted_bits = model(X, delay_true, SNR_db_low=snr_db, SNR_db_high=snr_db, mode="joint_encoder")
                predictions = torch.clamp(predicted_bits, 0.0, 1.0)
                predictions_round = torch.round(predictions)
                # if add_parity_check_bit:
                #     parity_check = has_even_number_of_ones(predictions_round)
                #     predictions_round = torch.round(conditional_min_flip(predictions, parity_check))[:, :-1, :]

                if CRC > 0:
                    predictions_round = candidate_list_decoding(
                        predicted_probabilities_crc=predictions, CRC=CRC, candidate_list_size=config.candidate_list_size
                    )

                errors_per_sample = (predictions_round != X).reshape(batch_size, -1).sum(dim=1)
                error_histogram += torch.bincount(errors_per_sample, minlength=bit_error + 1)
                sync_error = (delay_true - sync_predictions).to(torch.float64)
                sync_totals[0] += sync_error.square().sum()
                sync_totals[1] += sync_error.abs().sum()
                sync_totals[2] += sync_error.numel()
                floor_diff = (torch.floor(sync_predictions) - torch.floor(delay_true)).to(torch.int64)
                values, counts = torch.unique(floor_diff, return_counts=True)
                # Transfer the small histogram once, rather than one GPU scalar at a time.
                for value, count in zip(values.cpu().tolist(), counts.cpu().tolist()):
                    floor_diff_counts[value] += count

                if rank == 0 and batch_index % 1000 == 0:
                    print(f"SNR {snr_db}: rank 0 processed {batch_index} local batches.", flush=True)

            if use_ddp:
                # All ranks participate even when their shard has zero samples.
                dist.all_reduce(error_histogram, op=dist.ReduceOp.SUM)
                dist.all_reduce(sync_totals, op=dist.ReduceOp.SUM)
                # Floor differences can be negative and are not assumed bounded.
                # Exchange only the small counters, never per-sample predictions.
                gathered_counts: list[dict[int, int] | None] = [None] * world_size
                dist.all_gather_object(gathered_counts, dict(floor_diff_counts))
                floor_diff_counts = Counter()
                for counts_by_rank in gathered_counts:
                    floor_diff_counts.update(cast("dict[int, int]", counts_by_rank))

            histogram = error_histogram.cpu().tolist()
            num_samples = sum(histogram)
            if num_samples == 0:
                raise ValueError("Inference requires at least one test sample across all ranks.")
            # Weight every sample equally, including partial batches and uneven shards.
            bit_errors = sum(errors * count for errors, count in enumerate(histogram))
            block_errors = num_samples - histogram[0]
            ber = bit_errors / (num_samples * bit_error)
            bler = block_errors / num_samples
            squared_error, absolute_error, sync_count = sync_totals.cpu().tolist()
            sync_mse = squared_error / sync_count
            sync_mae = absolute_error / sync_count
            infer_ber_list.append(ber)
            infer_bler_list.append(bler)

            if rank == 0:
                print(
                    f"For SNR = {snr_db}, sync delay MSE is {sync_mse}, "
                    f"sync delay MAE is {sync_mae}, floor diff counts are "
                    f"{dict(sorted(floor_diff_counts.items()))}",
                    flush=True,
                )
                print(
                    f"For SNR = {snr_db}, the average BER is {ber}, the average BLER is {bler} ({num_samples} samples across {world_size} workers)",
                    flush=True,
                )
                wandb.log({
                    "custom_step": float(snr_db),
                    "BER v.s. SNR (AWGN)": ber,
                    "BLER v.s. SNR (AWGN)": bler,
                    "Inference samples": num_samples,
                    "Inference bit errors": bit_errors,
                    "Inference block errors": block_errors,
                    "Inference sync delay MSE": sync_mse,
                    "Inference sync delay MAE": sync_mae,
                    "Inference bit error distribution": wandb.Table(
                        data=[[errors, count] for errors, count in enumerate(histogram)],
                        columns=["Bit errors", "Count"],
                    ),
                    "Inference floor difference distribution": wandb.Table(
                        data=[[value, count, count / sync_count] for value, count in sorted(floor_diff_counts.items())],
                        columns=["Floor difference", "Count", "Ratio"],
                    ),
                })

    if rank == 0:
        print("The snr_db values for inference are: ", inference_snr_db)
        print("The average bit error rate is: ", [round(value, 7) for value in infer_ber_list])
        print("The average block error rate is: ", [round(value, 7) for value in infer_bler_list])
        table_ber = wandb.Table(
            data=[[float(snr), ber] for snr, ber in zip(inference_snr_db, infer_ber_list)],
            columns=["SNR", "BER"],
        )
        table_bler = wandb.Table(
            data=[[float(snr), bler] for snr, bler in zip(inference_snr_db, infer_bler_list)],
            columns=["SNR", "BLER"],
        )
        wandb.log({
            "BER_BlockFadingChannel": wandb.plot_table(
                vega_spec_name="BER_BlockFadingChannel", data_table=table_ber, fields={"x": "SNR", "y": "BER"}
            ),
            "BLER_BlockFadingChannel": wandb.plot_table(
                vega_spec_name="BLER_BlockFadingChannel", data_table=table_bler, fields={"x": "SNR", "y": "BLER"}
            ),
        })
        print("Finished inference!", flush=True)
    return model


def compute_ber_bler(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the bit error rate and block error rate.

    Args:
        predictions (torch.Tensor): The predicted values of shape (batch_size, k, 1).
        targets (torch.Tensor): The target values of shape (batch_size, k, 1).

    Returns:
        The bit error rate and block error rate.

    """
    # compute the bit error rate
    ber = torch.sum(torch.abs(predictions - targets)) / (targets.shape[0] * targets.shape[1])

    # compute the block error rate
    bler = torch.sum(torch.any(torch.abs(predictions - targets), dim=1)) / targets.shape[0]

    return ber, bler


def calculate_bit_error_statistics(
    prediction_raw: torch.Tensor,
    predictions_rounds: torch.Tensor,
    targets: torch.Tensor,
    bit_error: int,
) -> dict[str, dict[int, int] | list[float] | list[int]]:
    """
    Separate raw wrong-bit probabilities by ground-truth value.

    The output includes values for ground-truth zero and one separately.

    Args:
        prediction_raw: Probabilities of one with shape ``(batch, k, 1)``.
        predictions_rounds: Thresholded bits with shape ``(batch, k, 1)``.
        targets: Ground-truth bits with shape ``(batch, k, 1)``.
        bit_error: Maximum number of errors represented in the histogram.

    Returns:
        The error histogram, wrong-bit probabilities, and erroneous bit indices.

    """
    # 1) ensure integer 0/1 for “boolean” tensors
    predictions_rounds = predictions_rounds.int()
    targets = targets.int()

    # 2) mask of wrong decisions (XOR)
    error_mask = predictions_rounds != targets  # shape (batch, k, 1)

    # 3) #wrong bits per sample  →  histogram
    errors_per_sample = error_mask.squeeze(-1).sum(dim=1)  # (batch,)
    error_distribution = {i: int((errors_per_sample == i).sum().item()) for i in range(bit_error + 1)}

    # ------------------------------------------------------------------
    # 4) split wrong-bit probabilities by the GT value in X
    # ------------------------------------------------------------------
    # masks
    wrong_and_gt1 = error_mask & (targets == 1)
    wrong_and_gt0 = error_mask & (targets == 0)

    error_probabilities_gt1 = [float(value) for value in prediction_raw[wrong_and_gt1]]
    error_probabilities_gt0 = [float(value) for value in prediction_raw[wrong_and_gt0]]

    # ------------------------------------------------------------------
    # 5) coordinates (sample, bit) for *all* wrong bits
    # ------------------------------------------------------------------
    wrong_coords = torch.nonzero(error_mask.squeeze(-1), as_tuple=False)
    error_indices = [int(bit_index) for _, bit_index in wrong_coords]

    return {
        "error_distribution": error_distribution,
        "error_probabilities_gt1": error_probabilities_gt1,
        "error_probabilities_gt0": error_probabilities_gt0,
        "error_indices": error_indices,
    }


def has_even_number_of_ones(x: torch.Tensor) -> torch.Tensor:
    """
    Checks if each sample in the batch has an even number of ones along the last dimension.

    Args:
        x (torch.Tensor): Binary tensor of shape (batch, k+1, 1) with values 0 or 1.

    Returns:
        torch.Tensor: Bool tensor of shape (batch, 1), True if even number of ones, False if odd.

    Raises:
        ValueError: If the input contains values other than zero and one.

    """
    if not torch.all((x == 0) | (x == 1)):
        raise ValueError("Input tensor must contain only 0s and 1s.")

    # Count ones along the last dimension (dim=2)
    ones_count = x.sum(dim=1)  # shape: (batch, 1)

    # Even number check → remainder == 0
    return ones_count % 2 == 0


def conditional_min_flip(
    prediction: torch.Tensor,
    check: torch.Tensor,
) -> torch.Tensor:
    """
    Flip the least-confident prediction when a parity check fails.

    If check[b] == True: keep prediction[b] unchanged.
    If check[b] == False:
        •	Find the element in prediction[b, :, 0] closest to 0.5.
        •	Replace that element with 1 - value.
        •	Leave all other elements unchanged.

    Args:
        prediction: Tensor of shape (batch, k+1, 1), values in [0,1]
        check: Bool tensor of shape (batch, 1)

    Returns:
        Modified tensor of shape (batch, k+1, 1)

    """
    output = prediction.clone()
    batch = prediction.shape[0]

    # Find index of value closest to 0.5 for each batch
    vals = prediction.squeeze(-1)  # (batch, k+1)
    idx = torch.argmin(torch.abs(vals - 0.5), dim=1)  # (batch,)

    # Create index grid for scatter update
    batch_idx = torch.arange(batch, device=prediction.device)

    # Only flip for check == False
    mask = ~check.view(-1)  # (batch,)

    # Gather original values to flip
    chosen_vals = vals[batch_idx, idx]  # (batch,)
    flipped_vals = 1.0 - chosen_vals

    # Assign back only for masked batches
    output[batch_idx[mask], idx[mask], 0] = flipped_vals[mask]

    return output
