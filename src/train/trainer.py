"""Training and evaluation utilities for the TurboAE model."""

import os
from pathlib import Path
from typing import Literal, cast

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import wandb
from src.inference import inference_loop
from src.models.components.crc import crc_padding
from src.train.utils import resolve_device

TrainingMode = Literal[
    "encoder",
    "decoder",
    "sync_encoder",
    "sync_block",
    "joint_encoder",
    "joint_decoder",
]


class Trainer:
    """Train, validate, and run inference for a TurboAE model."""

    def __init__(
        self,
        config: wandb.sdk.wandb_config.Config,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        sync_loss_function: torch.nn.Module,
        optimizer_encoder: Optimizer,
        optimizer_decoder: Optimizer,
        optimizer_sync: Optimizer,
        **kwargs: object,
    ) -> None:
        self.local_rank = int(os.environ.get("LOCAL_RANK", config.workstation_GPU_index))
        self.global_rank = int(os.environ.get("RANK", config.workstation_GPU_index))
        self.device = resolve_device(self.local_rank)
        self.config = config
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.sync_loss_function = sync_loss_function
        self.optimizer_encoder = optimizer_encoder
        self.optimizer_decoder = optimizer_decoder
        self.optimizer_sync = optimizer_sync
        # Attach optional components supplied by callers.
        for k, v in kwargs.items():
            setattr(self, k, v)
        # load snapshot if exists
        if Path(self.config.snapshot_path).exists():
            print("Loading snapshot")
            self._load_snapshot(self.config.snapshot_path)

        # Wrap the model with DDP when needed. An inference-only run produces no
        # gradients to synchronize, so it uses the unwrapped model and only
        # communicates model state and evaluation statistics during inference.
        # Training intentionally switches between encoder, synchronization, and
        # decoder paths, leaving different parameter sets unused in each phase.
        if self.config.use_ddp and not getattr(self.config, "is_inference", False):
            if self.device.type == "cuda":
                self.model = DistributedDataParallel(
                    self.model,
                    device_ids=[self.local_rank],
                    find_unused_parameters=True,
                )
            else:
                self.model = DistributedDataParallel(
                    self.model,
                    find_unused_parameters=True,
                )

    @staticmethod
    def _debug_assert_finite(name: str, tensor: torch.Tensor, mode: str) -> None:
        """Raise an informative error when a tensor contains NaN or infinity."""
        # Return immediately in the normal case. Computing detailed statistics can
        # be expensive and is only useful when a non-finite value has been detected.
        if torch.isfinite(tensor).all().item():
            return

        # Diagnostics must not become part of the autograd graph, so inspect a
        # detached view while leaving the original tensor and its gradients intact.
        detached = tensor.detach()
        finite_mask = torch.isfinite(detached)
        finite_count = finite_mask.sum().item()
        total_count = detached.numel()
        nonfinite_count = total_count - finite_count

        # Min, max, and mean are undefined when every value is NaN or infinity.
        # In that case, report only the tensor shape and finite-value counts.
        if finite_count == 0:
            statistics = f"shape={tuple(detached.shape)}, finite=0/{total_count}, nonfinite={nonfinite_count}"
        else:
            # Exclude NaN and infinite entries so they do not contaminate the
            # summary statistics of the remaining valid values.
            finite_values = detached[finite_mask]
            statistics = (
                f"shape={tuple(detached.shape)}, "
                f"finite={finite_count}/{total_count}, "
                f"nonfinite={nonfinite_count}, "
                f"min={finite_values.min().item():.6g}, "
                f"max={finite_values.max().item():.6g}, "
                f"mean={finite_values.mean().item():.6g}"
            )

        raise FloatingPointError(f"{name} became non-finite during {mode}: {statistics}")

    def _prepare_bit_predictions_for_loss(self, predictions: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Validate and clamp predicted bit probabilities before loss evaluation.

        BCE-based losses evaluate logarithms of the predicted probabilities, so
        their inputs are kept away from exactly zero and one. Other losses only
        require predictions to remain in the valid probability range ``[0, 1]``.

        Args:
            predictions: Tensor containing predicted probabilities for each bit.
            mode: Current training or evaluation mode, used in error diagnostics.

        Returns:
            The predictions clamped to the range required by the configured loss.

        Raises:
            FloatingPointError: If ``predictions`` contains NaN or infinity.

        """
        # Fail early with diagnostic information; clamping cannot repair NaN or
        # infinite values and would otherwise hide where numerical instability began.
        self._debug_assert_finite("bit predictions before loss", predictions, mode)

        # Loss names containing "bce" expect probabilities. Keep them inside an
        # epsilon margin to avoid evaluating log(0) in binary cross-entropy.
        loss_function = str(getattr(self.config, "loss_function", "")).lower()
        if "bce" in loss_function:
            eps = float(getattr(self.config, "bce_epsilon", 1e-7))

            # Constrain a user-provided epsilon to a valid interval. The upper
            # limit must stay below 0.5 so the lower bound remains below the upper.
            eps = max(0.0, min(eps, 0.5 - 1e-12))
            return torch.clamp(predictions, eps, 1.0 - eps)

        # Non-BCE losses do not need an epsilon margin, but predictions are still
        # probabilities and therefore must remain within their mathematical range.
        return torch.clamp(predictions, 0.0, 1.0)

    def _clip_and_check_gradients(self, mode: str, *optimizers: Optimizer) -> bool:
        """Validate and clip gradients before the supplied optimizers step.

        ``True`` tells the caller that it is safe to call ``optimizer.step()``.
        ``False`` means a non-finite gradient was found and all participating
        optimizers were cleared, so the current update must be skipped. When
        ``skip_nonfinite_gradients`` is disabled, the original error is raised
        instead of returning ``False``.

        Gradient checking and clipping are both disabled when
        ``config.grad_clip_norm`` is unset or non-positive.

        Args:
            mode: Training mode used to identify errors in diagnostics.
            *optimizers: Optimizers participating in the current update. Passing
                multiple optimizers keeps joint updates all-or-nothing.

        Returns:
            Whether the caller may safely apply the optimizer step.

        """
        # A missing or non-positive threshold explicitly disables this safeguard.
        max_norm = getattr(self.config, "grad_clip_norm", None)
        if max_norm is None or max_norm <= 0:
            return True

        # Optimizer groups can include frozen or currently unused parameters.
        # Only tensors populated by backward() need validation and clipping.
        parameters = [param for optimizer in optimizers for group in optimizer.param_groups for param in group["params"] if param.grad is not None]
        if not parameters:
            return True

        # Inspect each gradient separately first so any failure identifies the
        # offending parameter instead of reporting only a non-finite total norm.
        for param_index, param in enumerate(parameters):
            gradient = param.grad
            if gradient is None:
                continue
            try:
                self._debug_assert_finite(f"gradient for parameter {param_index}", gradient, mode)
            except FloatingPointError as error:
                # A non-finite gradient can either skip this optimizer step or
                # stop training, according to the configured recovery policy.
                if getattr(self.config, "skip_nonfinite_gradients", True):
                    print(f"Skipping {mode} optimizer step because {error}")
                    # Clear every optimizer involved in a joint update; retaining
                    # any gradients could leak this invalid batch into the next one.
                    for optimizer in optimizers:
                        optimizer.zero_grad()
                    return False
                raise

        # clip_grad_norm_ treats all collected gradients as one vector. It scales
        # them in place only when their combined norm exceeds max_norm and returns
        # the norm measured before clipping.
        try:
            total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=float(max_norm), error_if_nonfinite=True)
        except RuntimeError as error:
            if getattr(self.config, "skip_nonfinite_gradients", True):
                print(f"Skipping {mode} optimizer step because gradient norm became non-finite: {error}")
                for optimizer in optimizers:
                    optimizer.zero_grad()
                return False
            raise

        # Keep this final assertion as a defensive check on the value returned by
        # PyTorch, even though error_if_nonfinite normally catches this condition.
        self._debug_assert_finite("gradient norm", total_norm, mode)
        return True

    def _load_snapshot(self, snapshot_path: str | os.PathLike[str]) -> None:
        snapshot = torch.load(snapshot_path, map_location=self.device)
        self.model.load_state_dict(snapshot["model_state_dict"])
        self.epochs_run = snapshot["epoch"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _generate_delay_true_epoch(self, batch_size: int | None = None) -> torch.Tensor:
        """
        Generate a random delay d for each batch, uniformly between 0 and dmax.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, nb, 1) containing the generated delays.

        """
        if batch_size is None:
            batch_size = self.config.batch_size
        if self.config.independent_delay_per_block:
            delay_true = torch.rand(batch_size, self.config.nb, 1).to(self.device) * self.config.dmax
        else:
            delay_true = torch.rand(batch_size, 1, 1).to(self.device) * self.config.dmax
            # expand the delay_true to have the same delay for each block in the block fading channel
            delay_true = delay_true.expand(-1, self.config.nb, -1)
        return delay_true

    def _generate_data_epoch(self) -> DataLoader[torch.Tensor]:
        train_data = torch.randint(
            low=0,
            high=2,
            size=(self.config.num_samples, self.config.input_size, 1),
            dtype=torch.float32,
        ).to(self.device)
        # Tensor implements the Dataset protocol, despite the narrower torch stub.
        train_dataset = cast("Dataset[torch.Tensor]", train_data)
        if self.config.use_ddp:
            return DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                generator=torch.Generator().manual_seed(self.config.seed),
                sampler=DistributedSampler(train_dataset),
            )
        return DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            generator=torch.Generator().manual_seed(self.config.seed),
            shuffle=True,
        )

    def _run_batch(
        self,
        X: torch.Tensor,
        delay_true: torch.Tensor,
        mode: TrainingMode,
    ) -> tuple[float, float, float]:
        # 6 different modes: encoder and decoder
        if mode == "encoder":
            self.optimizer_encoder.zero_grad()
            SNR_db_low = self.config.train_enc_snr_low
            SNR_db_high = self.config.train_enc_snr_high

            predictions = self.model.forward(X, delay_true, SNR_db_low, SNR_db_high, mode="encoder")
            predictions = self._prepare_bit_predictions_for_loss(predictions, mode)
            if self.config.CRC:
                # add parity check bit to X in order to calculate loss, because the prediction predict the length of k+1
                X = crc_padding(X, self.config.CRC)

            loss = self.loss_fn(predictions, X)
            sync_error_mse = 0
            sync_error_mae = 0

        elif mode == "decoder":
            self.optimizer_decoder.zero_grad()
            SNR_db_low = self.config.train_dec_snr_low
            SNR_db_high = self.config.train_dec_snr_high

            predictions = self.model.forward(X, delay_true, SNR_db_low, SNR_db_high, mode="decoder")
            predictions = self._prepare_bit_predictions_for_loss(predictions, mode)
            if self.config.CRC:
                # add parity check bit to X in order to calculate loss
                # because the prediction predict the length of k+1
                X = crc_padding(X, self.config.CRC)
            loss = self.loss_fn(predictions, X)
            sync_error_mse = 0
            sync_error_mae = 0

        elif mode == "sync_encoder":
            self.optimizer_encoder.zero_grad()
            self.optimizer_sync.zero_grad()
            SNR_db_low = self.config.train_sync_snr_low
            SNR_db_high = self.config.train_sync_snr_high

            predicted_delay = self.model.forward(X, delay_true, SNR_db_low, SNR_db_high, mode="sync_encoder")  # shape (batch_size, 1, 1)

            if predicted_delay.dim() == 2:
                predicted_delay = predicted_delay.unsqueeze(2)  # shape (batch_size, 1, 1)
            predicted_delay = predicted_delay.expand_as(delay_true)
            self._debug_assert_finite("predicted delay", predicted_delay, mode)

            loss = self.sync_loss_function(predicted_delay, delay_true)
            self._debug_assert_finite("sync loss", loss, mode)
            sync_error = delay_true - predicted_delay
            sync_error_mse = torch.mean(sync_error**2).item()
            sync_error_mae = torch.mean(torch.abs(sync_error)).item()

        elif mode == "sync_block":
            self.optimizer_sync.zero_grad()
            SNR_db_low = self.config.train_sync_snr_low
            SNR_db_high = self.config.train_sync_snr_high

            predicted_delay = self.model.forward(X, delay_true, SNR_db_low, SNR_db_high, mode="sync_block")

            if predicted_delay.dim() == 2:
                predicted_delay = predicted_delay.unsqueeze(2)  # shape (batch_size, 1, 1)
            predicted_delay = predicted_delay.expand_as(delay_true)
            self._debug_assert_finite("predicted delay", predicted_delay, mode)

            loss = self.sync_loss_function(predicted_delay, delay_true)
            self._debug_assert_finite("sync loss", loss, mode)
            sync_error = delay_true - predicted_delay
            sync_error_mse = torch.mean(sync_error**2).item()
            sync_error_mae = torch.mean(torch.abs(sync_error)).item()

        elif mode == "joint_encoder":
            self.optimizer_encoder.zero_grad()
            self.optimizer_sync.zero_grad()
            SNR_db_low = self.config.train_enc_snr_low
            SNR_db_high = self.config.train_enc_snr_high

            predicted_delay, predicted_bits = self.model.forward(X, delay_true, SNR_db_low, SNR_db_high, mode="joint_encoder")

            if predicted_delay.dim() == 2:
                predicted_delay = predicted_delay.unsqueeze(2)  # shape (batch_size, 1, 1)

            predicted_delay = predicted_delay.expand_as(delay_true)
            self._debug_assert_finite("predicted delay", predicted_delay, mode)

            predictions = self._prepare_bit_predictions_for_loss(predicted_bits, mode)
            if self.config.CRC:
                # add CRC bit to X in order to calculate loss
                # because the prediction predict the length of k+1
                X = crc_padding(X, self.config.CRC)
            loss = self.loss_fn(predictions, X) + self.config.alpha * self.sync_loss_function(predicted_delay, delay_true)
            sync_error = delay_true - predicted_delay
            sync_error_mse = torch.mean(sync_error**2).item()
            sync_error_mae = torch.mean(torch.abs(sync_error)).item()

        elif mode == "joint_decoder":
            self.optimizer_decoder.zero_grad()
            self.optimizer_sync.zero_grad()
            SNR_db_low = self.config.train_dec_snr_low
            SNR_db_high = self.config.train_dec_snr_high

            predicted_delay, predicted_bits = self.model.forward(X, delay_true, SNR_db_low, SNR_db_high, mode="joint_decoder")
            if predicted_delay.dim() == 2:
                predicted_delay = predicted_delay.unsqueeze(2)  # shape (batch_size, 1, 1)
            predicted_delay = predicted_delay.expand_as(delay_true)
            self._debug_assert_finite("predicted delay", predicted_delay, mode)

            predictions = self._prepare_bit_predictions_for_loss(predicted_bits, mode)
            if self.config.CRC:
                # add CRC bit to X in order to calculate loss
                # because the prediction predict the length of k+1
                X = crc_padding(X, self.config.CRC)
            loss = self.loss_fn(predictions, X) + self.config.alpha * self.sync_loss_function(predicted_delay, delay_true)
            sync_error = delay_true - predicted_delay
            sync_error_mse = torch.mean(sync_error**2).item()
            sync_error_mae = torch.mean(torch.abs(sync_error)).item()

        # calculate loss
        self._debug_assert_finite("loss", loss, mode)
        loss.backward()
        if mode == "encoder":
            if self._clip_and_check_gradients(mode, self.optimizer_encoder):
                self.optimizer_encoder.step()
        elif mode == "decoder":
            if self._clip_and_check_gradients(mode, self.optimizer_decoder):
                self.optimizer_decoder.step()
        elif mode == "sync_encoder":
            if self._clip_and_check_gradients(mode, self.optimizer_encoder):
                self.optimizer_encoder.step()
        elif mode == "sync_block":
            if self._clip_and_check_gradients(mode, self.optimizer_sync):
                self.optimizer_sync.step()
        elif mode == "joint_encoder":
            if self._clip_and_check_gradients(mode, self.optimizer_encoder, self.optimizer_sync):
                self.optimizer_encoder.step()
                self.optimizer_sync.step()
        elif mode == "joint_decoder" and self._clip_and_check_gradients(mode, self.optimizer_decoder, self.optimizer_sync):
            self.optimizer_decoder.step()
            self.optimizer_sync.step()

        return loss.item(), sync_error_mse, sync_error_mae

    def _run_epoch(self, mode: TrainingMode) -> tuple[float, float, float]:
        # generate data for each epoch
        X_train = self._generate_data_epoch()
        # set model to training mode
        self.model.train()
        # ! update the learning rate
        for param_group in self.optimizer_encoder.param_groups:
            param_group["lr"] = self.config.learning_rate
        for param_group in self.optimizer_decoder.param_groups:
            param_group["lr"] = self.config.learning_rate
        for param_group in self.optimizer_sync.param_groups:
            param_group["lr"] = self.config.sync_learning_rate

        train_loss_batches = []
        sync_error_mse_batches = []
        sync_error_mae_batches = []
        # iterate over the data
        for batch_index, X in enumerate(X_train):
            if self.config.is_delay:
                delay_true = self._generate_delay_true_epoch(batch_size=X.shape[0])
            loss, sync_error_mse, sync_error_mae = self._run_batch(
                X.to(self.device),
                delay_true,
                mode,
            )
            train_loss_batches.append(loss)
            sync_error_mse_batches.append(sync_error_mse)
            sync_error_mae_batches.append(sync_error_mae)
        train_loss_per_epoch = sum(train_loss_batches) / len(train_loss_batches)
        sync_error_mse_per_epoch = sum(sync_error_mse_batches) / len(sync_error_mse_batches)
        sync_error_mae_per_epoch = sum(sync_error_mae_batches) / len(sync_error_mae_batches)
        # print(
        #     f"\tGPU: {self.global_rank} | "
        #     f"\tEpoch: {self.config.current_epoch} | "
        #     f"\tAverage Train loss ({mode}): {train_loss_per_epoch} | "
        #     f"\tAverage Sync Error (Delay: {self.config.is_delay}): {sync_error_per_epoch} | "
        #     f"\tTime taken: {end_time - start_time:.2f} sec\n"
        # )
        return train_loss_per_epoch, sync_error_mse_per_epoch, sync_error_mae_per_epoch

    def _save_snapshot(self, epoch: int) -> None:
        # Saving model
        print(f"Saving model [epoch {epoch}]")
        if self.config.use_ddp:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.module.state_dict(),
                },
                f"./trained_models/{self.config.save_model_name}/{self.config.save_model_name}_epoch{epoch}.pth",
            )
        else:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                },
                f"./trained_models/{self.config.save_model_name}/{self.config.save_model_name}_epoch{epoch}.pth",
            )

    def val_TurboAE_RayleighBlockFading(self, mode: Literal["encoder", "joint_encoder"]) -> tuple[float, float, float, float, float]:
        """
        Validate the model over the Rayleigh block-fading channel.

        Returns:
            Overall loss, BER, BLER, synchronization MSE, and synchronization MAE.

        """
        test_bce_loss = 0
        test_BER = 0
        test_BLER = 0
        test_sync_loss = 0
        test_sync_error_mse = 0
        test_sync_error_mae = 0
        num_batches = 0
        # generate data for each epoch
        X_val = self._generate_data_epoch()
        # set model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for batch_index, X in enumerate(X_val):
                # valdate on train_enc_snr_low
                SNR_db_low = self.config.train_enc_snr_low
                SNR_db_high = self.config.train_enc_snr_low
                X = X.to(self.device)
                delay_true = self._generate_delay_true_epoch(batch_size=X.shape[0])

                if mode == "encoder":
                    predictions = self.model.forward(X, delay_true, SNR_db_low, SNR_db_high, mode="encoder")
                    predictions = self._prepare_bit_predictions_for_loss(predictions, mode)
                    if self.config.CRC:
                        # add CRC bit to X in order to calculate loss
                        # because the prediction predict the length of k+1
                        X_crc = crc_padding(X, self.config.CRC)
                        test_bce_loss += self.loss_fn(predictions, X_crc).item()

                        # round the predictions to 0 or 1
                        predictions_round = torch.round(predictions)
                        test_BER += calculate_BER(predictions_round, X_crc)
                        test_BLER += calculate_BLER(predictions_round, X_crc)
                    else:
                        test_bce_loss += self.loss_fn(predictions, X).item()

                        # round the predictions to 0 or 1
                        predictions_round = torch.round(predictions)
                        test_BER += calculate_BER(predictions_round, X)
                        test_BLER += calculate_BLER(predictions_round, X)

                    predicted_delay = self.model.forward(X, delay_true, SNR_db_low, SNR_db_high, mode="sync_block")

                    if predicted_delay.dim() == 2:
                        predicted_delay = predicted_delay.unsqueeze(2)  # shape (batch_size, 1, 1)
                    predicted_delay = predicted_delay.expand_as(delay_true)

                    test_sync_loss += self.sync_loss_function(predicted_delay, delay_true).item()

                    sync_error = delay_true - predicted_delay
                    test_sync_error_mse += torch.mean(sync_error**2).item()
                    test_sync_error_mae += torch.mean(torch.abs(sync_error)).item()
                    num_batches += 1

                elif mode == "joint_encoder":
                    predicted_delay, predicted_bits = self.model.forward(X, delay_true, SNR_db_low, SNR_db_high, mode="joint_encoder")
                    if predicted_delay.dim() == 2:
                        predicted_delay = predicted_delay.unsqueeze(2)  # shape (batch_size, 1, 1)
                    predicted_delay = predicted_delay.expand_as(delay_true)

                    predictions = self._prepare_bit_predictions_for_loss(predicted_bits, mode)
                    if self.config.CRC:
                        # add CRC bit to X in order to calculate loss
                        # because the prediction predict the length of k+1
                        X_crc = crc_padding(X, self.config.CRC)
                        test_bce_loss += self.loss_fn(predictions, X_crc).item()

                        # round the predictions to 0 or 1
                        predictions_round = torch.round(predictions)
                        test_BER += calculate_BER(predictions_round, X_crc)
                        test_BLER += calculate_BLER(predictions_round, X_crc)
                    else:
                        test_bce_loss += self.loss_fn(predictions, X).item()

                        # round the predictions to 0 or 1
                        predictions_round = torch.round(predictions)
                        test_BER += calculate_BER(predictions_round, X)
                        test_BLER += calculate_BLER(predictions_round, X)

                    test_sync_loss += self.sync_loss_function(predicted_delay, delay_true).item()
                    sync_error = delay_true - predicted_delay
                    test_sync_error_mse += torch.mean(sync_error**2).item()
                    test_sync_error_mae += torch.mean(torch.abs(sync_error)).item()
                    num_batches += 1

        test_overall_loss_avg = (test_bce_loss + self.config.alpha * test_sync_loss) / num_batches
        test_BER_avg = test_BER / num_batches
        test_BLER_avg = test_BLER / num_batches
        test_sync_error_mse_avg = test_sync_error_mse / num_batches
        test_sync_error_mae_avg = test_sync_error_mae / num_batches
        return (
            test_overall_loss_avg,
            test_BER_avg,
            test_BLER_avg,
            test_sync_error_mse_avg,
            test_sync_error_mae_avg,
        )

    def train_TurboAE_RayleighBlockFading(self) -> None:
        """Train the TurboAE model over the Rayleigh block-fading channel."""
        # Tell wandb to watch what the model gets up to: gradients, weights, and more!
        wandb.watch(self.model, self.loss_fn, log="all", log_freq=5000)
        # record the number of trainable parameters in the model
        wandb.config.update(
            {"trainable_parameters": count_parameters(self.model)},
            allow_val_change=True,
        )
        print(f"Number of total parameters:     {total_parameters(self.model)}")
        print(f"Number of trainable parameters: {count_parameters(self.model)}")
        print("Starting training")
        stuck_counter = 0
        lowest_cost = 2  # set to a high value

        for epoch in range(self.config.start_epoch, self.config.epochs + 1):
            if epoch > 1:
                wandb.config.update({"current_epoch": epoch}, allow_val_change=True)
            else:
                self.config.current_epoch = epoch

            if epoch <= self.config.epochs_integrate_sync:
                train_loss_values = []
                train_sync_error_mse_values = []
                train_sync_error_mae_values = []

                # train for one epoch
                for _ in range(self.config.num_train_enc):
                    _, _, _ = self._run_epoch(mode="encoder")
                for _ in range(self.config.num_train_dec):
                    train_loss, _, _ = self._run_epoch(mode="decoder")
                    train_loss_values.append(train_loss)
                for _ in range(self.config.num_train_sync_enc):
                    _, _, _ = self._run_epoch(mode="sync_encoder")
                for _ in range(self.config.num_train_sync_block):
                    _, train_sync_error_mse, train_sync_error_mae = self._run_epoch(mode="sync_block")
                    train_sync_error_mse_values.append(train_sync_error_mse)
                    train_sync_error_mae_values.append(train_sync_error_mae)

                train_loss_per_epoch = sum(train_loss_values) / len(train_loss_values) if train_loss_values else 0.0
                train_sync_error_mse_per_epoch = (
                    sum(train_sync_error_mse_values) / len(train_sync_error_mse_values) if train_sync_error_mse_values else 0.0
                )
                train_sync_error_mae_per_epoch = (
                    sum(train_sync_error_mae_values) / len(train_sync_error_mae_values) if train_sync_error_mae_values else 0.0
                )

                # compute the validation loss, test on train_enc_snr_low
                (
                    test_overall_loss_avg,
                    test_ber_avg,
                    test_bler_avg,
                    val_sync_error_mse_per_epoch,
                    val_sync_error_mae_per_epoch,
                ) = self.val_TurboAE_RayleighBlockFading(mode="encoder")

            else:
                if epoch == self.config.epochs_integrate_sync + 2:
                    print(f"Epoch {epoch - 1}: Start joint training of encoder and sync and decoder.\n")
                    # ! update the learning rate of decoder and encoder to a larger value for joint training
                    # because the loss landscape of joint training is more complex, we need a larger learning rate to escape local minima
                    for param_group in self.optimizer_encoder.param_groups:
                        param_group["lr"] = self.config.learning_rate * 100
                    for param_group in self.optimizer_decoder.param_groups:
                        param_group["lr"] = self.config.learning_rate * 100
                    # for param_group in self.optimizer_sync.param_groups:
                    #     param_group["lr"] = self.config.sync_learning_rate * 10
                    # update the learning rate in wandb config as well
                    wandb.config.update(
                        {"learning_rate": self.config.learning_rate * 100},
                        allow_val_change=True,
                    )
                    # update stuck counter and lowest cost for joint training
                    stuck_counter = 0

                train_loss_values = []
                train_sync_error_mse_values = []
                train_sync_error_mae_values = []

                # train for one epoch
                for _ in range(self.config.num_train_joint_enc):
                    _, _, _ = self._run_epoch(mode="joint_encoder")
                for _ in range(self.config.num_train_joint_dec):
                    train_loss, train_sync_error_mse, train_sync_error_mae = self._run_epoch(mode="joint_decoder")
                    train_loss_values.append(train_loss)
                    train_sync_error_mse_values.append(train_sync_error_mse)
                    train_sync_error_mae_values.append(train_sync_error_mae)

                train_loss_per_epoch = sum(train_loss_values) / len(train_loss_values) if train_loss_values else 0.0
                train_sync_error_mse_per_epoch = (
                    sum(train_sync_error_mse_values) / len(train_sync_error_mse_values) if train_sync_error_mse_values else 0.0
                )
                train_sync_error_mae_per_epoch = (
                    sum(train_sync_error_mae_values) / len(train_sync_error_mae_values) if train_sync_error_mae_values else 0.0
                )

                # compute the validation loss, test on train_enc_snr_low
                (
                    test_overall_loss_avg,
                    test_ber_avg,
                    test_bler_avg,
                    val_sync_error_mse_per_epoch,
                    val_sync_error_mae_per_epoch,
                ) = self.val_TurboAE_RayleighBlockFading(mode="joint_encoder")
            # update batch_size, learning_rate, stuck_counter and other parameters if needed
            if epoch in {1, self.config.epochs_integrate_sync + 2}:
                lowest_cost = test_overall_loss_avg
                print(f"Epoch {epoch}: Initialize lowest cost to {lowest_cost:.6f}\n")
            if test_overall_loss_avg > lowest_cost:
                stuck_counter += 1
                if stuck_counter > self.config.max_stuck_counter and self.config.batch_size < 990:
                    print(f"\nLoss Stuck for {stuck_counter} epochs, increase batch size\n")
                    wandb.config.update(
                        {"batch_size": self.config.batch_size * 2},
                        allow_val_change=True,
                    )
                    # self.config.batch_size = self.config.batch_size * 2
                    wandb.config.update(
                        {"num_samples": self.config.num_samples * 2},
                        allow_val_change=True,
                    )
                    # self.config.num_samples = self.config.num_samples * 2

                    stuck_counter = 0
                if stuck_counter > self.config.max_stuck_counter and self.config.batch_size > 990 and self.config.learning_rate >= 1e-5:
                    print(f"\nLoss Stuck for {stuck_counter} epochs, reduce learning rate\n")
                    wandb.config.update(
                        {"learning_rate": self.config.learning_rate / 10},
                        allow_val_change=True,
                    )
                    # self.config.learning_rate = self.config.learning_rate / 10

                    stuck_counter = 0
                if self.config.learning_rate <= 1e-6:
                    print(f"\nLoss Stuck for {stuck_counter} epochs, but stop updating batch size and learning rate\n")
            else:
                stuck_counter = 0
                lowest_cost = test_overall_loss_avg

            print(
                f"Epoch: {epoch:>2}/{self.config.epochs:<2} | "
                f"Train loss: {train_loss_per_epoch:>10.6f} | "
                f"Train Sync MSE: {train_sync_error_mse_per_epoch:>10.6f} | "
                f"Train Sync MAE: {train_sync_error_mae_per_epoch:>10.6f} | "
                f"Val. loss: {test_overall_loss_avg:>10.6f} | "
                f"Val. BER ({self.config.train_enc_snr_low:>2} dB): {test_ber_avg:>10.6f} | "
                f"Val. BLER ({self.config.train_enc_snr_low:>2} dB): {test_bler_avg:>10.6f} | "
                f"Val. Sync Error MSE ({self.config.train_enc_snr_low:>2} dB): {val_sync_error_mse_per_epoch:>10.6f} | "
                f"Val. Sync Error MAE ({self.config.train_enc_snr_low:>2} dB): {val_sync_error_mae_per_epoch:>10.6f} | "
                f"Batch size: {self.config.batch_size:>4} | "
                f"Learning rate: {self.config.learning_rate:>10.6e} | "
                f"Sync learning rate: {self.config.sync_learning_rate:>10.6e} | "
                f"Stuck: {stuck_counter:>3}"
            )

            # Log the train_loss_one_epoch and val_loss_one_epoch to wandb
            wandb.log(
                {
                    "train_loss": train_loss_per_epoch,
                    "val_loss": test_overall_loss_avg,
                    "val_BER": test_ber_avg,
                    "val_BLER": test_bler_avg,
                    "train_sync_error_mse": train_sync_error_mse_per_epoch,
                    "train_sync_error_mae": train_sync_error_mae_per_epoch,
                    "val_sync_error_mse": val_sync_error_mse_per_epoch,
                    "val_sync_error_mae": val_sync_error_mae_per_epoch,
                },
                step=epoch,
            )

            # save snapshot
            if self.global_rank == 0 and epoch % self.config.save_every == 0:
                self._save_snapshot(epoch)

    def inference(self, test_dataloader: DataLoader[object]) -> None:
        """Run inference over a test data loader."""
        inference_loop(
            config=self.config,
            model=self.model,
            test_dataloader=test_dataloader,
            device=self.device,
            is_delay=self.config.is_delay,
            CRC=self.config.CRC,
            candidate_list_size=self.config.candidate_list_size,
        )


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.

    Args:
        model (torch.nn.Module): The model to count the parameters of.

    Returns:
        int: The total number of trainable parameters in the model.

    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def total_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of total parameters in a model.

    Args:
        model (torch.nn.Module): The model to count the parameters of.

    Returns:
        int: The total number of trainable parameters in the model.

    """
    return sum(p.numel() for p in model.parameters())


def calculate_BER(
    predictions: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    Calculate the bit error rate (BER) of the model's predictions.

    Args:
        predictions (torch.Tensor): The model's predictions.
        labels (torch.Tensor): The true labels.

    Returns:
        float: The bit error rate of the model's predictions.

    """
    return torch.sum(torch.abs(predictions - labels)).item() / labels.numel()


def calculate_BLER(
    predictions: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    Calculate the block error rate (BLER) of the model's predictions.

    Args:
        predictions (torch.Tensor): The model's predictions.
        labels (torch.Tensor): The true labels.

    Returns:
        float: The block error rate of the model's predictions.

    """
    return torch.sum(torch.any(torch.abs(predictions - labels), dim=1)).item() / labels.shape[0]


def calculate_sync_error(
    predicted_starting_point_probability: torch.Tensor,
    true_starting_point_onehot: torch.Tensor,
) -> float:
    """
    Calculate the synchronization error between the predicted and true starting points.

    Args:
        predicted_starting_point_probability (torch.Tensor): The predicted starting point probabilities of shape (batch_size, dmax/ts+1).
        true_starting_point_onehot (torch.Tensor): The true starting point one-hot encoded tensor of shape (batch_size, dmax/ts+1).

    Returns:
        float: The synchronization error between the predicted and true starting points.

    """
    # The mean value of (true_starting_point_onehot - predicted_starting_point_probability)^2 / tp
    return torch.mean(
        torch.sum(
            (true_starting_point_onehot - predicted_starting_point_probability) ** 2,
            dim=1,
        )
    ).item()


def add_even_parity_bit(x: torch.Tensor) -> torch.Tensor:
    """
    Adds an even parity check bit along dimension 1 of a binary tensor.

    Args:
        x (torch.Tensor): Binary tensor of shape (batch, k, 1) with values 0 or 1.

    Returns:
        torch.Tensor: Tensor of shape (batch, k+1, 1) with even parity bit appended.

    Raises:
        ValueError: If the input contains values other than zero and one.

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
