"""CRC padding and checking for batched tensors of shape (batch_size, k, 1)."""

from functools import lru_cache

import torch

_CRC_POLYNOMIALS = {
    0: 0b1,
    1: 0b11,  # x + 1
    2: 0b111,  # x^2 + x + 1
    3: 0b1011,  # x^3 + x + 1
    # CRC6, CRC11, and CRC16 from sionna.phy.fec.crc.CRCEncoder.
    6: 0b1100001,  # x^6 + x^5 + 1
    11: 0b111000100001,  # x^11 + x^10 + x^9 + x^5 + 1
    16: 0b10001000000100001,  # x^16 + x^12 + x^5 + 1
}


def _validate_crc_input(bits: torch.Tensor, CRC: int) -> None:
    """Check the supported CRC widths and batched bit layout."""
    if CRC not in _CRC_POLYNOMIALS:
        raise ValueError("CRC must be one of {0, 1, 2, 3, 6, 11, 16}.")
    if bits.ndim != 3 or bits.shape[-1] != 1:
        raise ValueError("Expected a tensor of shape (batch_size, sequence_length, 1).")


@lru_cache(maxsize=32)
def _crc_generator_rows(k: int, CRC: int) -> tuple[tuple[int, ...], ...]:
    """Build the parity generator matrix using Sionna's shift/XOR recurrence."""
    polynomial = _CRC_POLYNOMIALS[CRC]
    leading_bit = 1 << CRC
    remainder = 1 << (CRC - 1)
    rows = []
    # Start with the last unit vector and shift its remainder to obtain
    # preceding rows, avoiding a full polynomial division for every row.
    for _ in range(k):
        remainder <<= 1
        if remainder & leading_bit:
            remainder ^= polynomial
        rows.append(tuple((remainder >> shift) & 1 for shift in range(CRC - 1, -1, -1)))
    return tuple(reversed(rows))


def crc_padding(information_bits: torch.Tensor, CRC: int) -> torch.Tensor:
    """
    Add CRC bit into batched information bits, return the batched information bits after adding CRC bits.

    Args:
        information_bits (torch.Tensor): the batched information bits. (batch_size, k, 1)
        CRC (int): The number of CRC bits to be added per k bit message.

    Returns:
        Tensor of shape (batch_size, k + CRC, 1), preserving dtype and device.
        For CRC=0, returns the input tensor itself.

    """
    _validate_crc_input(information_bits, CRC)
    if CRC == 0:
        return information_bits

    k = information_bits.shape[1]
    generator = torch.tensor(_crc_generator_rows(k, CRC), dtype=torch.float32, device=information_bits.device).reshape(k, CRC)
    # Compute binary sums in float32 even for half-precision inputs or under
    # autocast, then reduce modulo two and restore the input dtype.
    with torch.autocast(device_type=information_bits.device.type, enabled=False):
        parity = torch.matmul(information_bits.squeeze(-1).to(torch.float32), generator).remainder(2)
    parity = parity.to(information_bits.dtype).unsqueeze(-1)
    information_bits_crc = torch.cat((information_bits, parity), dim=1)
    return information_bits_crc


def crc_removal(predicted_probabilities_crc: torch.Tensor, CRC: int) -> tuple[torch.Tensor, torch.Tensor]:
    """0. round the predicted probabilities to binary values (0 or 1).

        0.1 If the predicted probability is greater than 0.5, it is rounded to 1.
        0.2 If the predicted probability is less than or equal to 0.5, it is rounded to 0.
    1. Remove the CRC bits from the batched predicted probabilities.
    2. Perform CRC checking on the remaining information bits.

    Args:
        predicted_probabilities_crc (torch.Tensor): The batched predicted probabilities with CRC. Each elements is in (0, 1). shape (batch_size, k + CRC, 1)
        CRC (int): The number of CRC bits to be removed per message.

    Returns:
        Hard information bits of shape (batch_size, k, 1), preserving dtype
        and device, and a boolean CRC result of shape (batch_size, 1).
        For CRC=0, all messages pass and no bits are removed.

    """
    _validate_crc_input(predicted_probabilities_crc, CRC)
    if predicted_probabilities_crc.shape[1] < CRC:
        raise ValueError("Input length must be greater than or equal to the CRC length.")

    hard_bits = (predicted_probabilities_crc > 0.5).to(predicted_probabilities_crc.dtype)
    if CRC == 0:
        return hard_bits, torch.ones((hard_bits.shape[0], 1), dtype=torch.bool, device=hard_bits.device)

    predicted_information_bits = hard_bits[:, :-CRC, :]
    expected_parity = crc_padding(predicted_information_bits, CRC)[:, -CRC:, :]
    crc_passed = (hard_bits[:, -CRC:, :] == expected_parity).all(dim=1)
    return predicted_information_bits, crc_passed


@torch.no_grad()
def candidate_list_decoding(predicted_probabilities_crc: torch.Tensor, CRC: int, candidate_list_size: int, verbose: bool = False) -> torch.Tensor:
    """Select the most likely CRC-valid word from a bounded candidate list.

    Adapt Sionna's PolarSCLDecoder path branching, metric pruning, and final
    CRC-aided selection to independent predicted bit probabilities. At each
    position, extend every surviving path with both bit values and keep the
    most likely ``candidate_list_size`` paths. Information and CRC bits both
    contribute to the likelihood. This uses no polar transform or SC recursion.

    Select the best CRC-valid completed path, or the best path overall if none
    pass. Ties keep the hard-decision branch first (0 for a probability of 0.5).

    Args:
        predicted_probabilities_crc (torch.Tensor): Floating probabilities of
            bit 1, shaped (batch_size, k + CRC, 1). Values must be finite and
            in [0, 1]; endpoints are supported for saturated predictions.
        CRC (int): CRC width supported by ``crc_padding``: 0, 1, 2, 3, 6, 11, 16.
        candidate_list_size (int): Positive maximum number of surviving paths,
            including the hard-decision path. Need not be a power of two.
        verbose (bool): If True, print the number of messages passing CRC.

    Returns:
        Hard information bits of shape (batch_size, k, 1), preserving dtype
        and device. CRC=0 and list size 1 reduce to thresholding at 0.5 and
        removing any CRC bits. The operation is not differentiable.

    """
    batch_size, length, _ = predicted_probabilities_crc.shape
    k = length - CRC
    if CRC == 0 or candidate_list_size == 1 or batch_size == 0:
        information_bits, passed = crc_removal(predicted_probabilities_crc, CRC)
        if verbose:
            print(f"Candidate list decoding: {passed.sum().item()}/{batch_size} messages passed CRC.")
        return information_bits

    # Sionna uses softplus(-(1 - 2*b) * LLR) for each branch, where
    # LLR = log(P(0)/P(1)). Subtract the hard-decision cost, common to all
    # paths, so matching a hard bit costs zero and flipping costs abs(LLR).
    # Compute logs in at least float32, including inside an autocast context.
    metric_dtype = torch.float64 if predicted_probabilities_crc.dtype == torch.float64 else torch.float32
    probabilities = predicted_probabilities_crc.squeeze(-1).to(metric_dtype)
    tiny = torch.finfo(metric_dtype).tiny
    flip_costs = ((1 - probabilities).clamp_min(tiny).log() - probabilities.clamp_min(tiny).log()).abs()
    hard_bits = probabilities > 0.5
    paths = torch.empty((batch_size, 1, 0), dtype=torch.bool, device=probabilities.device)
    metrics = torch.zeros((batch_size, 1), dtype=metric_dtype, device=probabilities.device)

    for position in range(length):
        num_paths = metrics.shape[1]
        branch_metrics = torch.cat((metrics, metrics + flip_costs[:, position : position + 1]), dim=1)
        # Stable sorting keeps hard decisions first when branch metrics tie.
        indices = branch_metrics.argsort(dim=1, stable=True)[:, :candidate_list_size]
        metrics = branch_metrics.gather(1, indices)
        parents = indices.remainder(num_paths)
        paths = paths.gather(1, parents.unsqueeze(-1).expand(-1, -1, position))
        next_bits = hard_bits[:, position : position + 1] ^ (indices >= num_paths)
        paths = torch.cat((paths, next_bits.unsqueeze(-1)), dim=-1)

    num_paths = paths.shape[1]
    information_bits, passed = crc_removal(paths.reshape(batch_size * num_paths, length, 1), CRC)
    passed = passed.reshape(batch_size, num_paths)
    # Paths are ordered by metric. argmax returns the first valid candidate,
    # and zero (the best path overall) when every candidate fails CRC.
    selected = passed.to(torch.int64).argmax(dim=1)
    information_bits = information_bits.reshape(batch_size, num_paths, k, 1)
    batch_indices = torch.arange(batch_size, device=probabilities.device)
    decoded = information_bits[batch_indices, selected].to(predicted_probabilities_crc.dtype)
    if verbose:
        print(f"Candidate list decoding: {passed.any(dim=1).sum().item()}/{batch_size} messages passed CRC.")
    return decoded


if __name__ == "__main__":
    # pytest.main([__file__, "-v", "-s"])
    # print("Reference CRC encoding:")
    # print(crc_padding(torch.tensor([[[1], [1], [0], [1]]], dtype=torch.float32), 3))

    print("Reference CRC decoding:")
    predicted_probs = torch.tensor([[[1], [1], [0], [1], [1], [1], [1]]], dtype=torch.float32)
    information_bits, crc_passed = crc_removal(predicted_probs, 3)
    print("Information bits:", information_bits)
    print("CRC passed:", crc_passed)
