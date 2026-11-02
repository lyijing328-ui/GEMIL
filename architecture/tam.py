"""Parameter-free tiered attention operators; legacy masking is the default.

The manuscript's range-shift formula and the archived implementation's random
masking are deliberately named separately. No tensors are registered, so adding
these settings does not change checkpoint parameter keys.
"""

import math

import torch


def _triple(values, name, upper=None, positive_sum=False):
    if values is None or len(values) != 3:
        raise ValueError("{} must contain three values (high, medium, low)".format(name))
    values = [float(value) for value in values]
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("{} must contain finite nonnegative values".format(name))
    if upper is not None and any(value > upper for value in values):
        raise ValueError("{} values must be <= {}".format(name, upper))
    if positive_sum and sum(values) <= 0:
        raise ValueError("{} must have a positive sum".format(name))
    return values


def configure_tam(model, conf):
    model.tam_mode = getattr(conf, "tam_mode", "mask")
    if model.tam_mode not in {"mask", "range_shift"}:
        raise ValueError("tam_mode must be mask or range_shift")
    model.stratified_group_ratios = _triple(
        getattr(conf, "stratified_group_ratios", [0.3, 0.4, 0.3]),
        "stratified_group_ratios", positive_sum=True)
    model.stratified_mask_rates = _triple(
        getattr(conf, "stratified_mask_rates", [0.1, 0.1, 0.1]),
        "stratified_mask_rates", upper=1.0)
    rho = getattr(conf, "tam_rho", None)
    if model.tam_mode == "range_shift" and rho is None:
        raise ValueError("range_shift requires explicit tam_rho; no historical rho is established")
    if model.tam_mode == "mask" and rho is not None:
        raise ValueError("tam_rho belongs to range_shift; use stratified_mask_rates for mask")
    model.tam_rho = None if rho is None else _triple(rho, "tam_rho")
    model.tam_apply_in_eval = bool(getattr(conf, "tam_apply_in_eval", False))
    if model.tam_mode == "mask" and model.tam_apply_in_eval:
        raise ValueError("random masking at evaluation is unsupported; keep tam_apply_in_eval false")


def tier_indices(scores, ratios):
    """Partition all N instances by rank; high/mid use floor, low gets remainder."""
    ratios = _triple(ratios, "stratified_group_ratios", positive_sum=True)
    ratios = [value / sum(ratios) for value in ratios]
    ordered = torch.argsort(scores, dim=-1, descending=True)
    n = scores.shape[-1]
    n_high, n_mid = int(n * ratios[0]), int(n * ratios[1])
    return ordered[:, :n_high], ordered[:, n_high:n_high + n_mid], ordered[:, n_high + n_mid:]


def modulate_attention(scores, ratios, mode="mask", mask_rates=None, rho=None):
    """Return logits before softmax. This function does not choose train/eval timing."""
    if scores.dim() != 2 or scores.shape[0] != 1:
        raise ValueError("TAM expects a single expert score vector shaped [1, N]")
    if scores.shape[1] <= 1:
        return scores
    groups = tier_indices(scores, ratios)
    if mode == "range_shift":
        rho = _triple(rho, "tam_rho")
        response_range = scores.max(dim=-1, keepdim=True)[0] - scores.min(dim=-1, keepdim=True)[0]
        coefficients = torch.zeros_like(scores)
        for indices, coefficient in zip(groups, rho):
            coefficients.scatter_(-1, indices, coefficient)
        return scores - coefficients * response_range
    if mode != "mask":
        raise ValueError("Unknown tam_mode: {}".format(mode))
    rates = _triple(mask_rates, "stratified_mask_rates", upper=1.0)
    parts = []
    for indices, rate in zip(groups, rates):
        n_drop = int(indices.shape[1] * rate)
        if n_drop:
            selected = torch.randperm(indices.shape[1], device=scores.device)[:n_drop]
            parts.append(indices[:, selected])
    if not parts:
        return scores
    mask = torch.ones_like(scores)
    mask.scatter_(-1, torch.cat(parts, dim=-1), 0)
    return scores.masked_fill(mask == 0, -1e9)
