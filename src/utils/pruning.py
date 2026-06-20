"""
Gradient pruning (Algorithm 2 in the paper).

For each layer we keep only the high-magnitude gradients: we find the threshold
at the given pruning percentile within that layer and zero everything at or
below it. Pruning per layer (rather than globally) means each layer keeps the
same fraction of its gradients, which is what the paper's algorithm specifies.

This shrinks the number of non-zero values that get encrypted, which is where
SEFL's communication and computation savings come from.
"""

import torch


def prune_gradients(grads, pruning_percent):
    """Zero the smallest-magnitude `pruning_percent`% of gradients in each layer.

    grads            -- list of gradient tensors, one per parameter/layer
    pruning_percent  -- 0..100; 60 means drop the smallest 60% per layer
    Returns a new list of pruned gradient tensors.
    """
    if pruning_percent <= 0:
        return [g.clone() for g in grads]

    pruned = []
    for g in grads:
        n = g.numel()
        k = int(n * pruning_percent / 100)        # how many to drop
        if k < 1 or n <= 1:
            pruned.append(g.clone())
            continue
        flat_abs = g.abs().flatten()
        # k-th smallest magnitude is the threshold; <= threshold gets zeroed
        threshold = torch.kthvalue(flat_abs, k).values
        mask = g.abs() > threshold
        pruned.append(g * mask)
    return pruned


def sparsity(grads):
    """Fraction of zero entries across all gradients (sanity/reporting)."""
    total = sum(g.numel() for g in grads)
    zeros = sum((g == 0).sum().item() for g in grads)
    return zeros / total if total else 0.0
