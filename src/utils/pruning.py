import torch


def prune_gradients(grads, pruning_percent):
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
    total = sum(g.numel() for g in grads)
    zeros = sum((g == 0).sum().item() for g in grads)
    return zeros / total if total else 0.0
