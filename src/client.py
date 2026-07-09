from collections import OrderedDict

import torch

from dataset import get_loaders
from utils.pruning import prune_gradients
from utils.crypto import he
import config


def get_weights(model):
    return [p.detach().clone() for p in model.parameters()]


def set_weights(model, weights):
    with torch.no_grad():
        for p, w in zip(model.parameters(), weights):
            p.copy_(w)


class Client:
    def __init__(self, cid, model, data_dir="client_data", device="cpu"):
        self.cid = cid
        self.model = model
        self.device = device
        self.train_loader, self.val_loader, self.n_train = get_loaders(
            cid, data_dir=data_dir, batch_size=config.BATCH_SIZE
        )
        self.criterion = torch.nn.NLLLoss()

    def compute_gradient(self, global_weights):
        """One local pass over the client's data -> mean gradient per layer."""
        set_weights(self.model, global_weights)
        self.model.train()

        accum, n_batches = None, 0
        for _ in range(config.LOCAL_EPOCHS):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.model.zero_grad()
                loss = self.criterion(self.model(x), y)
                loss.backward()
                g = [p.grad.detach().clone() for p in self.model.parameters()]
                accum = g if accum is None else [a + b for a, b in zip(accum, g)]
                n_batches += 1
        return [a / max(n_batches, 1) for a in accum]

    def prune_and_encrypt(self, grads, public_key, pruning_percent):
        """Prune the gradient, then encrypt only the surviving non-zeros.

        Returns (sparse_update, shapes, total_length) where sparse_update is
        (indices, enc_values). The pruned (zero) positions are not encrypted or
        sent; the server fills them in with its public key.
        """
        pruned = prune_gradients(grads, pruning_percent)
        vec, shapes = he.flatten(pruned)
        idx, enc = he.encrypt_sparse(public_key, vec)
        return (idx, enc), shapes, len(vec)

    def decrypt(self, private_key, cipher_vector, shapes):
        """Decrypt the server's aggregated ciphertext back into gradients."""
        vec = he.decrypt_vector(private_key, cipher_vector)
        return he.unflatten(vec, shapes)
