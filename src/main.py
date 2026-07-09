import os
import time
import argparse

import numpy as np
import torch
from pytorch_lightning import seed_everything

import config
from model import build_model
from client import Client, get_weights, set_weights
from server import Server
from utils.crypto.he import (PaillierKeypair, flatten, unflatten,
                       encrypt_sparse, aggregate_sparse, decrypt_vector)
from utils.pruning import prune_gradients
from dataset import FeatureDataset
from utils.metrics import evaluate
from torch.utils.data import DataLoader


def build_global_val_loader(data_dir, num_clients):
    """Concatenate every client's val split into one loader for global accuracy."""
    xs, ys = [], []
    for cid in range(num_clients):
        xp = os.path.join(data_dir, f"client_{cid}_x_val.npy")
        yp = os.path.join(data_dir, f"client_{cid}_y_val.npy")
        if os.path.exists(xp):
            xs.append(np.load(xp))
            ys.append(np.load(yp))
    x, y = np.concatenate(xs), np.concatenate(ys)
    return DataLoader(FeatureDataset(x, y), batch_size=256, shuffle=False)


def global_step(global_model, summed_grads, num_clients_in_round, lr):
    """w <- w - eta * (1/K) * sum_k grad_k."""
    weights = get_weights(global_model)
    new_weights = [w - lr * (g / num_clients_in_round) for w, g in zip(weights, summed_grads)]
    set_weights(global_model, new_weights)


def run_round(clients, server, global_model, key_size, pruning, encrypt, lr):
    """Run one FedSGD/SEFL round, return (timing_dict)."""
    global_weights = get_weights(global_model)

    # 1. key generation centre: fresh key pair for this round, shared by clients
    pk, sk = PaillierKeypair.generate_keypair(n_length=key_size) if encrypt else (None, None)

    # 2. each client computes + prunes its gradient
    pruned_grads, shapes = [], None
    for c in clients:
        grads = c.compute_gradient(global_weights)
        pg = prune_gradients(grads, pruning)
        vec, shapes = flatten(pg)
        pruned_grads.append(vec)

    total_length = len(pruned_grads[0])
    t_enc = t_agg = t_dec = 0.0
    n_encrypted = total_length  # default (plaintext path encrypts nothing)

    if encrypt:
        # 3. each client encrypts ONLY its non-zero (surviving) gradients
        t0 = time.time()
        sparse_updates = [encrypt_sparse(pk, v) for v in pruned_grads]
        t_enc = time.time() - t0
        n_encrypted = int(np.mean([len(idx) for idx, _ in sparse_updates]))

        # 4. server homomorphically sums them, encrypting zeros for pruned spots
        t0 = time.time()
        agg_cipher = aggregate_sparse(sparse_updates, total_length, pk)
        t_agg = time.time() - t0

        # 5. decrypt the aggregate (client side, shared sk)
        t0 = time.time()
        summed_vec = decrypt_vector(sk, agg_cipher)
        t_dec = time.time() - t0
    else:
        # plaintext aggregation (identical result, no crypto cost)
        summed_vec = np.sum(pruned_grads, axis=0)

    summed_grads = unflatten(summed_vec, shapes)
    global_step(global_model, summed_grads, len(clients), lr)
    return {"enc_s": t_enc, "agg_s": t_agg, "dec_s": t_dec,
            "encrypted_per_client": n_encrypted, "total_params": total_length}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key_size", type=int, default=config.DEFAULT_KEY_SIZE)
    parser.add_argument("--pruning", type=int, default=config.DEFAULT_PRUNING)
    parser.add_argument("--rounds", type=int, default=config.NUM_ROUNDS)
    parser.add_argument("--clients_per_round", type=int, default=config.CLIENTS_PER_ROUND)
    parser.add_argument("--data_dir", default="client_data")
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--no-encrypt", dest="encrypt", action="store_false",
                        help="aggregate in plaintext (fast); accuracy is identical")
    parser.set_defaults(encrypt=True)
    args = parser.parse_args()

    seed_everything(config.SEED, workers=True)
    device = "cpu"

    global_model = build_model().to(device)
    val_loader = build_global_val_loader(args.data_dir, config.NUM_CLIENTS)

    # one Client object per speaker (they share the global model for weight I/O)
    all_clients = [Client(cid, global_model, data_dir=args.data_dir, device=device)
                   for cid in range(config.NUM_CLIENTS)]

    rng = np.random.default_rng(config.SEED)
    server = Server()

    print(f"SEFL | key_size={args.key_size} pruning={args.pruning}% "
          f"encrypt={args.encrypt} clients/round={args.clients_per_round}")

    for rnd in range(1, args.rounds + 1):
        selected = [all_clients[i] for i in rng.choice(
            config.NUM_CLIENTS, args.clients_per_round, replace=False)]
        timing = run_round(selected, server, global_model,
                           args.key_size, args.pruning, args.encrypt, config.LEARNING_RATE)

        if rnd % args.eval_every == 0 or rnd == args.rounds:
            m = evaluate(global_model, val_loader, device)
            extra = ""
            if args.encrypt:
                extra = (f" | enc {timing['enc_s']:.1f}s agg {timing['agg_s']:.1f}s "
                         f"dec {timing['dec_s']:.1f}s | encrypted "
                         f"{timing['encrypted_per_client']}/{timing['total_params']} vals/client")
            print(f"round {rnd:3d} | acc {m['acc']:.4f} f1 {m['f1']:.4f} "
                  f"prec {m['precision']:.4f} loss {m['loss']:.4f}{extra}")


if __name__ == "__main__":
    main()
