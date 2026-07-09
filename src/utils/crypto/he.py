import numpy as np
import torch
from joblib import Parallel, delayed
import multiprocessing

from .paillier import PaillierKeypair  # noqa: F401  (re-export for callers)

_N_JOBS = max(1, multiprocessing.cpu_count())


def flatten(grads):
    """List of tensors -> (1D numpy vector, list of shapes)."""
    shapes = [tuple(g.shape) for g in grads]
    vec = np.concatenate([g.detach().cpu().numpy().ravel() for g in grads])
    return vec, shapes


def unflatten(vec, shapes):
    """Inverse of flatten: 1D vector + shapes -> list of tensors."""
    out, i = [], 0
    for shape in shapes:
        n = int(np.prod(shape)) if shape else 1
        out.append(torch.tensor(vec[i:i + n].reshape(shape), dtype=torch.float32))
        i += n
    return out


def encrypt_vector(public_key, vec, parallel=True):
    """Encrypt each element of a 1D numpy vector under `public_key`."""
    vec = np.asarray(vec).ravel()
    if parallel and len(vec) > 1000:
        enc = Parallel(n_jobs=_N_JOBS)(delayed(public_key.encrypt)(float(v)) for v in vec)
    else:
        enc = [public_key.encrypt(float(v)) for v in vec]
    return enc


def aggregate(encrypted_vectors):
    """Element-wise homomorphic sum of several encrypted vectors.

    The server runs this on ciphertext only. All inputs must be encrypted under
    the same public key (the round's shared key pair).
    """
    agg = list(encrypted_vectors[0])
    for ev in encrypted_vectors[1:]:
        agg = [a + b for a, b in zip(agg, ev)]
    return agg


def decrypt_vector(private_key, enc_vec, parallel=True):
    """Decrypt an encrypted vector back to a 1D numpy float array."""
    if parallel and len(enc_vec) > 1000:
        dec = Parallel(n_jobs=_N_JOBS)(delayed(private_key.decrypt)(c) for c in enc_vec)
    else:
        dec = [private_key.decrypt(c) for c in enc_vec]
    return np.array(dec, dtype=np.float64)


# --- Sparse path -------------------------------------------------------------
# After pruning, most of the gradient is zero. The client encrypts only the
# surviving non-zero values and sends them together with their positions (a
# plaintext index list). The server holds the public key, so it can encrypt the
# zeros itself to fill the pruned positions and keep every client's vector
# aligned for the homomorphic sum. This is what makes the encrypted payload
# shrink with the pruning percentage.

def encrypt_sparse(public_key, vec, parallel=True):
    """Encrypt only the non-zero entries of `vec`.

    Returns (indices, enc_values): the positions that survived pruning and their
    ciphertexts. The zeros are not encrypted or transmitted.
    """
    vec = np.asarray(vec).ravel()
    idx = np.nonzero(vec)[0]
    enc = encrypt_vector(public_key, vec[idx], parallel=parallel)
    return idx, enc


def aggregate_sparse(sparse_updates, total_length, public_key):
    """Homomorphically sum sparse client updates into a full-length cipher vector.

    sparse_updates -- list of (indices, enc_values) from each client
    total_length   -- length of the full flattened gradient
    public_key     -- used to encrypt zeros for positions no client sent

    The server only ever adds ciphertexts; it never decrypts.
    """
    agg = [None] * total_length
    for idx, enc_values in sparse_updates:
        for i, c in zip(idx, enc_values):
            agg[i] = c if agg[i] is None else agg[i] + c

    # positions nobody contributed to are exactly zero; the server encrypts a
    # single zero with the public key and reuses it to fill them
    zero_cipher = public_key.encrypt(0.0)
    return [c if c is not None else zero_cipher for c in agg]
