"""
SEFL server.

The server is honest-but-curious: it aggregates client updates but must never
see a plaintext gradient. Because Paillier is additively homomorphic, the server
can sum the encrypted client gradients without any key:

    E_g = E_1 (+) E_2 (+) ... (+) E_K

and broadcast the encrypted result. Decryption happens back on the clients.
That is the whole server - it holds no private key and does no decryption.
"""

from utils.crypto import he


class Server:
    def aggregate(self, sparse_updates, total_length, public_key):
        """Homomorphic sum of the clients' sparse encrypted gradients.

        Clients send only their non-zero ciphertexts plus the positions. The
        server uses the public key to encrypt the zeros for the pruned positions
        so every update aligns, then adds them. It holds no private key and does
        no decryption.
        """
        return he.aggregate_sparse(sparse_updates, total_length, public_key)
