from utils.crypto import he


class Server:
    def aggregate(self, sparse_updates, total_length, public_key):
        return he.aggregate_sparse(sparse_updates, total_length, public_key)
