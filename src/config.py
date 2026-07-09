# --- Federated setup ---------------------------------------------------------
NUM_CLIENTS = 91            # one client per CREMA-D speaker
CLIENTS_PER_ROUND = 20     # K selected clients each round
NUM_ROUNDS = 200           # total training rounds (epochs)
BATCH_SIZE = 20            # local minibatch size B
LEARNING_RATE = 0.1        # eta (FedSGD)
LOCAL_EPOCHS = 1
VAL_SPLIT = 0.2            # 80/20 train/val per client

# --- Model (MLP) -------------------------------------------------------------
INPUT_DIM = 988            # OpenSMILE emobase feature length
HIDDEN_DIMS = [256, 128]   # two dense layers
NUM_CLASSES = 4            # neutral, sad, happy, angry
DROPOUT = 0.2
EMOTIONS = ["ANG", "HAP", "NEU", "SAD"]   # sorted -> stable label ids

# --- SEFL: encryption + pruning ----------------------------------------------
# Paillier key size in bits. Paper sweeps these; 128 is the default for the
# accuracy experiments, 1024 for the efficiency headline numbers.
KEY_SIZES = [128, 256, 512, 1024]
DEFAULT_KEY_SIZE = 128

# Gradient pruning percentage (per layer). Paper sweeps these.
PRUNING_PERCENTAGES = [20, 40, 60, 80]
DEFAULT_PRUNING = 60

# --- Reproducibility ---------------------------------------------------------
SEED = 8                   # paper uses seed_everything(8)
