# SEFL

Implementation of **Secure and Efficient Federated Learning by Combining
Homomorphic Encryption and Gradient Pruning in Speech Emotion Recognition**
(ISPEC 2023).

📄 Paper: [Springer LNCS 14341](https://doi.org/10.1007/978-981-99-7032-2_1)

SEFL is a privacy-preserving federated learning method for speech emotion
recognition (SER) that combines Paillier homomorphic encryption (PHE) with a
novel magnitude-based gradient pruning technique. Each client trains a small MLP
locally on OpenSMILE emobase features, prunes the low-magnitude gradients, and
encrypts the pruned gradient with PHE; the honest-but-curious server aggregates
the encrypted gradients through homomorphic addition without ever seeing
plaintext, and the clients decrypt the aggregated result to update their model.
Pruning shrinks the encrypted messages, so the added privacy comes with much
lower communication and computation cost. This is the first work to implement PHE
in an FL setup for SER: with a 1024-bit key it cuts computation time by up to 25%
and communication traffic by up to 70%, while keeping accuracy around the ~70%
baseline acceptable for SER on the four basic emotions.

## Methods

FedSGD with **Paillier homomorphic-encryption secure aggregation** and
**magnitude gradient pruning**. The model is an MLP with two dense layers
`[256, 128]`, ReLU and 0.2 dropout, trained on CREMA-D / OpenSMILE emobase
(988-dim features) over four emotions (neutral, sad, happy, angry). Key size
(128 / 256 / 512 / 1024) and pruning percentage (20 / 40 / 60 / 80%) are both
configurable; the server only ever performs homomorphic addition and never
decrypts.

## Setup

```bash
pip install -r requirements.txt
```

`gmpy2` needs the GMP library first: `sudo apt-get install libgmp-dev`
(Debian/Ubuntu) or `brew install gmp` (macOS).

## Usage

```bash
cd src

# 1) prepare data: CREMA-D -> OpenSMILE emobase features -> one client per speaker
python prepare_data.py --data_path ./CREMA-D/AudioWAV --out_dir ./client_data

# 2) full SEFL: 1024-bit Paillier key, 60% gradient pruning
python main.py --key_size 1024 --pruning 60 --data_dir ./client_data

# plaintext FedSGD baseline (no encryption)
python main.py --no-encrypt --data_dir ./client_data
```

## Citation

```bibtex
@inproceedings{mohammadi2023sefl,
  title={Secure and Efficient Federated Learning by Combining Homomorphic Encryption and Gradient Pruning in Speech Emotion Recognition},
  author={Mohammadi, Samaneh and Sinaei, Sima and Balador, Ali and Flammini, Francesco},
  booktitle={Information Security Practice and Experience (ISPEC 2023)},
  series={Lecture Notes in Computer Science},
  volume={14341},
  pages={1--16},
  year={2023},
  publisher={Springer},
  doi={10.1007/978-981-99-7032-2_1}
}
```

## License

MIT. The Paillier implementation under `src/utils/crypto/` derives from the
FATE project and retains its Apache-2.0 license headers.
