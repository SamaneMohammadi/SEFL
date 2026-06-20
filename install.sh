#!/usr/bin/env bash
set -e
# gmpy2 needs GMP headers (Debian/Ubuntu: libgmp-dev; macOS: brew install gmp)
pip install -r requirements.txt
echo "Done."
