#!/usr/bin/env bash

# Prepare the local data directories for LM-SafeMedRec.
#
# IMPORTANT:
# This script does NOT download or redistribute MIMIC-III-derived
# patient records or processed GAMENet pickle files.
#
# Researchers must obtain their own authorized access to MIMIC-III
# and prepare the required files according to the applicable
# data-use agreement and GAMENet preprocessing instructions.
#
# Run from the repository root:
#   bash scripts/get_data.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

mkdir -p \
    "$ROOT/data/gamenet" \
    "$ROOT/data/descriptions" \
    "$ROOT/data/text_emb"

echo "LM-SafeMedRec data directories are ready."

echo ""
echo "IMPORTANT:"
echo "1. Obtain authorized MIMIC-III access through PhysioNet."
echo "2. Follow the applicable data-use agreement."
echo "3. Prepare the required GAMENet-compatible files yourself."
echo "4. Place the authorized files in data/gamenet/."
echo "5. Add the required code-description files to data/descriptions/."

echo ""
echo "No patient-level data has been downloaded by this script."
