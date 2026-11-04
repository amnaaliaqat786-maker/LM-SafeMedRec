#!/usr/bin/env bash

# Download and prepare medical-code descriptions.
# Does NOT download patient-level data.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/data/descriptions"
TMP="$(mktemp -d)"

trap 'rm -rf "$TMP"' EXIT

mkdir -p "$DEST"

echo "Downloading CMS ICD-9-CM Version 32 archive..."

curl -fL \
  "https://www.cms.gov/medicare/coding/icd9providerdiagnosticcodes/downloads/icd-9-cm-v32-master-descriptions.zip" \
  -o "$TMP/cms32.zip"

echo "Extracting diagnosis descriptions..."
unzip -p "$TMP/cms32.zip" CMS32_DESC_LONG_DX.txt \
  > "$TMP/CMS32_DESC_LONG_DX.txt"

echo "Extracting procedure descriptions..."
unzip -p "$TMP/cms32.zip" CMS32_DESC_LONG_SG.txt \
  > "$TMP/CMS32_DESC_LONG_SG.txt"

echo "Creating diagnosis CSV..."

if command -v py >/dev/null 2>&1; then
    PYTHON=(py -3.14)
elif command -v python >/dev/null 2>&1; then
    PYTHON=(python)
else
    echo "ERROR: Python was not found."
    exit 1
fi

# Create the conversion program.
cat > "$TMP/convert_dx.py" <<'PY'
import csv
import sys

source, destination = sys.argv[1], sys.argv[2]

with open(source, encoding="latin-1") as infile, \
     open(destination, "w", newline="", encoding="utf-8") as outfile:

    writer = csv.writer(outfile)
    writer.writerow(["code", "description"])

    for line in infile:
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            writer.writerow(parts)
PY

"${PYTHON[@]}" "$TMP/convert_dx.py" \
  "$TMP/CMS32_DESC_LONG_DX.txt" "$TMP/cms32_dx.csv"

echo "Downloading WHO ATC 2021 description snapshot..."

curl -fL \
  "https://raw.githubusercontent.com/fabkury/atcd/c5c6833428619b58380fe8a06dcb00e6dc511087/WHO%20ATC-DDD%202021-12-03.csv" \
  -o "$TMP/who_atc_2021.csv"

echo "Validating downloaded files..."

"${PYTHON[@]}" - "$TMP" <<'PY'
import csv
import os
import sys

folder = sys.argv[1]

dx = os.path.join(folder, "cms32_dx.csv")
sg = os.path.join(folder, "CMS32_DESC_LONG_SG.txt")
atc = os.path.join(folder, "who_atc_2021.csv")

with open(dx, encoding="utf-8", newline="") as f:
    rows = list(csv.reader(f))
    assert rows[0] == ["code", "description"]
    assert len(rows) > 1000

with open(sg, encoding="latin-1") as f:
    assert sum(1 for line in f if line.strip()) > 1000

with open(atc, encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    assert {"atc_code", "atc_name"}.issubset(reader.fieldnames or set())
    assert sum(1 for _ in reader) > 1000

print("Diagnosis entries:", len(rows) - 1)
print("Procedure descriptions: validated")
print("ATC descriptions: validated")
PY

# Move files into place only after validation succeeds.
mv "$TMP/cms32_dx.csv" "$DEST/cms32_dx.csv"
mv "$TMP/CMS32_DESC_LONG_SG.txt" "$DEST/CMS32_DESC_LONG_SG.txt"
mv "$TMP/who_atc_2021.csv" "$DEST/who_atc_2021.csv"

echo ""
echo "All three description files are ready."
echo "Saved in: $DEST"
echo ""
echo "No patient-level data was downloaded."
echo "Do not redistribute the WHO ATC snapshot without checking its terms."
