#!/usr/bin/env bash
# Downloads everything the project needs into data/.
#   - processed MIMIC-III records, vocabularies and graphs released with GAMENet
#   - ICD-9-CM v32 diagnosis and procedure descriptions (CMS, public domain)
#   - WHO ATC code names (from the fabkury/atcd scrape of the WHO index)
# Run from the repository root:  bash scripts/get_data.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
mkdir -p "$ROOT/data/gamenet" "$ROOT/data/descriptions"

git clone -q --depth 1 https://github.com/sjy1203/GAMENet.git "$TMP/GAMENet"
cp "$TMP"/GAMENet/data/{records_final,voc_final,ddi_A_final,ehr_adj_final}.pkl "$ROOT/data/gamenet/"

git clone -q --depth 1 https://github.com/ufbmi/icd-tools.git "$TMP/icd-tools"
cp "$TMP/icd-tools/icd-9-cm-v32/CMS32_DESC_LONG_SG.txt" "$ROOT/data/descriptions/"

curl -sSL -o "$ROOT/data/descriptions/who_atc_2021.csv" \
  "https://raw.githubusercontent.com/fabkury/atcd/master/WHO%20ATC-DDD%202021-12-03.csv"

pip download --no-deps -q -d "$TMP/pkgs" icd9cms==0.2.1
python -I -m zipfile -e "$TMP"/pkgs/icd9cms-0.2.1-py3-none-any.whl "$TMP/icd9cms"
python -I - "$TMP/icd9cms/icd9cms/data/CMS32_DESC_LONG_SHORT_DX.xlsx" "$ROOT/data/descriptions/cms32_dx.csv" <<'EOF'
import sys, pandas as pd
x = pd.read_excel(sys.argv[1], dtype=str)
x[["DIAGNOSIS CODE", "LONG DESCRIPTION"]].to_csv(sys.argv[2], index=False)
EOF
rm -rf "$TMP"
echo "done: $(ls "$ROOT/data/gamenet" "$ROOT/data/descriptions" | tr '\n' ' ')"
