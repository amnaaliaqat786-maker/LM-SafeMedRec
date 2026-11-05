"""Turn medical code descriptions into vectors.

Encoders:
  --encoder tfidf
      Offline lexical baseline (word + character TF-IDF -> SVD).

  --encoder hf --model NAME
      Hugging Face model with masked mean pooling.

Two modes:
  Default:
      Uses GAMENet vocabulary and creates vocabulary-aligned embeddings.
      Requires authorized GAMENet files.

  --all-codes:
      Uses public medical code descriptions only.
      Does NOT require patient records or GAMENet vocabulary files.
      Saves code identifiers alongside embeddings for later alignment.

Examples:
  python text_embed.py --encoder tfidf
  python text_embed.py --encoder hf --model emilyalsentzer/Bio_ClinicalBERT

  python text_embed.py --encoder hf --model Qwen/Qwen2.5-0.5B-Instruct --all-codes

Output:
  data/text_emb/<name>.npz

Vocabulary mode:
  diag, proc, med

All-codes mode:
  diag_codes, diag
  proc_codes, proc
  med_codes, med
"""

import argparse
import os
import numpy as np

from data import (
    load_all,
    code_texts,
    ROOT,
    _dx_table,
    _sg_table,
    _atc_table,
)

OUT_DIR = os.path.join(ROOT, "data", "text_emb")


def public_code_texts():
    """Load public medical descriptions using data.py text preparation rules."""

    dx = _dx_table()
    sg = _sg_table()
    atc = _atc_table()

    tables = {
        "diag": dx,
        "proc": sg,
        "med": atc,
    }

    texts = {}
    codes = {}

    for key, table in tables.items():
        valid = {}

        for code, description in table.items():
            if code is None or description is None:
                continue

            code = str(code).strip()
            description = str(description).strip()

            if not code or not description:
                continue

            if description.lower() in {"nan", "none"}:
                continue

            valid[code] = description

        ordered_codes = sorted(valid)

        codes[key] = ordered_codes
        prepared_texts = []

        for code in ordered_codes:
            description = valid[code]

            if key == "med":
                # Match the ATC level-3 + level-4 preparation in data.py.
                level3 = atc.get(code[:3], "")

                description = description.lower()

                if level3 and level3.lower() not in description:
                    description = f"{level3.lower()}: {description}"

            prepared_texts.append(description)

        texts[key] = prepared_texts

        print(f"{key}: {len(ordered_codes)} public descriptions")

    return texts, codes

def tfidf_embed(texts, dim=128, seed=0):
    """Create TF-IDF embeddings using word and character features."""

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from scipy.sparse import hstack

    allt = texts["diag"] + texts["proc"] + texts["med"]

    if not allt:
        raise ValueError("No medical descriptions were found.")

    w = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    ).fit(allt)

    c = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        sublinear_tf=True,
    ).fit(allt)

    combined = hstack([
        w.transform(allt),
        c.transform(allt),
    ])

    actual_dim = min(dim, combined.shape[0] - 1, combined.shape[1] - 1)

    if actual_dim < 1:
        raise ValueError("Not enough descriptions or features for TF-IDF SVD.")

    svd = TruncatedSVD(
        n_components=actual_dim,
        random_state=seed,
    ).fit(combined)

    def transform(items):
        if not items:
            return np.empty((0, actual_dim), dtype=np.float32)

        features = hstack([
            w.transform(items),
            c.transform(items),
        ])

        return svd.transform(features).astype(np.float32)

    return {
        key: transform(texts[key])
        for key in ("diag", "proc", "med")
    }


def hf_embed(texts, model_name, batch=32, prefix="", max_length=64):
    """Create masked mean-pooled Hugging Face embeddings."""

    import torch
    from transformers import AutoTokenizer, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)
    print("Loading tokenizer:", model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading model:", model_name)

    dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=dtype,
    ).to(device).eval()

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))

    tokenizer.padding_side = "right"

    output = {}

    for key, kind in (
        ("diag", "diagnosis"),
        ("proc", "procedure"),
        ("med", "medication class"),
    ):

        items = [
            f"{prefix}{kind}: {description}"
            for description in texts[key]
        ]

        if not items:
            hidden_size = getattr(
                model.config,
                "hidden_size",
                getattr(model.config, "d_model", None),
            )

            if hidden_size is None:
                raise ValueError("Could not determine model hidden size.")

            output[key] = np.empty(
                (0, hidden_size),
                dtype=np.float32,
            )

            print(key, output[key].shape)
            continue

        rows = []

        for start in range(0, len(items), batch):
            batch_items = items[start:start + batch]

            encoded = tokenizer(
                batch_items,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            with torch.inference_mode():
                hidden = model(**encoded).last_hidden_state.float()

            mask = encoded["attention_mask"].unsqueeze(-1).float()

            pooled = (
                (hidden * mask).sum(dim=1)
                / mask.sum(dim=1).clamp(min=1.0)
            )

            rows.append(pooled.cpu().numpy())

        output[key] = np.concatenate(rows).astype(np.float32)

        print(key, output[key].shape)

    return output


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate medical code description embeddings."
    )

    parser.add_argument(
        "--encoder",
        choices=["tfidf", "hf"],
        default="tfidf",
    )

    parser.add_argument(
        "--model",
        default="emilyalsentzer/Bio_ClinicalBERT",
    )

    parser.add_argument(
        "--name",
        default=None,
        help="Output file name.",
    )

    parser.add_argument(
        "--all-codes",
        action="store_true",
        help="Use public descriptions without loading GAMENet patient data.",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--dim",
        type=int,
        default=128,
        help="TF-IDF embedding dimension.",
    )

    args = parser.parse_args()

    if args.all_codes:
        print("\nMode: PUBLIC CODE DESCRIPTIONS")
        print("Patient records and GAMENet vocabulary are not loaded.\n")

        texts, codes = public_code_texts()

    else:
        print("\nMode: GAMENet vocabulary-aligned embeddings\n")

        _, vocabulary, *_ = load_all()
        texts, stats = code_texts(vocabulary)

        print("Description coverage:", stats)
        codes = None

    if args.encoder == "tfidf":
        embeddings = tfidf_embed(texts, dim=args.dim)

    else:
        embeddings = hf_embed(
            texts,
            model_name=args.model,
            batch=args.batch,
            max_length=args.max_length,
        )

    if args.name:
        name = args.name

    elif args.encoder == "tfidf":
        name = "tfidf_all_codes" if args.all_codes else "tfidf"

    else:
        model_name = args.model.split("/")[-1]
        name = f"{model_name}_all_codes" if args.all_codes else model_name

    os.makedirs(OUT_DIR, exist_ok=True)

    output_path = os.path.join(OUT_DIR, f"{name}.npz")

    if args.all_codes:
        saved_arrays = {}

        for key in ("diag", "proc", "med"):
            saved_arrays[f"{key}_codes"] = np.asarray(
                codes[key],
                dtype=str,
            )

            saved_arrays[key] = embeddings[key]

        np.savez(output_path, **saved_arrays)

    else:
        np.savez(output_path, **embeddings)

    print("\nEmbeddings saved successfully:")
    print(output_path)
