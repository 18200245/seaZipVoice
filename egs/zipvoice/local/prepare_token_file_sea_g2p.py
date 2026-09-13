#!/usr/bin/env python3
# Copyright    2024-2026  Xiaomi Corp. & Community
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Generate tokens file mapping IPA tokens to IDs for SEA-G2P.
Padding token '_' is guaranteed to be at index 0.
"""

import argparse
import logging
from collections import Counter
from pathlib import Path
from typing import List

from lhotse import load_manifest_lazy

try:
    from sea_g2p import SEAPipeline
except ImportError:
    SEAPipeline = None


def get_args():
    parser = argparse.ArgumentParser(
        description="Generate token-to-ID mapping file using SEA-G2P."
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        nargs="+",
        required=True,
        help="Path to one or more manifest files (e.g., cuts_train.jsonl.gz)",
    )

    parser.add_argument(
        "--tokens",
        type=Path,
        required=True,
        help="Path to output tokens file (e.g., data/tokens_sea_g2p.txt)",
    )

    parser.add_argument(
        "--lang",
        type=str,
        default="vi",
        choices=["vi", "th", "id"],
        help="Language code for SEA-G2P (vi, th, id).",
    )

    parser.add_argument(
        "--base-tokens",
        type=Path,
        default=None,
        help="Optional path to existing base tokens file to extend (e.g., pretrained tokens.txt).",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for parallel phonemization.",
    )

    return parser.parse_args()


def prepare_tokens(
    manifest_files: List[Path],
    token_file: Path,
    lang: str,
    base_tokens_file: Optional[Path] = None,
    batch_size: int = 256,
):
    if SEAPipeline is None:
        raise RuntimeError(
            "sea-g2p is not installed. Please install it with:\n"
            "pip install sea-g2p"
        )

    pipeline = SEAPipeline(lang=lang)
    counter = Counter()

    for manifest_path in manifest_files:
        logging.info(f"Loading manifest from {manifest_path}")
        manifest = load_manifest_lazy(manifest_path)
        batch = []

        for cut in manifest:
            if not cut.supervisions:
                continue
            text = cut.supervisions[0].text
            if text:
                batch.append(text)

            if len(batch) >= batch_size:
                phoneme_strs = pipeline.run(batch)
                if isinstance(phoneme_strs, str):
                    phoneme_strs = [phoneme_strs]
                for p_str in phoneme_strs:
                    counter.update(p_str)
                batch.clear()

        if batch:
            phoneme_strs = pipeline.run(batch)
            if isinstance(phoneme_strs, str):
                phoneme_strs = [phoneme_strs]
            for p_str in phoneme_strs:
                counter.update(p_str)
            batch.clear()

    unique_tokens = set(counter.keys())

    # Ensure padding token '_' is not in unique_tokens
    if "_" in unique_tokens:
        unique_tokens.remove("_")

    # Sort new tokens by frequency descending
    sorted_tokens = sorted(unique_tokens, key=lambda t: counter[t], reverse=True)

    if base_tokens_file is not None and Path(base_tokens_file).is_file():
        logging.info(f"Extending base tokens from {base_tokens_file}")
        base_tokens = []
        with open(base_tokens_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\r\n")
                if line:
                    token = line.split("\t")[0]
                    base_tokens.append(token)
        base_set = set(base_tokens)
        new_tokens = [t for t in sorted_tokens if t not in base_set]
        all_tokens = base_tokens + new_tokens
        logging.info(
            f"Base vocab size: {len(base_tokens)}, added {len(new_tokens)} new SEA-G2P tokens -> Total: {len(all_tokens)}"
        )
    else:
        all_tokens = ["_"] + sorted_tokens

    token_file.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Writing {len(all_tokens)} tokens to {token_file}")

    with open(token_file, "w", encoding="utf-8") as f:
        for index, token in enumerate(all_tokens):
            f.write(f"{token}\t{index}\n")

    logging.info(f"Successfully generated {token_file} with {len(all_tokens)} tokens.")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)

    args = get_args()
    prepare_tokens(
        args.manifest,
        args.tokens,
        lang=args.lang,
        base_tokens_file=args.base_tokens,
        batch_size=args.batch_size,
    )
