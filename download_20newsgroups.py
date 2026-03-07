"""Download the 20 Newsgroups dataset and store it in a filesystem layout.

This script is provided so the repository can remain small (no dataset files checked in),
while still making it easy for users to generate the expected `data/20_newsgroups/` tree.

Usage:
    python download_20newsgroups.py

It will create `data/20_newsgroups/<category>/` and write one text file per post.
"""

from pathlib import Path
from sklearn.datasets import fetch_20newsgroups
import hashlib


def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def main(out_dir: Path = Path("data") / "20_newsgroups"):
    print("Downloading 20 Newsgroups dataset (this may take a few minutes)...")
    data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))

    print(f"Creating output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, (label, text) in enumerate(zip(data.target, data.data)):
        category = data.target_names[label].replace(".", "_")
        category_dir = out_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)

        fname = f"{idx:06d}-{_hash_text(text)[:10]}.txt"
        path = category_dir / fname
        with open(path, "w", encoding="utf-8", errors="ignore") as f:
            f.write(text)

    print("All done. Dataset saved to:", out_dir)


if __name__ == "__main__":
    main()
