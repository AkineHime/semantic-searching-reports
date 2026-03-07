"""Download the 20 Newsgroups dataset from UCI and extract it.

The UCI version is the officially hosted dataset:
https://archive.ics.uci.edu/dataset/113/twenty+newsgroups

This script downloads the archive, extracts it, and places the files under
`data/20_newsgroups/` in the expected structure.

Usage:
    python download_20newsgroups.py
"""

from pathlib import Path
import tarfile
import urllib.request


UCI_TARBALL_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/20newsgroups/20news-18828.tar.gz"


def main(out_dir: Path = Path("data") / "20_newsgroups"):
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tarball_path = out_dir.parent / "20news-18828.tar.gz"
    if not tarball_path.exists():
        print(f"Downloading 20 Newsgroups dataset from UCI to {tarball_path}...")
        urllib.request.urlretrieve(UCI_TARBALL_URL, tarball_path)
    else:
        print(f"Using existing tarball: {tarball_path}")

    print("Extracting dataset (this may take a minute)...")
    with tarfile.open(tarball_path, "r:gz") as tar:
        # The archive contains a top-level directory named '20news-18828'
        # Extract only files; ignore intermediate directories.
        def is_within_directory(directory, target):
            abs_directory = Path(directory).resolve()
            abs_target = Path(target).resolve()
            return abs_directory in abs_target.parents or abs_directory == abs_target

        for member in tar.getmembers():
            if not member.isfile():
                continue
            member_path = Path(member.name)
            # Keep only the dataset content (skip any unexpected paths)
            if member_path.parts[0] != "20news-18828":
                continue

            dest_path = out_dir / Path(*member_path.parts[1:])
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as src, open(dest_path, "wb") as dst:
                dst.write(src.read())

    print("Dataset extraction complete.")
    print("Data is available under:", out_dir)


if __name__ == "__main__":
    main()
