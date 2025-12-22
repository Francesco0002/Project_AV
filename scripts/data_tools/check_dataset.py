import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.paths import soccernet_root

def main():
    root = soccernet_root()
    base = root / "tracking-2023"
    print("[INFO] base:", base)

    for split in ["train", "challenge", "test"]:
        z = base / f"{split}.zip"
        print(f"{split}: zip={'OK' if z.exists() else 'MISSING'} -> {z}")

    # cerca una sequenza se già estratta
    train_dir = base / "train"
    if train_dir.exists():
        seqs = sorted([p for p in train_dir.iterdir() if p.is_dir()])
        print("[INFO] train sequences found:", len(seqs))
        if seqs:
            print("[INFO] example:", seqs[0])
            # mostra sotto-cartelle
            for child in sorted(seqs[0].iterdir()):
                print(" -", child.name)
    else:
        print("[INFO] train not extracted yet:", train_dir)

if __name__ == "__main__":
    main()
