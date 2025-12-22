import argparse
import shutil
import sys
from pathlib import Path
import zipfile


def find_repo_root(start: Path) -> Path:
    """Risale le cartelle finché trova la root della repo (cartella che contiene 'src')."""
    for p in [start] + list(start.parents):
        if (p / "src").is_dir():
            return p
    return start.parent


REPO_ROOT = find_repo_root(Path(__file__).resolve())
sys.path.insert(0, str(REPO_ROOT))  # così 'from src...' funziona sempre


def get_soccernet_root(cli_root: str | None) -> Path:
    """Se --root non è fornito, prova a leggere configs/local_paths.json via src.paths."""
    if cli_root:
        return Path(cli_root)

    try:
        from src.paths import soccernet_root  # type: ignore
        return soccernet_root()
    except Exception:
        # fallback: data/soccernet nella repo
        return REPO_ROOT / "data" / "soccernet"


def zip_contains_topdir(zip_path: Path, split: str) -> bool:
    """
    True se dentro lo zip i file sono sotto una cartella di primo livello 'split/'
    es: 'train/SNMOT-060/img1/000001.jpg'
    """
    with zipfile.ZipFile(zip_path, "r") as z:
        names = [n for n in z.namelist() if n and not n.endswith("/")]
        if not names:
            return False
        sample = names[: min(200, len(names))]
        return all(Path(n).parts and Path(n).parts[0] == split for n in sample)


def is_nonempty_dir(p: Path) -> bool:
    return p.exists() and p.is_dir() and any(p.iterdir())


def extract_one(zip_path: Path, base: Path, split: str, force: bool) -> None:
    if not zip_path.exists():
        print(f"[SKIP] {split}: zip non trovato -> {zip_path}")
        return

    expected_dir = base / split

    if is_nonempty_dir(expected_dir) and not force:
        print(f"[SKIP] {split}: già estratto -> {expected_dir}")
        print("       (usa --force per ri-estrarre)")
        return

    if force and expected_dir.exists():
        print(f"[INFO] Rimuovo cartella esistente (force): {expected_dir}")
        shutil.rmtree(expected_dir, ignore_errors=True)

    # Evita doppia annidazione:
    # - se lo zip contiene già 'train/...', estrai in base
    # - altrimenti estrai in base/train
    has_topdir = zip_contains_topdir(zip_path, split)
    dest_root = base if has_topdir else (base / split)
    dest_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Estraggo {zip_path.name}")
    print(f"       dest_root = {dest_root}")
    shutil.unpack_archive(str(zip_path), str(dest_root))

    # riepilogo
    if expected_dir.exists():
        seqs = sorted([p for p in expected_dir.iterdir() if p.is_dir()])
        print(f"[OK] {split}: estratto. sequences={len(seqs)}")
        if seqs:
            ex = seqs[0]
            print(f"[INFO] Esempio sequenza: {ex.name}")
            for child in sorted(ex.iterdir()):
                print("  -", child.name + ("/" if child.is_dir() else ""))
    else:
        print(f"[WARN] {split}: estrazione completata ma cartella attesa non trovata: {expected_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None,
                    help="SoccerNet root (override configs/local_paths.json). Es: .\\data\\soccernet")
    ap.add_argument("--splits", nargs="+", default=["train", "challenge"],
                    choices=["train", "test", "challenge"])
    ap.add_argument("--force", action="store_true", help="Rimuove la cartella estratta e riestrae.")
    args = ap.parse_args()

    root = get_soccernet_root(args.root).resolve()
    base = root / "tracking-2023"

    print(f"[INFO] repo_root      = {REPO_ROOT}")
    print(f"[INFO] soccernet_root = {root}")
    print(f"[INFO] base           = {base}")

    for split in args.splits:
        zip_path = base / f"{split}.zip"
        extract_one(zip_path, base, split, args.force)


if __name__ == "__main__":
    main()
