import argparse
from pathlib import Path
from SoccerNet.Downloader import SoccerNetDownloader

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", nargs="+", default=["challenge"], choices=["train","test","challenge"])
    args = ap.parse_args()

    root = Path("data") / "soccernet"
    root.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Download dir: {root.resolve()}")

    d = SoccerNetDownloader(LocalDirectory=str(root))
    d.downloadDataTask(task="tracking-2023", split=args.split)

    print("[OK] Download completed.")

if __name__ == "__main__":
    main()
