from pathlib import Path
from SoccerNet.Downloader import SoccerNetDownloader

def main():
    root = Path("data") / "soccernet"
    root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Download dir: {root.resolve()}")

    d = SoccerNetDownloader(LocalDirectory=str(root))
    d.downloadDataTask(task="tracking-2023", split=["challenge"])

    print("[OK] Download completed.")

if __name__ == "__main__":
    main()
