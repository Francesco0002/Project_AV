import json
from pathlib import Path

#legge i path locali

def load_local_paths() -> dict:
    p = Path("configs") / "local_paths.json"
    if not p.exists():
        raise FileNotFoundError(
            "Manca configs/local_paths.json. Crealo copiando l'esempio e impostando soccernet_root."
        )
    return json.loads(p.read_text(encoding="utf-8"))

def soccernet_root() -> Path:
    return Path(load_local_paths()["soccernet_root"])