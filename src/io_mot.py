from pathlib import Path

def write_tracking_line(fp, frame_id: int, obj_id: int, x: int, y: int, w: int, h: int):
    fp.write(f"{frame_id},{obj_id},{x},{y},{w},{h}\n")

def write_behavior_line(fp, frame_id: int, region_id: int, n_players: int):
    fp.write(f"{frame_id},{region_id},{n_players}\n")

def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)