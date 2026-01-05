# mot_to_behavior_gt.py
from pathlib import Path
import argparse
import json
import cv2
import os

# Robust import src.*
SOCCERNET = os.environ.get("SOCCERNET", "")
if SOCCERNET and Path(SOCCERNET).exists():
    sys.path.insert(0, SOCCERNET)
else:
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))

from src.roi import denorm_roi, bottom_center_xywh

def load_rois_json(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(data, dict):
        if "regions" in data and isinstance(data["regions"], list):
            return data["regions"]
        if "rois" in data and isinstance(data["rois"], list):
            return data["rois"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Formato ROI JSON non supportato: {path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="cartella img1 per ricavare W,H e num frame")
    ap.add_argument("--mot", required=True, help="gt/gt.txt (o tracking pred) in formato MOT")
    ap.add_argument("--rois", required=True, help="roi json")
    ap.add_argument("--out", required=True, help="output behavior_gt.txt")
    ap.add_argument("--skip-conf-zero", action="store_true",
                    help="se colonna 7 esiste e vale 0, ignora (tipico MOT ignore)")
    args = ap.parse_args()

    frames_dir = Path(args.frames)
    imgs = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() in [".jpg", ".png", ".jpeg"]])
    if not imgs:
        raise RuntimeError("Nessun frame trovato")

    im0 = cv2.imread(str(imgs[0]))
    H, W = im0.shape[:2]
    n_frames = len(imgs)

    rois_raw = load_rois_json(args.rois)
    rois = sorted([denorm_roi(r, W, H) for r in rois_raw], key=lambda r: int(r.id))

    # carica mot per frame -> lista bbox
    per_frame = {}
    with Path(args.mot).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split(",")
            if len(p) < 6:
                continue
            fr = int(float(p[0]))
            x  = float(p[2]); y = float(p[3])
            w  = float(p[4]); h = float(p[5])

            if args.skip_conf_zero and len(p) >= 7:
                conf = float(p[6])
                if conf == 0:
                    continue

            per_frame.setdefault(fr, []).append((x, y, w, h))

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    with outp.open("w", encoding="utf-8") as fo:
        for fr in range(1, n_frames + 1):
            counts = {int(r.id): 0 for r in rois}
            for (x, y, w, h) in per_frame.get(fr, []):
                fx, fy = bottom_center_xywh(x, y, w, h)
                for r in rois:
                    if r.contains_point(fx, fy):
                        counts[int(r.id)] += 1

            for r in rois:
                fo.write(f"{fr},{int(r.id)},{counts[int(r.id)]}\n")

    print(f"[OK] wrote {outp}")

if __name__ == "__main__":
    main()

#run tipico
#python mot_to_behavior_gt.py \
#  --frames data/.../SNMOT-060/img1 \
#  --mot    data/.../SNMOT-060/gt/gt.txt \
#  --rois   configs/roi/tracking/SNMOT-060.json \
#  --out    outputs/behavior_GT_060.txt \
#  --skip-conf-zero