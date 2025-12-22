#lasciamo un entrypoint che in futuro collegheremo a YOLO 
# + ByteTrack/BoT-SORT, e scrive in outputs/:

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.roi import denorm_roi, bottom_center_xywh
from src.io_mot import write_tracking_line, write_behavior_line, ensure_parent

def load_rois_json(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["regions"]  # [{"id":1,"x":..,"y":..,"w":..,"h":..}, ...]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="Cartella img1/ con 000001.jpg ...")
    ap.add_argument("--rois", required=True, help="configs/roi/<seq>.json")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--video-id", type=int, default=1)
    ap.add_argument("--team-id", type=int, default=1)

    ap.add_argument("--weights", default="yolo8m.pt")
    ap.add_argument("--tracker", default="botsort.yaml")  # o bytetrack.yaml
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--imgsz", type=int, default=1280)
    args = ap.parse_args()

    frames_dir = Path(args.frames)
    imgs = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() in [".jpg", ".png", ".jpeg"]])
    if not imgs:
        raise RuntimeError(f"Nessun frame in: {frames_dir}")

    im0 = cv2.imread(str(imgs[0]))
    H, W = im0.shape[:2]

    rois_norm = load_rois_json(args.rois)
    rois = [denorm_roi(r, W, H) for r in rois_norm]

    outdir = Path(args.outdir)
    tracking_path = outdir / f"tracking_{args.video_id}_{args.team_id:02d}.txt"
    behavior_path = outdir / f"behavior_{args.video_id}_{args.team_id:02d}.txt"
    ensure_parent(tracking_path)
    ensure_parent(behavior_path)

    model = YOLO(args.weights)

    id_map = {}
    next_id = 1

    with tracking_path.open("w", encoding="utf-8") as ftrk, behavior_path.open("w", encoding="utf-8") as fbeh:
        results = model.track(
            source=str(frames_dir),
            stream=True,
            persist=True,
            tracker=args.tracker,
            classes=[0],      # person
            conf=args.conf,
            imgsz=args.imgsz,
            verbose=False
        )

        for frame_idx, r in enumerate(results, start=1):
            counts = {roi.id: 0 for roi in rois}

            if r.boxes is None or r.boxes.xyxy is None or r.boxes.id is None:
                for roi in rois:
                    write_behavior_line(fbeh, frame_idx, roi.id, 0)
                continue

            xyxy = r.boxes.xyxy.cpu().numpy()
            tids = r.boxes.id.cpu().numpy().astype(int)

            for bb, tid in zip(xyxy, tids):
                x1, y1, x2, y2 = bb.tolist()
                x = x1
                y = y1
                w = x2 - x1
                h = y2 - y1

                bc = bottom_center_xywh(x, y, w, h)
                in_any = any(roi.contains_point(bc[0], bc[1]) for roi in rois)
                if not in_any:
                    continue  # criterio ROI richiesto

                if tid not in id_map:
                    id_map[tid] = next_id
                    next_id += 1
                oid = id_map[tid]

                write_tracking_line(
                    ftrk, frame_idx, oid,
                    int(round(x)), int(round(y)), int(round(w)), int(round(h))
                )

                for roi in rois:
                    if roi.contains_point(bc[0], bc[1]):
                        counts[roi.id] += 1

            for roi in rois:
                write_behavior_line(fbeh, frame_idx, roi.id, counts[roi.id])

    print(f"[OK] {tracking_path}")
    print(f"[OK] {behavior_path}")

if __name__ == "__main__":
    main()
