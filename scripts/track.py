# track.py
import os
import sys
from pathlib import Path
import argparse
import json
from dataclasses import dataclass

import cv2
import numpy as np

# -------------------------
# Robust import src.*
# -------------------------
SOCCERNET = os.environ.get("SOCCERNET", "")
if SOCCERNET and Path(SOCCERNET).exists():
    sys.path.insert(0, SOCCERNET)
else:
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))

from src.roi import denorm_roi, bottom_center_xywh
from src.io_mot import write_tracking_line, ensure_parent


# -------------------------
# ROI loading (supports regions / rois / list / contest roi1 roi2)
# -------------------------
def load_rois_json(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))

    # caso 1) formato già supportato
    if isinstance(data, dict):
        if "regions" in data and isinstance(data["regions"], list):
            return data["regions"]
        if "rois" in data and isinstance(data["rois"], list):
            return data["rois"]

        # caso 2) formato contest: roi1 / roi2
        if "roi1" in data and "roi2" in data:
            out = []
            for i, key in enumerate(["roi1", "roi2"], start=1):
                r = data[key]
                out.append({
                    "id": i,
                    "x": r["x"],
                    "y": r["y"],
                    "w": r.get("w", r.get("width")),
                    "h": r.get("h", r.get("height")),
                })
            return out

    # caso 3) lista
    if isinstance(data, list):
        return data

    raise ValueError(f"Formato ROI JSON non supportato: {path}")


# -------------------------
# MOT detections
# -------------------------
def load_mot_det(det_path: Path):
    """
    MOT-det: frame, -1, x, y, w, h, conf, -1, -1, -1
    Ritorna dict: frame_idx -> (xyxy[N,4], conf[N])
    """
    dets = {}
    with det_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split(",")
            if len(p) < 7:
                continue
            frame = int(float(p[0]))
            x = float(p[2]); y = float(p[3])
            w = float(p[4]); h = float(p[5])
            conf = float(p[6])
            x1, y1, x2, y2 = x, y, x + w, y + h
            dets.setdefault(frame, []).append((x1, y1, x2, y2, conf))

    out = {}
    for k, lst in dets.items():
        arr = np.array(lst, dtype=np.float32)
        out[k] = (arr[:, :4], arr[:, 4])
    return out


# -------------------------
# Wrapper for Ultralytics trackers
# -------------------------
@dataclass
class SimpleDet:
    """
    Wrapper minimo per tracker.update() (ByteTrack/BoT-SORT Ultralytics).
    Espone: xyxy, xywh, conf, cls, __len__
    """
    xyxy: np.ndarray  # (N,4) float32
    conf: np.ndarray  # (N,) float32
    cls: np.ndarray   # (N,) float32

    @property
    def xywh(self):
        # cx,cy,w,h
        if len(self.xyxy) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        x1 = self.xyxy[:, 0]
        y1 = self.xyxy[:, 1]
        x2 = self.xyxy[:, 2]
        y2 = self.xyxy[:, 3]
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        return np.stack([cx, cy, w, h], axis=1).astype(np.float32)

    def __len__(self):
        return int(self.conf.shape[0])

    def __getitem__(self, idx):
        return SimpleDet(
            xyxy=self.xyxy[idx],
            conf=self.conf[idx],
            cls=self.cls[idx],
        )


def build_tracker(tracker_yaml: str, fps: int, overrides: dict):
    from ultralytics.utils.checks import check_yaml
    from ultralytics.utils import IterableSimpleNamespace, YAML
    from ultralytics.trackers.track import TRACKER_MAP

    yaml_path = check_yaml(tracker_yaml)
    cfg = IterableSimpleNamespace(**YAML.load(yaml_path))

    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)

    if cfg.tracker_type not in {"bytetrack", "botsort"}:
        raise ValueError(f"tracker_type non supportato: {cfg.tracker_type}")

    tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=fps)
    return tracker, cfg


def draw_rois(vis, rois, counts=None, alpha=0.25):
    """
    Disegna ROI 1 rossa, ROI 2 blu, overlay semitrasparente.
    counts: dict {roi_id: n_players} opzionale, per stampare i conteggi.
    """
    overlay = vis.copy()

    # colori BGR (OpenCV): rosso e blu
    color_map = {
        1: (0, 0, 255),   # ROI1 rosso
        2: (255, 0, 0),   # ROI2 blu
    }

    for roi in rois:
        rid = int(roi.id)
        x1, y1 = int(roi.x), int(roi.y)
        x2, y2 = int(roi.x + roi.w), int(roi.y + roi.h)

        c = color_map.get(rid, (255, 255, 255))

        # riempimento su overlay
        cv2.rectangle(overlay, (x1, y1), (x2, y2), c, -1)
        # bordo pieno
        cv2.rectangle(vis, (x1, y1), (x2, y2), c, 3)

        label = f"ROI {rid}"
        if counts is not None:
            label += f": {counts.get(rid, 0)}"
        cv2.putText(vis, label, (x1 + 6, y1 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)

    # blend overlay -> vis
    cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--frames", required=True, help="Cartella img1/ con 000001.jpg ...")
    ap.add_argument("--det", required=True, help="File det MOT-like (frame,-1,x,y,w,h,conf,...)")
    ap.add_argument("--rois", required=True, help="configs/roi/...json")

    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--video-id", type=int, required=True, help="K (1..5) o id video")
    ap.add_argument("--team-id", type=int, required=True, help="XX (team id, verrà zero-padded)")

    ap.add_argument("--tracker", default="botsort.yaml", help="botsort.yaml o bytetrack.yaml")
    ap.add_argument("--fps", type=int, default=25)

    ap.add_argument("--det-conf-min", type=float, default=0.05)

    # override tracker
    ap.add_argument("--track-high", type=float, default=None)
    ap.add_argument("--track-low", type=float, default=None)
    ap.add_argument("--new-track", type=float, default=None)
    ap.add_argument("--match-thresh", type=float, default=None)
    ap.add_argument("--track-buffer", type=int, default=None)
    ap.add_argument("--fuse-score", type=int, default=None)  # 0/1

    # 2 video separati
    ap.add_argument("--save-video-track", default=None,
                    help="Salva video SOLO tracking (bbox+ID)")
    ap.add_argument("--save-video-beh", default=None,
                    help="Salva video SOLO behavior (ROI+conteggi)")
    ap.add_argument("--beh-show-tracks", action="store_true",
                    help="Nel video behavior disegna anche bbox+ID (opzionale)")

    ap.add_argument("--draw-rois", action="store_true",
                    help="Disegna ROI (necessario per save-video-beh)")

    args = ap.parse_args()

    frames_dir = Path(args.frames)
    if not frames_dir.exists():
        raise RuntimeError(f"--frames non esiste: {frames_dir}")

    imgs = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() in [".jpg", ".png", ".jpeg"]])
    if not imgs:
        raise RuntimeError(f"Nessun frame in: {frames_dir}")

    im0 = cv2.imread(str(imgs[0]))
    if im0 is None:
        raise RuntimeError(f"Impossibile leggere: {imgs[0]}")
    H, W = im0.shape[:2]

    # ROIs (ordinate per ID)
    rois_raw = load_rois_json(args.rois)
    rois = [denorm_roi(r, W, H) for r in rois_raw]
    rois = sorted(rois, key=lambda r: int(r.id))
    roi_ids = [int(r.id) for r in rois]

    # dets
    dets_by_frame = load_mot_det(Path(args.det))

    # tracker
    overrides = {
        "track_high_thresh": args.track_high,
        "track_low_thresh": args.track_low,
        "new_track_thresh": args.new_track,
        "match_thresh": args.match_thresh,
        "track_buffer": args.track_buffer,
        "fuse_score": bool(args.fuse_score) if args.fuse_score is not None else None,
    }
    tracker, cfg = build_tracker(args.tracker, args.fps, overrides)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # tracking_K_XX.txt / behavior_K_XX.txt
    tracking_path = outdir / f"tracking_{args.video_id}_{args.team_id:02d}.txt"
    behavior_path = outdir / f"behavior_{args.video_id}_{args.team_id:02d}.txt"
    ensure_parent(tracking_path)
    ensure_parent(behavior_path)

    # video writers
    vw_track = None
    vw_beh = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    if args.save_video_track:
        ensure_parent(Path(args.save_video_track))
        vw_track = cv2.VideoWriter(str(args.save_video_track), fourcc, float(args.fps), (W, H))

    if args.save_video_beh:
        ensure_parent(Path(args.save_video_beh))
        vw_beh = cv2.VideoWriter(str(args.save_video_beh), fourcc, float(args.fps), (W, H))

    # remap ids to compact 1..N
    id_map = {}
    next_id = 1

    with tracking_path.open("w", encoding="utf-8") as ftrk, behavior_path.open("w", encoding="utf-8") as fbeh:
        for frame_idx, img_path in enumerate(imgs, start=1):
            im = cv2.imread(str(img_path))
            if im is None:
                continue

            xyxy, conf = dets_by_frame.get(
                frame_idx,
                (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32))
            )

            # pre-filter detections
            if len(conf):
                keep = conf >= float(args.det_conf_min)
                xyxy = xyxy[keep]
                conf = conf[keep]

            det_obj = SimpleDet(
                xyxy=xyxy.astype(np.float32),
                conf=conf.astype(np.float32),
                cls=np.zeros((len(conf),), dtype=np.float32)  # single class
            )

            tracks = tracker.update(det_obj, im, None)

            # behavior counts per ROI
            counts = {rid: 0 for rid in roi_ids}
            kept_tracks = []

            if tracks is not None and len(tracks):
                for t in np.asarray(tracks):
                    x1, y1, x2, y2, tid, score, cls_id, det_idx = t.tolist()

                    # clamp
                    x1 = float(max(0, min(W - 1, x1)))
                    y1 = float(max(0, min(H - 1, y1)))
                    x2 = float(max(0, min(W - 1, x2)))
                    y2 = float(max(0, min(H - 1, y2)))
                    if x2 <= x1 or y2 <= y1:
                        continue

                    w = x2 - x1
                    h = y2 - y1

                    # remap ID -> 1..N
                    tid = int(tid)
                    if tid not in id_map:
                        id_map[tid] = next_id
                        next_id += 1
                    oid = id_map[tid]

                    # write tracking (xywh top-left)
                    write_tracking_line(
                        ftrk, frame_idx, oid,
                        int(round(x1)), int(round(y1)),
                        int(round(w)), int(round(h))
                    )

                    # behavior: bottom-center inside ROI?
                    fx, fy = bottom_center_xywh(x1, y1, w, h)
                    for roi in rois:
                        if roi.contains_point(fx, fy):
                            counts[int(roi.id)] += 1

                    kept_tracks.append((int(x1), int(y1), int(x2), int(y2), oid, float(score)))

            # write behavior EXACT format: frame_id,region_id,n_players
            for rid in roi_ids:
                fbeh.write(f"{frame_idx},{rid},{counts[rid]}\n")

            # -------------------------
            # VIDEO 1: tracking only
            # -------------------------
            if vw_track is not None:
                vis_t = im.copy()
                for (x1, y1, x2, y2, oid, sc) in kept_tracks:
                    cv2.rectangle(vis_t, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis_t, f"ID {oid}", (x1, max(0, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                vw_track.write(vis_t)

            # -------------------------
            # VIDEO 2: behavior only
            # -------------------------
            if vw_beh is not None:
                vis_b = im.copy()

                if args.draw_rois:
                    draw_rois(vis_b, rois, counts=counts, alpha=0.25)

                if args.beh_show_tracks:
                    for (x1, y1, x2, y2, oid, sc) in kept_tracks:
                        cv2.rectangle(vis_b, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(vis_b, f"ID {oid}", (x1, max(0, y1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                vw_beh.write(vis_b)

    if vw_track is not None:
        vw_track.release()
    if vw_beh is not None:
        vw_beh.release()

    print(f"[OK] tracking: {tracking_path}")
    print(f"[OK] behavior: {behavior_path}")
    if args.save_video_track:
        print(f"[OK] video tracking: {args.save_video_track}")
    if args.save_video_beh:
        print(f"[OK] video behavior: {args.save_video_beh}")
    print(f"[INFO] tracker: {cfg.tracker_type} ({args.tracker})")


if __name__ == "__main__":
    main()