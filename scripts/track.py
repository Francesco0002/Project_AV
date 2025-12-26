# scripts/track.py
import sys
from pathlib import Path

# aggiunge la root del progetto al PYTHONPATH (fix import src.*)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import json
from dataclasses import dataclass

import cv2
import numpy as np

from src.roi import denorm_roi, bottom_center_xywh
from src.io_mot import write_tracking_line, write_behavior_line, ensure_parent


def load_rois_json(path: str) -> list[dict]:
    # utf-8-sig rimuove automaticamente l'eventuale BOM
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return data["regions"]  # [{"id":1,"x":..,"y":..,"w":..,"h":..}, ...]


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


@dataclass
class SimpleDet:
    """
    Wrapper minimo per tracker.update() (ByteTrack/BoT-SORT Ultralytics).
    Espone: xyxy, xywh, conf, cls, __len__, __getitem__
    """
    xyxy: np.ndarray  # (N,4) float32
    conf: np.ndarray  # (N,) float32
    cls: np.ndarray   # (N,) float32

    @property
    def xywh(self):
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
        return SimpleDet(self.xyxy[idx], self.conf[idx], self.cls[idx])


def build_tracker(tracker_yaml: str, fps: int, overrides: dict):
    """
    Crea BYTETracker o BOTSORT dai yaml Ultralytics.
    """
    from ultralytics.utils.checks import check_yaml
    from ultralytics.utils import IterableSimpleNamespace, YAML
    from ultralytics.trackers.track import TRACKER_MAP

    yaml_path = check_yaml(tracker_yaml)  # risolve "bytetrack.yaml"/"botsort.yaml"
    cfg = IterableSimpleNamespace(**YAML.load(yaml_path))

    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)

    if cfg.tracker_type not in {"bytetrack", "botsort"}:
        raise ValueError(f"tracker_type non supportato: {cfg.tracker_type}")

    tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=fps)
    return tracker, cfg


def draw_rois(vis, rois):
    for roi in rois:
        x1, y1, x2, y2 = int(roi.x), int(roi.y), int(roi.x + roi.w), int(roi.y + roi.h)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(vis, f"ROI {roi.id}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--frames", required=True, help="Cartella img1/ con 000001.jpg ...")
    ap.add_argument("--det", required=True, help="File det MOT-like (frame,-1,x,y,w,h,conf,...)")
    ap.add_argument("--rois", required=True, help="configs/roi/tracking/<seq>.json")

    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--video-id", type=int, default=1)
    ap.add_argument("--team-id", type=int, default=1)

    ap.add_argument("--tracker", default="botsort.yaml", help="botsort.yaml o bytetrack.yaml")
    ap.add_argument("--fps", type=int, default=25)

    # filtro conf prima del tracker (utile se det è 'raw' a 0.05)
    ap.add_argument("--det-conf-min", type=float, default=0.20)

    # override principali (opzionali)
    ap.add_argument("--track-high", type=float, default=None)
    ap.add_argument("--track-low", type=float, default=None)
    ap.add_argument("--new-track", type=float, default=None)
    ap.add_argument("--match-thresh", type=float, default=None)
    ap.add_argument("--track-buffer", type=int, default=None)
    ap.add_argument("--fuse-score", type=int, default=None)  # 0/1

    # video file
    ap.add_argument("--save-video", default=None, help="Path mp4 output (es outputs/vis.mp4)")
    ap.add_argument("--draw-rois", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0, help="0=all, altrimenti limita (es 600)")

    # preview live
    ap.add_argument("--show", action="store_true", help="Mostra preview in tempo reale (imshow)")
    ap.add_argument("--show-every", type=int, default=1, help="Mostra 1 frame ogni N (default=1)")
    ap.add_argument("--show-scale", type=float, default=1.0, help="Scala preview (es 0.75, 0.5)")

    args = ap.parse_args()

    frames_dir = Path(args.frames)
    imgs = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() in [".jpg", ".png", ".jpeg"]])
    if not imgs:
        raise RuntimeError(f"Nessun frame in: {frames_dir}")

    im0 = cv2.imread(str(imgs[0]))
    if im0 is None:
        raise RuntimeError(f"Impossibile leggere: {imgs[0]}")
    H, W = im0.shape[:2]

    # ROIs
    rois_norm = load_rois_json(args.rois)
    rois = [denorm_roi(r, W, H) for r in rois_norm]

    # Detections
    det_path = Path(args.det)
    dets_by_frame = load_mot_det(det_path)

    # Tracker
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
    tracking_path = outdir / f"tracking_{args.video_id}_{args.team_id:02d}.txt"
    behavior_path = outdir / f"behavior_{args.video_id}_{args.team_id:02d}.txt"
    ensure_parent(tracking_path)
    ensure_parent(behavior_path)

    # Video writer
    vw = None
    if args.save_video:
        vpath = Path(args.save_video)
        ensure_parent(vpath)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(vpath), fourcc, args.fps, (W, H))

    # rimappa ID tracker -> ID compatti
    id_map = {}
    next_id = 1

    n_frames = len(imgs)
    limit = args.max_frames if args.max_frames and args.max_frames > 0 else n_frames

    try:
        with tracking_path.open("w", encoding="utf-8") as ftrk, behavior_path.open("w", encoding="utf-8") as fbeh:
            for frame_idx in range(1, min(n_frames, limit) + 1):
                im = cv2.imread(str(imgs[frame_idx - 1]))
                if im is None:
                    continue

                counts = {roi.id: 0 for roi in rois}

                xyxy, conf = dets_by_frame.get(
                    frame_idx,
                    (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32))
                )

                if len(conf):
                    keep = conf >= float(args.det_conf_min)
                    xyxy = xyxy[keep]
                    conf = conf[keep]

                det_obj = SimpleDet(
                    xyxy=xyxy.astype(np.float32),
                    conf=conf.astype(np.float32),
                    cls=np.zeros((len(conf),), dtype=np.float32)  # person=0
                )

                tracks = tracker.update(det_obj, im, None)

                kept_tracks = []

                if tracks is not None and len(tracks):
                    for t in np.asarray(tracks):
                        x1, y1, x2, y2, tid, score, cls_id, det_idx = t.tolist()
                        x = x1
                        y = y1
                        w = x2 - x1
                        h = y2 - y1

                        bc = bottom_center_xywh(x, y, w, h)

                        # ROI gating
                        in_any = any(roi.contains_point(bc[0], bc[1]) for roi in rois)
                        if not in_any:
                            continue

                        tid = int(tid)
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

                        kept_tracks.append((int(x1), int(y1), int(x2), int(y2), oid))

                for roi in rois:
                    write_behavior_line(fbeh, frame_idx, roi.id, counts[roi.id])

                # build visualization if needed (video OR show)
                if (vw is not None) or args.show:
                    vis = im.copy()
                    if args.draw_rois:
                        draw_rois(vis, rois)

                    for (x1, y1, x2, y2, oid) in kept_tracks:
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(vis, f"ID {oid}", (x1, max(0, y1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    if vw is not None:
                        vw.write(vis)

                    if args.show and (frame_idx % args.show_every == 0):
                        show_im = vis
                        if args.show_scale != 1.0:
                            show_im = cv2.resize(
                                show_im, None,
                                fx=args.show_scale, fy=args.show_scale,
                                interpolation=cv2.INTER_LINEAR
                            )
                        cv2.imshow("Tracking preview", show_im)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == 27:  # q o ESC
                            break

    finally:
        if vw is not None:
            vw.release()
        if args.show:
            cv2.destroyAllWindows()

    print(f"[OK] tracking:  {tracking_path}")
    print(f"[OK] behavior:  {behavior_path}")
    if args.save_video:
        print(f"[OK] video:     {args.save_video}")
    print(f"[INFO] tracker:  {cfg.tracker_type} ({args.tracker})")


if __name__ == "__main__":
    main()
