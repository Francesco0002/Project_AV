# detect_foot.py
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# -------------------------
# Mask utilities (HSV field)
# -------------------------
def hsv_inrange_mask(bgr_img, h_lo=20, h_hi=110, s_lo=25, v_lo=20):
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    lower = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
    upper = np.array([h_hi, 255, 255], dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)


def clean_mask_light(mask):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask


def break_horizontal_bridges(mask, kw=3, kh=25):
    """
    Rimuove 'ponti' orizzontali sottili (cartelloni) che connettono spalti e campo.
    Opening con kernel verticale (kw x kh). Se kh<=0: off.
    """
    if kh is None or kh <= 0:
        return mask
    kw = max(1, int(kw))
    kh = max(1, int(kh))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)


def keep_component_bottom_touch(mask, bottom_frac=0.18):
    """Fallback: tieni la componente più grande che tocca la fascia bassa."""
    H, W = mask.shape[:2]
    m = (mask > 0).astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return mask

    y0 = int(H * (1.0 - float(bottom_frac)))
    bottom_labels = np.unique(labels[y0:H, :])
    bottom_labels = bottom_labels[bottom_labels != 0]
    if len(bottom_labels) == 0:
        return mask

    best = max(bottom_labels, key=lambda l: stats[l, cv2.CC_STAT_AREA])
    out = np.zeros_like(mask)
    out[labels == best] = 255
    return out


def keep_components_by_footpoints_union(
    mask,
    foot_pts,
    min_fp_total=2,
    min_fp_each=1,
    max_components=2,
    radius=7,
    fallback_bottom_frac=0.18,
    cc_close_ksize=0,
):
    """
    Tiene l'UNIONE delle top-K componenti agganciate dai footpoints.
    Utile quando la linea di metà campo spezza la maschera in due componenti.
    """
    H, W = mask.shape[:2]
    m = (mask > 0).astype(np.uint8)

    # opzionale: ricuce piccole interruzioni (es. linee bianche sottili)
    if cc_close_ksize and int(cc_close_ksize) > 0:
        ksz = int(cc_close_ksize)
        if ksz % 2 == 0:
            ksz += 1
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return mask

    fp_count = np.zeros((n,), dtype=np.int32)
    r = int(max(1, radius))

    # conta footpoints per label usando finestra
    for (bx, by) in foot_pts:
        bx = max(0, min(W - 1, int(bx)))
        by = max(0, min(H - 1, int(by)))

        x1 = max(0, bx - r); x2 = min(W, bx + r + 1)
        y1 = max(0, by - r); y2 = min(H, by + r + 1)

        win = labels[y1:y2, x1:x2]
        win = win[win != 0]
        if win.size == 0:
            continue

        l = int(np.bincount(win.ravel()).argmax())
        fp_count[l] += 1

    # se non ho abbastanza agganci -> fallback bottom-touch
    if int(fp_count.sum()) < int(min_fp_total) or fp_count.max() == 0:
        return keep_component_bottom_touch(mask, bottom_frac=float(fallback_bottom_frac))

    # seleziona componenti "valide"
    valid = np.where(fp_count >= int(min_fp_each))[0]
    valid = valid[valid != 0]

    if valid.size == 0:
        return keep_component_bottom_touch(mask, bottom_frac=float(fallback_bottom_frac))

    # ordina per (fp_count desc, area desc)
    valid_sorted = sorted(
        valid.tolist(),
        key=lambda l: (int(fp_count[l]), int(stats[l, cv2.CC_STAT_AREA])),
        reverse=True
    )

    # prendi top-K
    k = int(max_components) if max_components else 2
    chosen = valid_sorted[:max(1, k)]

    out = np.zeros_like(mask)
    for l in chosen:
        out[labels == l] = 255
    return out


def on_field_window(mask, bx, by, radius=5):
    H, W = mask.shape[:2]
    x1 = max(0, bx - radius)
    x2 = min(W, bx + radius + 1)
    y1 = max(0, by - radius)
    y2 = min(H, by + radius + 1)
    return mask[y1:y2, x1:x2].any()


def overlay_mask(im_bgr, mask, alpha=0.35, color=(0, 255, 0)):
    out = im_bgr.copy()
    if mask is None:
        return out
    m = mask.astype(bool)
    overlay = out.copy()
    overlay[m] = (np.array(color, dtype=np.uint8))
    out = cv2.addWeighted(overlay, float(alpha), out, 1 - float(alpha), 0)
    return out


def apply_preset(args):
    if args.preset is None:
        return args

    if args.preset == "general":
        args.field_filter = True
        args.pitch_component = True
        if args.component_mode is None:
            args.component_mode = "footpoints"

        if args.pitch_bottom_frac is None:
            args.pitch_bottom_frac = 0.18

        if args.bridge_kh is None:
            args.bridge_kh = 25
        if args.bridge_kw is None:
            args.bridge_kw = 3

        if args.fp_radius is None:
            args.fp_radius = 7
        if args.fp_min_total is None:
            args.fp_min_total = 2
        if args.fp_min_each is None:
            args.fp_min_each = 1
        if args.fp_max_components is None:
            args.fp_max_components = 2

        if args.cc_close_ksize is None:
            args.cc_close_ksize = 0  # 0=off, prova 5 se serve

        if args.mask_dilate is None:
            args.mask_dilate = 5
        if args.foot_radius is None:
            args.foot_radius = 5

        if args.edge_margin is None:
            args.edge_margin = 0.0
        if args.big_h_frac is None:
            args.big_h_frac = 0.18

    elif args.preset == "hard":
        args.field_filter = True
        args.pitch_component = True
        if args.component_mode is None:
            args.component_mode = "footpoints"

        if args.pitch_bottom_frac is None:
            args.pitch_bottom_frac = 0.18

        if args.bridge_kh is None:
            args.bridge_kh = 25
        if args.bridge_kw is None:
            args.bridge_kw = 3

        if args.fp_radius is None:
            args.fp_radius = 7
        if args.fp_min_total is None:
            args.fp_min_total = 2
        if args.fp_min_each is None:
            args.fp_min_each = 1
        if args.fp_max_components is None:
            args.fp_max_components = 2

        if args.cc_close_ksize is None:
            args.cc_close_ksize = 0

        if args.mask_dilate is None:
            args.mask_dilate = 5
        if args.foot_radius is None:
            args.foot_radius = 5

        if args.edge_margin is None:
            args.edge_margin = 8.0
        if args.big_h_frac is None:
            args.big_h_frac = 0.16

    return args


def _parse_classes(s: str):
    """
    s:
      - "0" o "0,1" -> [0,1]
      - "all" / "" / "none" -> None (nessun filtro)
    """
    if s is None:
        return None
    s = s.strip().lower()
    if s in ("", "all", "none"):
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", default="outputs/detections/det.txt")
    ap.add_argument("--weights", default="yolo11x.pt")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--save-vis", type=int, default=0)

    ap.add_argument("--device", default=None)
    ap.add_argument("--half", action="store_true")

    ap.add_argument("--preset", choices=["general", "hard"], default=None)

    # filtri classi 
    ap.add_argument("--classes", type=str, default="0",
                    help="Classi da TENERE (es '0' o '0,1'). Usa 'all' per non filtrare.")
    ap.add_argument("--exclude-classes", type=str, default="",
                    help="Classi da ESCLUDERE (es '32' per togliere ball se fai --classes all).")

    # filtro campo
    ap.add_argument("--field-filter", dest="field_filter", action="store_true")
    ap.add_argument("--pitch-component", action="store_true")
    ap.add_argument("--component-mode", choices=["footpoints", "bottom"], default=None)

    ap.add_argument("--pitch-bottom-frac", type=float, default=None)

    # rompe connessioni ads
    ap.add_argument("--bridge-kw", type=int, default=None)
    ap.add_argument("--bridge-kh", type=int, default=None)

    # footpoints union params
    ap.add_argument("--fp-radius", type=int, default=None)
    ap.add_argument("--fp-min-total", dest="fp_min_total", type=int, default=None)
    ap.add_argument("--fp-min-each", dest="fp_min_each", type=int, default=None)
    ap.add_argument("--fp-max-components", type=int, default=None)
    ap.add_argument("--cc-close-ksize", type=int, default=None,
                    help="0=off, prova 5 se la linea di metà campo spezza il prato")

    ap.add_argument("--mask-dilate", type=int, default=None)
    ap.add_argument("--foot-radius", type=int, default=None)

    # HSV
    ap.add_argument("--h-lo", type=int, default=20)
    ap.add_argument("--h-hi", type=int, default=110)
    ap.add_argument("--s-lo", type=int, default=25)
    ap.add_argument("--v-lo", type=int, default=20)

    # edge margin
    ap.add_argument("--edge-margin", type=float, default=None)
    ap.add_argument("--big-h-frac", type=float, default=None)

    # Debug
    ap.add_argument("--save-fieldvis", type=int, default=0)
    ap.add_argument("--fieldvis-every", type=int, default=30)
    ap.add_argument("--fieldvis-alpha", type=float, default=0.35)

    # Cache maschera
    ap.add_argument("--mask-every", type=int, default=1)

    args = ap.parse_args()
    args = apply_preset(args)

    # fallback
    if args.component_mode is None:
        args.component_mode = "footpoints"
    if args.pitch_bottom_frac is None:
        args.pitch_bottom_frac = 0.18
    if args.bridge_kw is None:
        args.bridge_kw = 3
    if args.bridge_kh is None:
        args.bridge_kh = 25
    if args.fp_radius is None:
        args.fp_radius = 7
    if args.fp_min_total is None:
        args.fp_min_total = 2
    if args.fp_min_each is None:
        args.fp_min_each = 1
    if args.fp_max_components is None:
        args.fp_max_components = 2
    if args.cc_close_ksize is None:
        args.cc_close_ksize = 0
    if args.mask_dilate is None:
        args.mask_dilate = 5
    if args.foot_radius is None:
        args.foot_radius = 5
    if args.edge_margin is None:
        args.edge_margin = 0.0
    if args.big_h_frac is None:
        args.big_h_frac = 0.18

    # parse class filters
    keep_classes = _parse_classes(args.classes)
    exclude_classes = set(_parse_classes(args.exclude_classes) or [])

    frames_dir = Path(args.frames)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    vis_dir = out_path.parent / f"vis_{out_path.stem}"
    if args.save_vis and args.save_vis > 0:
        vis_dir.mkdir(parents=True, exist_ok=True)

    fieldvis_dir = out_path.parent / f"fieldvis_{out_path.stem}"
    if args.save_fieldvis == 1:
        fieldvis_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)

    results = model.predict(
        source=str(frames_dir),
        stream=True,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        classes=keep_classes,        # filtro classi
        device=args.device,
        half=args.half,
        verbose=False,
    )

    n_written = 0

    cached_mask_edge = None
    cached_mask_keep = None
    cached_dist = None
    cached_hw = None

    with out_path.open("a", encoding="utf-8") as f_out:
        for frame_idx, r in enumerate(results, start=1):
            im = r.orig_img
            H, W = im.shape[:2]

            if r.boxes is not None and r.boxes.xyxy is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                clsids = r.boxes.cls.cpu().numpy().astype(np.int32)
            else:
                xyxy = np.zeros((0, 4), dtype=np.float32)
                confs = np.zeros((0,), dtype=np.float32)
                clsids = np.zeros((0,), dtype=np.int32)

            # extra sicurezza: escludi classi anche se keep_classes=all
            if clsids.size > 0 and len(exclude_classes) > 0:
                keep = np.array([c not in exclude_classes for c in clsids], dtype=bool)
                xyxy = xyxy[keep]
                confs = confs[keep]
                clsids = clsids[keep]

            # footpoints per stimare il campo (usiamo SOLO le classi tenute)
            raw_pts = []
            for (x1, y1, x2, y2) in xyxy:
                bx = int(round((x1 + x2) * 0.5))
                by = int(round(y2))
                bx = max(0, min(W - 1, bx))
                by = max(0, min(H - 1, by))
                raw_pts.append((bx, by))

            mask_edge = None
            mask_keep = None
            dist = None

            need_mask = bool(args.field_filter) or (args.save_fieldvis == 1)
            if need_mask:
                need_recompute = (
                    cached_mask_edge is None
                    or cached_hw != (H, W)
                    or (args.mask_every and args.mask_every > 1 and (frame_idx % args.mask_every == 1))
                    or (args.mask_every == 1)
                )

                if need_recompute:
                    mask_raw = hsv_inrange_mask(im, args.h_lo, args.h_hi, args.s_lo, args.v_lo)
                    mask_raw = break_horizontal_bridges(mask_raw, kw=int(args.bridge_kw), kh=int(args.bridge_kh))

                    mask_edge = mask_raw
                    if args.pitch_component:
                        if args.component_mode == "footpoints":
                            mask_edge = keep_components_by_footpoints_union(
                                mask_edge,
                                raw_pts,
                                min_fp_total=int(args.fp_min_total),
                                min_fp_each=int(args.fp_min_each),
                                max_components=int(args.fp_max_components),
                                radius=int(args.fp_radius),
                                fallback_bottom_frac=float(args.pitch_bottom_frac),
                                cc_close_ksize=int(args.cc_close_ksize),
                            )
                        else:
                            mask_edge = keep_component_bottom_touch(mask_edge, bottom_frac=float(args.pitch_bottom_frac))

                    mask_edge = clean_mask_light(mask_edge)

                    mask_keep = mask_edge.copy()
                    if args.mask_dilate and int(args.mask_dilate) > 0:
                        k = cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE, (int(args.mask_dilate), int(args.mask_dilate))
                        )
                        mask_keep = cv2.dilate(mask_keep, k, iterations=1)

                    dist = None
                    if args.edge_margin and float(args.edge_margin) > 0:
                        # distanza dal bordo del prato: distanceTransform sui pixel >0 (prato)
                        dist = cv2.distanceTransform(mask_edge, distanceType=cv2.DIST_L2, maskSize=5)

                    cached_mask_edge = mask_edge
                    cached_mask_keep = mask_keep
                    cached_dist = dist
                    cached_hw = (H, W)
                else:
                    mask_edge = cached_mask_edge
                    mask_keep = cached_mask_keep
                    dist = cached_dist

            kept = []
            kept_pts = []

            if args.field_filter and len(xyxy) > 0:
                for (x1, y1, x2, y2), c in zip(xyxy, confs):
                    bx = int(round((x1 + x2) * 0.5))
                    by = int(round(y2))
                    bx = max(0, min(W - 1, bx))
                    by = max(0, min(H - 1, by))

                    if mask_keep is not None:
                        if not on_field_window(mask_keep, bx, by, radius=int(args.foot_radius)):
                            continue

                        if dist is not None and float(args.edge_margin) > 0:
                            h_pix = float(y2 - y1)
                            is_big = h_pix > (float(args.big_h_frac) * H)
                            if is_big and dist[by, bx] < float(args.edge_margin):
                                continue

                    x = int(round(x1)); y = int(round(y1))
                    w = int(round(x2 - x1)); h = int(round(y2 - y1))
                    f_out.write(f"{frame_idx},-1,{x},{y},{w},{h},{float(c):.6f},-1,-1,-1\n")
                    n_written += 1
                    kept.append((x1, y1, x2, y2, c))
                    kept_pts.append((bx, by))
            else:
                for (x1, y1, x2, y2), c in zip(xyxy, confs):
                    x = int(round(x1)); y = int(round(y1))
                    w = int(round(x2 - x1)); h = int(round(y2 - y1))
                    f_out.write(f"{frame_idx},-1,{x},{y},{w},{h},{float(c):.6f},-1,-1,-1\n")
                    n_written += 1
                    kept.append((x1, y1, x2, y2, c))
                    bx = int(round((x1 + x2) * 0.5)); by = int(round(y2))
                    kept_pts.append((max(0, min(W - 1, bx)), max(0, min(H - 1, by))))

            if args.save_vis and args.save_vis > 0 and frame_idx <= args.save_vis:
                vis = im.copy()
                for (x1, y1, x2, y2, c) in kept:
                    cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(vis, f"{c:.2f}", (int(x1), max(0, int(y1) - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imwrite(str(vis_dir / f"{frame_idx:06d}.jpg"), vis)

            if args.save_fieldvis == 1 and (frame_idx % int(args.fieldvis_every) == 0):
                fv = overlay_mask(im, mask_edge, alpha=float(args.fieldvis_alpha), color=(0, 255, 0))
                for (bx, by) in raw_pts:
                    cv2.circle(fv, (bx, by), 3, (0, 0, 255), -1)
                for (bx, by) in kept_pts:
                    cv2.circle(fv, (bx, by), 3, (255, 0, 0), -1)

                cv2.putText(
                    fv,
                    "GREEN=field mask | RED=raw feet | BLUE=kept feet",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2
                )
                cv2.imwrite(str(fieldvis_dir / f"fieldvis_{frame_idx:06d}.jpg"), fv)

    print(f"[OK] detections written: {n_written}")
    print(f"[OK] det file: {out_path.resolve()}")
    if args.save_vis and args.save_vis > 0:
        print(f"[OK] vis dir:       {vis_dir.resolve()}")
    if args.save_fieldvis == 1:
        print(f"[OK] fieldvis dir:  {fieldvis_dir.resolve()}")


if __name__ == "__main__":
    main()