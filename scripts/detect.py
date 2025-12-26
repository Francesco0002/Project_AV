# scripts/detect.py
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def field_mask_hsv(bgr_img, h_lo=20, h_hi=110, s_lo=25, v_lo=20):
    """Mask binaria del prato usando HSV. Ritorna uint8 {0,255}."""
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    lower = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
    upper = np.array([h_hi, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # pulizia base
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask


def keep_pitch_component(mask, bottom_frac=0.18):
    """
    Tiene solo la componente connessa più grande che tocca una fascia bassa dell'immagine.
    Elimina 'verde' sugli spalti/pubblicità che passa l'HSV.
    """
    H, W = mask.shape[:2]
    m = (mask > 0).astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return mask

    y0 = int(H * (1.0 - bottom_frac))
    bottom_labels = np.unique(labels[y0:H, :])
    bottom_labels = bottom_labels[bottom_labels != 0]

    if len(bottom_labels) == 0:
        return mask

    best = max(bottom_labels, key=lambda l: stats[l, cv2.CC_STAT_AREA])

    out = np.zeros_like(mask)
    out[labels == best] = 255
    return out


def on_field_window(mask, bx, by, radius=5):
    """True se in una finestra attorno ai piedi c'è almeno un pixel di prato."""
    H, W = mask.shape[:2]
    x1 = max(0, bx - radius)
    x2 = min(W, bx + radius + 1)
    y1 = max(0, by - radius)
    y2 = min(H, by + radius + 1)
    return mask[y1:y2, x1:x2].any()


def apply_preset(args):
    """
    Preset:
      - general: robusto (SNMOT-060)
      - hard: come general + edge-margin (SNMOT-166)
    I parametri numerici con default None vengono riempiti dal preset.
    """
    if args.preset is None:
        return args

    if args.preset == "general":
        args.field_filter = True
        args.pitch_component = True
        if args.pitch_bottom_frac is None:
            args.pitch_bottom_frac = 0.18
        if args.mask_dilate is None:
            args.mask_dilate = 7
        if args.foot_radius is None:
            args.foot_radius = 5
        if args.edge_margin is None:
            args.edge_margin = 0.0
        if args.big_h_frac is None:
            args.big_h_frac = 0.18

    elif args.preset == "hard":
        args.field_filter = True
        args.pitch_component = True
        if args.pitch_bottom_frac is None:
            args.pitch_bottom_frac = 0.18
        if args.mask_dilate is None:
            args.mask_dilate = 9
        if args.foot_radius is None:
            args.foot_radius = 5
        if args.edge_margin is None:
            args.edge_margin = 8.0
        if args.big_h_frac is None:
            args.big_h_frac = 0.16

    return args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="Cartella img1/ con frame .jpg")
    ap.add_argument("--out", default="outputs/detections/det.txt", help="File output detections (MOT-like)")
    ap.add_argument("--weights", default="yolo11x.pt")
    ap.add_argument("--conf", type=float, default=0.05, help="Soglia minima YOLO (usa 0.05 per raw)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU (Ultralytics)")
    ap.add_argument("--save-vis", type=int, default=30, help="Salva N frame con bbox (0 = no)")

    # Speed / GPU
    ap.add_argument("--device", default=None, help="0 per GPU, 'cpu' per CPU, None=auto")
    ap.add_argument("--half", action="store_true", help="FP16 su GPU (più veloce)")

    # Preset
    ap.add_argument("--preset", choices=["general", "hard"], default=None,
                    help="general (robusto, consigliato per SNMOT-060) | hard (aggiunge edge-margin)")

    # --- FILTRO CAMPO ---
    ap.add_argument("--field-filter", action="store_true",
                    help="Tieni solo box con 'piedi' sul prato (verde)")
    ap.add_argument("--pitch-component", action="store_true",
                    help="Tieni solo la componente del prato che tocca la fascia bassa (anti verde sugli spalti)")
    ap.add_argument("--pitch-bottom-frac", type=float, default=None,
                    help="Frazione (0-1) fascia bassa per pitch-component (es 0.18).")
    ap.add_argument("--mask-dilate", type=int, default=None,
                    help="Kernel (odd) dilatazione mask prato (es 7/9). None = preset.")
    ap.add_argument("--foot-radius", type=int, default=None,
                    help="Raggio finestra (px) intorno ai piedi (es 5). None = preset.")

    # HSV (default permissivi)
    ap.add_argument("--h-lo", type=int, default=20)
    ap.add_argument("--h-hi", type=int, default=110)
    ap.add_argument("--s-lo", type=int, default=25)
    ap.add_argument("--v-lo", type=int, default=20)

    # --- staff vicino linea: edge-margin SOLO box grandi ---
    ap.add_argument("--edge-margin", type=float, default=None,
                    help="(px) vicino bordo prato: scarta SOLO box grandi. None = preset.")
    ap.add_argument("--big-h-frac", type=float, default=None,
                    help="box 'grande' se h > big_h_frac * H. None = preset.")

    # --- FASCE BORDI IMMAGINE (opzionale) ---
    ap.add_argument("--border-bottom", type=int, default=0,
                    help="(px) scarta box con piedi negli ultimi N px in basso (0=off)")
    ap.add_argument("--border-left", type=int, default=0,
                    help="(px) scarta box con piedi entro N px dal bordo sinistro (0=off)")
    ap.add_argument("--border-right", type=int, default=0,
                    help="(px) scarta box con piedi entro N px dal bordo destro (0=off)")

    args = ap.parse_args()
    args = apply_preset(args)

    # fallback se non preset e non passati
    if args.pitch_bottom_frac is None:
        args.pitch_bottom_frac = 0.18
    if args.mask_dilate is None:
        args.mask_dilate = 7
    if args.foot_radius is None:
        args.foot_radius = 5
    if args.edge_margin is None:
        args.edge_margin = 0.0
    if args.big_h_frac is None:
        args.big_h_frac = 0.18

    frames_dir = Path(args.frames)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # reset file
    if out_path.exists():
        out_path.unlink()

    vis_dir = out_path.parent / f"vis_{out_path.stem}"
    if args.save_vis > 0:
        vis_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)

    results = model.predict(
        source=str(frames_dir),
        stream=True,              # tienilo True (stabile in memoria)
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        classes=[0],              # person
        device=args.device,       # es: 0
        half=args.half,           # FP16
        verbose=False,
    )


    n_written = 0

    with out_path.open("a", encoding="utf-8") as f_out:
        for frame_idx, r in enumerate(results, start=1):
            im = r.orig_img  # BGR
            H, W = im.shape[:2]

            if r.boxes is None or r.boxes.xyxy is None or len(r.boxes) == 0:
                continue

            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()

            mask_keep = None  # dilatata per "tenere"
            dist = None       # su mask_edge (non dilatata) per edge-margin

            if args.field_filter:
                # mask reale
                mask_edge = field_mask_hsv(im, args.h_lo, args.h_hi, args.s_lo, args.v_lo)

                # elimina verdi sugli spalti/pubblicità
                if args.pitch_component:
                    mask_edge = keep_pitch_component(mask_edge, bottom_frac=args.pitch_bottom_frac)

                # mask tollerante
                mask_keep = mask_edge.copy()
                if args.mask_dilate and args.mask_dilate > 0:
                    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.mask_dilate, args.mask_dilate))
                    mask_keep = cv2.dilate(mask_keep, k, iterations=1)

                if args.edge_margin and args.edge_margin > 0:
                    dist = cv2.distanceTransform(mask_edge, distanceType=cv2.DIST_L2, maskSize=5)

            kept = []

            for (x1, y1, x2, y2), c in zip(xyxy, confs):
                # piedi
                bx = int(round((x1 + x2) * 0.5))
                by = int(round(y2))
                bx = max(0, min(W - 1, bx))
                by = max(0, min(H - 1, by))

                # fasce bordi
                if args.border_bottom > 0 and by >= (H - args.border_bottom):
                    continue
                if args.border_left > 0 and bx <= args.border_left:
                    continue
                if args.border_right > 0 and bx >= (W - args.border_right):
                    continue

                # filtro prato
                if mask_keep is not None:
                    if not on_field_window(mask_keep, bx, by, radius=args.foot_radius):
                        continue

                    # edge-margin: scarta solo box grandi vicino al bordo
                    if dist is not None and args.edge_margin > 0:
                        h_pix = float(y2 - y1)
                        is_big = h_pix > (args.big_h_frac * H)
                        if is_big and dist[by, bx] < args.edge_margin:
                            continue

                # output MOT-like det: frame,-1,x,y,w,h,conf,-1,-1,-1
                x = int(round(x1))
                y = int(round(y1))
                w = int(round(x2 - x1))
                h = int(round(y2 - y1))
                f_out.write(f"{frame_idx},-1,{x},{y},{w},{h},{float(c):.6f},-1,-1,-1\n")
                n_written += 1
                kept.append((x1, y1, x2, y2, c))

            # vis
            if args.save_vis > 0 and frame_idx <= args.save_vis:
                vis = im.copy()
                for (x1, y1, x2, y2, c) in kept:
                    cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(vis, f"{c:.2f}", (int(x1), int(y1) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imwrite(str(vis_dir / f"{frame_idx:06d}.jpg"), vis)

    print(f"[OK] detections written: {n_written}")
    print(f"[OK] det file: {out_path.resolve()}")
    if args.save_vis > 0:
        print(f"[OK] vis dir:  {vis_dir.resolve()}")


if __name__ == "__main__":
    main()
