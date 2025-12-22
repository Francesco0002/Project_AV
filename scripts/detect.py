# scripts/detect.py
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="Cartella img1/ con frame .jpg")
    ap.add_argument("--out", default="outputs/detections/det.txt", help="File output detections (MOT-like)")
    ap.add_argument("--weights", default="yolo11x.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--save-vis", type=int, default=30, help="Salva N frame con bbox disegnate (0 = no)")
    args = ap.parse_args()

    frames_dir = Path(args.frames)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vis_dir = out_path.parent / "vis"
    if args.save_vis > 0:
        vis_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)

    # Ultralytics può leggere una cartella di immagini come sorgente
    results = model.predict(
        source=str(frames_dir),
        stream=True,
        conf=args.conf,
        imgsz=args.imgsz,
        classes=[0],   # COCO "person"
        verbose=False,
    )

    # MOT-like det format: frame,-1,x,y,w,h,conf,-1,-1,-1
    # (comodo per debug e per alcuni tracker)
    n_written = 0
    for frame_idx, r in enumerate(results, start=1):
        im = r.orig_img
        if r.boxes is None or r.boxes.xyxy is None or len(r.boxes) == 0:
            # nessuna detection in questo frame
            continue

        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        # scriviamo tutte le detections del frame
        with out_path.open("a", encoding="utf-8") as f:
            for (x1, y1, x2, y2), c in zip(xyxy, confs):
                x = int(round(x1))
                y = int(round(y1))
                w = int(round(x2 - x1))
                h = int(round(y2 - y1))
                f.write(f"{frame_idx},-1,{x},{y},{w},{h},{float(c):.6f},-1,-1,-1\n")
                n_written += 1

        # salva visualizzazione per i primi N frame
        if args.save_vis > 0 and frame_idx <= args.save_vis:
            for (x1, y1, x2, y2), c in zip(xyxy, confs):
                cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(im, f"{c:.2f}", (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imwrite(str(vis_dir / f"{frame_idx:06d}.jpg"), im)

    print(f"[OK] detections written: {n_written}")
    print(f"[OK] det file: {out_path.resolve()}")
    if args.save_vis > 0:
        print(f"[OK] vis dir:  {vis_dir.resolve()}")


if __name__ == "__main__":
    main()
