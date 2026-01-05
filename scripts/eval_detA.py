# scripts/eval_detA.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def iou_xywh(a, b):
    # a,b: [x,y,w,h]
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True, help="det.txt (frame,-1,x,y,w,h,conf,...)")
    ap.add_argument("--gt", required=True, help="gt.txt (frame,id,x,y,w,h,...)")
    ap.add_argument("--alpha", type=float, default=0.5, help="IoU threshold (default 0.5)")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold for dets")
    args = ap.parse_args()

    det_path = Path(args.det)
    gt_path = Path(args.gt)

    det = pd.read_csv(det_path, header=None)
    # det: frame,id,x,y,w,h,conf,...
    det = det.iloc[:, :7]
    det.columns = ["frame", "id", "x", "y", "w", "h", "conf"]
    det = det[det["conf"] >= args.conf].copy()

    gt = pd.read_csv(gt_path, header=None)
    # gt: frame,id,x,y,w,h, ... (MOT usually has more cols)
    gt = gt.iloc[:, :6]
    gt.columns = ["frame", "id", "x", "y", "w", "h"]

    frames = sorted(set(gt["frame"].unique()).union(det["frame"].unique()))

    TP = FP = FN = 0

    for fr in frames:
        g = gt[gt["frame"] == fr][["x", "y", "w", "h"]].to_numpy(dtype=float)
        d = det[det["frame"] == fr][["x", "y", "w", "h"]].to_numpy(dtype=float)

        if len(g) == 0:
            FP += len(d)
            continue
        if len(d) == 0:
            FN += len(g)
            continue

        # cost matrix for Hungarian: minimize (1 - IoU), block pairs < alpha
        cost = np.ones((len(g), len(d)), dtype=float) * 1e6
        for i in range(len(g)):
            for j in range(len(d)):
                val = iou_xywh(g[i], d[j])
                if val >= args.alpha:
                    cost[i, j] = 1.0 - val

        row_ind, col_ind = linear_sum_assignment(cost)
        matched = 0
        for i, j in zip(row_ind, col_ind):
            if cost[i, j] < 1e5:  # valid match
                matched += 1

        TP += matched
        FP += (len(d) - matched)
        FN += (len(g) - matched)

    det_re = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    det_pr = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    det_a  = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0  # DetAα (Jaccard)

    print(f"alpha={args.alpha:.2f} conf>={args.conf:.2f}")
    print(f"TP={TP} FP={FP} FN={FN}")
    print(f"DetRe={det_re:.4f}  DetPr={det_pr:.4f}  DetA={det_a:.4f}")


if __name__ == "__main__":
    main()