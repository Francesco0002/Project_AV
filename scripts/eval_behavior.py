# eval_behavior.py
from pathlib import Path
import argparse

def load_behavior(path: Path):
    """
    behavior file: frame_id,region_id,n_players
    ritorna dict[(frame, region)] = n_players
    """
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split(",")
            if len(p) < 3:
                continue
            fr = int(float(p[0]))
            rg = int(float(p[1]))
            n  = int(float(p[2]))
            out[(fr, rg)] = n
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="behavior_K_XX.txt predetto")
    ap.add_argument("--gt", required=True, help="behavior GT")
    args = ap.parse_args()

    pred = load_behavior(Path(args.pred))
    gt   = load_behavior(Path(args.gt))

    keys = sorted(gt.keys())  # valutiamo su ciò che esiste in GT
    if not keys:
        raise RuntimeError("GT vuoto o non parsabile")

    abs_sum = 0.0
    for k in keys:
        g = gt[k]
        p = pred.get(k, 0)  # se manca, contalo 0
        abs_sum += abs(p - g)

    mae = abs_sum / len(keys)
    nmae = (10.0 - min(10.0, mae)) / 10.0

    print(f"MAE = {mae:.6f}")
    print(f"nMAE = {nmae:.6f}")

if __name__ == "__main__":
    main()

#Esempio run
#python eval_behavior.py --pred outputs/behavior_1_01.txt --gt GT/behavior_1_01.txt