# scripts/eval_hota05.py
import argparse
from pathlib import Path
import numpy as np

# Compat NumPy 2.x: TrackEval usa alias deprecati
if not hasattr(np, "float"): np.float = float
if not hasattr(np, "int"): np.int = int
if not hasattr(np, "bool"): np.bool = bool

import trackeval


DEFAULT_HOTA_ALPHAS = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
TARGET_ALPHA = 0.50


def write_seqmap(path: Path, seqs: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("name\n")
        for s in seqs:
            f.write(f"{s}\n")


def ensure_mot10cols(src: Path, dst: Path) -> None:
    """
    TrackEval (MOTChallenge2DBox) legge i risultati in formato MOT.
    Accetta bene il classico: frame,id,x,y,w,h,conf,-1,-1,-1
    Se il tuo file ha 6 o 7 colonne, lo portiamo a 10.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    out_lines = []

    for line in src.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        cols = [c.strip() for c in line.split(",")]
        if len(cols) == 10:
            out_lines.append(",".join(cols))
        elif len(cols) == 7:
            # frame,id,x,y,w,h,conf  -> aggiungi -1,-1,-1
            out_lines.append(",".join(cols + ["-1", "-1", "-1"]))
        elif len(cols) == 6:
            # frame,id,x,y,w,h -> aggiungi conf=1 e -1,-1,-1
            out_lines.append(",".join(cols + ["1", "-1", "-1", "-1"]))
        else:
            raise ValueError(f"Formato non supportato ({len(cols)} colonne): {src}\nEsempio riga: {line}")

    dst.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")


def find_hota_block(obj):
    """Cerca ricorsivamente il dict che contiene 'HOTA', 'DetA', 'AssA'."""
    if isinstance(obj, dict):
        if "HOTA" in obj and "DetA" in obj and "AssA" in obj:
            return obj
        for v in obj.values():
            found = find_hota_block(v)
            if found is not None:
                return found
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-root", required=True,
                    help=r"Cartella split con le sequenze e la GT, es: data\soccernet\tracking-2023\train")
    ap.add_argument("--seq", required=True, help="Nome sequenza, es: SNMOT-166")
    ap.add_argument("--pred", required=True, help="Il tuo file di tracking (outputs/tracking_*.txt)")
    ap.add_argument("--tracker-name", default="mytracker", help="Nome del tracker (cartella TrackEval)")
    ap.add_argument("--workdir", default="outputs/trackeval_work", help="Cartella di lavoro (creata automaticamente)")
    args = ap.parse_args()

    gt_root = Path(args.gt_root)
    if not gt_root.exists():
        raise FileNotFoundError(f"GT root non trovato: {gt_root}")

    pred_src = Path(args.pred)
    if not pred_src.exists():
        raise FileNotFoundError(f"Pred non trovato: {pred_src}")

    workdir = Path(args.workdir)
    seqmap = workdir / "seqmaps" / "custom.txt"
    write_seqmap(seqmap, [args.seq])

    # struttura trackers:
    # TRACKERS_FOLDER/<tracker_name>/data/<seq>.txt   (SKIP_SPLIT_FOL=True)
    trackers_folder = workdir / "trackers"
    pred_dst = trackers_folder / args.tracker_name / "data" / f"{args.seq}.txt"
    ensure_mot10cols(pred_src, pred_dst)

    # --- TrackEval config ---
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config["PRINT_RESULTS"] = True
    eval_config["PRINT_ONLY_COMBINED"] = True

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config["GT_FOLDER"] = str(gt_root)
    dataset_config["TRACKERS_FOLDER"] = str(trackers_folder)
    dataset_config["OUTPUT_FOLDER"] = str(workdir / "results")
    dataset_config["SKIP_SPLIT_FOL"] = True  # IMPORTANTISSIMO: usa GT_FOLDER/<seq>/gt/gt.txt
    dataset_config["SEQMAP_FILE"] = str(seqmap)
    dataset_config["TRACKERS_TO_EVAL"] = [args.tracker_name]
    dataset_config["SPLIT_TO_EVAL"] = "train"     # valore richiesto dalla classe, ma split fol è skipped
    dataset_config["BENCHMARK"] = "MOT17"         # valore richiesto dalla classe, ma split fol è skipped
    dataset_config["DO_PREPROC"] = False

    metrics_config = {"METRICS": ["HOTA"], "THRESHOLD": 0.5}

    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = [trackeval.metrics.HOTA(metrics_config)]

    output_res, _ = evaluator.evaluate(dataset_list, metrics_list)

    hota_block = find_hota_block(output_res)
    if hota_block is None:
        raise RuntimeError("Non sono riuscito a trovare i risultati HOTA dentro output_res (struttura inattesa).")

    # Estrai arrays e prendi alpha=0.5
    hota_arr = np.array(hota_block["HOTA"], dtype=float).reshape(-1)
    deta_arr = np.array(hota_block["DetA"], dtype=float).reshape(-1)
    assa_arr = np.array(hota_block["AssA"], dtype=float).reshape(-1)

    if len(hota_arr) == 1:
        # caso: già configurato a singola alpha
        hota05, deta05, assa05 = float(hota_arr[0]), float(deta_arr[0]), float(assa_arr[0])
    else:
        try:
            idx = DEFAULT_HOTA_ALPHAS.index(TARGET_ALPHA)
        except ValueError:
            raise RuntimeError(f"TARGET_ALPHA={TARGET_ALPHA} non presente in DEFAULT_HOTA_ALPHAS={DEFAULT_HOTA_ALPHAS}")
        hota05, deta05, assa05 = float(hota_arr[idx]), float(deta_arr[idx]), float(assa_arr[idx])

    print("\n====================")
    print(f"SEQ: {args.seq}")
    print(f"HOTA_0.5 = {hota05:.6f}")
    print(f"DetA_0.5 = {deta05:.6f}")
    print(f"AssA_0.5 = {assa05:.6f}")
    print("====================\n")


if __name__ == "__main__":
    main()
