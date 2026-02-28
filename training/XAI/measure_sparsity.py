import os
import glob
import argparse
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

def load_pt(path):
    x = torch.load(path, map_location="cpu")
    x = x.detach().cpu().numpy().astype(np.float32)
    if x.ndim != 2:
        raise ValueError(f"Heatmap must be 2D. Got shape={x.shape} in {path}")
    return x

def effective_activation_ratio(x, eps=1e-12):
    """
    Threshold-free sparsity proxy:
    eff = (sum x)^2 / (N * sum x^2), in [0,1].
    Smaller -> sparser; Larger -> more spread.
    """
    x = np.maximum(x, 0.0).astype(np.float64)
    N = x.size
    s1 = x.sum()
    s2 = (x * x).sum()
    if s2 < eps or N == 0:
        return 0.0
    return float((s1 * s1) / (N * s2))

def binary_active_ratio(x, tau):
    """Fraction of pixels above absolute threshold tau."""
    return float((x > tau).mean())

def main():
    ap = argparse.ArgumentParser(description="Measure average sparsity of XAI heatmaps (.pt).")
    ap.add_argument("--dataset", required=True, help="e.g., FF-DF")
    ap.add_argument("--methods", nargs="+", required=True,
                    help="Method folder names under output/, e.g., GradCAM Baseline-CenterSample")
    ap.add_argument("--out_root", default="/mnt/datadisk0/deepfakebench/DeepfakeBench/training/XAI/output",
                    help="Root dir containing <method>/<dataset>/*.pt")
    ap.add_argument("--thresholds", nargs="*", type=float, default=[0.5, 0.7, 0.9],
                    help="Absolute thresholds to compute active ratios")
    ap.add_argument("--save_csv", default="", help="Optional: path to save CSV summary")
    args = ap.parse_args()

    rows = []
    for method in args.methods:
        pt_dir = Path(args.out_root) / method / args.dataset
        paths = sorted(glob.glob(str(pt_dir / "*.pt")))
        if not paths:
            print(f"[WARN] No .pt found for method={method} dataset={args.dataset} at {pt_dir}")
            continue

        eff_list = []
        thr_lists = {tau: [] for tau in args.thresholds}

        for p in tqdm(paths, desc=f"{method}", leave=False):
            x = load_pt(p)
            x = np.clip(x, 0.0, 1.0)

            eff_list.append(effective_activation_ratio(x))
            for tau in args.thresholds:
                thr_lists[tau].append(binary_active_ratio(x, tau))

        # summarize
        rec = {
            "Method": method,
            "Dataset": args.dataset,
            "Files": len(paths),
            "EffActivation_Mean": float(np.mean(eff_list)),
            "EffActivation_Median": float(np.median(eff_list)),
        }
        for tau in args.thresholds:
            vals = np.array(thr_lists[tau], dtype=np.float64)
            rec[f"Active@{tau}_Mean"] = float(vals.mean()) if len(vals) else 0.0
            rec[f"Active@{tau}_Median"] = float(np.median(vals)) if len(vals) else 0.0
        rows.append(rec)

    if not rows:
        print("No results.")
        return

    df = pd.DataFrame(rows)
    # nice column order
    base_cols = ["Method", "Dataset", "Files", "EffActivation_Mean", "EffActivation_Median"]
    thr_cols = []
    for tau in args.thresholds:
        thr_cols += [f"Active@{tau}_Mean", f"Active@{tau}_Median"]
    df = df[base_cols + thr_cols]

    # print summary
    with pd.option_context("display.max_columns", None, "display.width", 150):
        print("\n=== Sparsity Summary ===")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
