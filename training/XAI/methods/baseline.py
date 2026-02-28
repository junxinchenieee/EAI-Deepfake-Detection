import os
import json
import argparse
from pathlib import Path
import numpy as np
import cv2
import torch
from tqdm import tqdm

def fname_from_path(p: str) -> str:
    return p.replace("/", "_").replace("\\", "_")

def save_pt(arr2d: np.ndarray, save_path: str):
    x = arr2d.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    x = (x - mn) / (mx - mn + 1e-8) if mx > mn else np.zeros_like(x, np.float32)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(x), save_path)

def make_grid(size: int = 256):
    y = (np.arange(size) + 0.5) / size
    x = (np.arange(size) + 0.5) / size
    return np.meshgrid(x, y)  # X, Y in [0,1]

def gaussian_blob(X: np.ndarray, Y: np.ndarray, cx: float, cy: float, sigma: float) -> np.ndarray:
    r2 = (X - cx) ** 2 + (Y - cy) ** 2
    return np.exp(-r2 / (2 * sigma * sigma)).astype(np.float32)

def main():
    ap = argparse.ArgumentParser(description="Baseline-RandGaussSample: sparse random Gaussian sampling.")
    ap.add_argument("--dataset", default="FF-DF")
    ap.add_argument("--target_k", type=float, default=0.05, help="Expected active ratio per image (0.01~0.10 typical)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility")
    ap.add_argument("--mapping_json", default=None,
        help="Use keys (test image paths) from pt2orig.json; default: training/XAI/output/json/<dataset>/pt2orig.json")
    ap.add_argument("--out_root", default="./output",
        help="Save to <out_root>/Baseline/<dataset>/*.pt")
    args = ap.parse_args()

    if not (0.0 < args.target_k < 1.0):
        raise ValueError("--target_k must be in (0,1)")

    base_xai = Path(__file__).resolve().parents[1] 
    if args.mapping_json is None:
        mapping_json = base_xai / "output" / "json" / args.dataset / "pt2orig.json"
    else:
        p = Path(args.mapping_json)
        if not p.is_absolute():
            p = base_xai / p
        mapping_json = p if p.is_file() else (p / args.dataset / "pt2orig.json")
    if not mapping_json.exists():
        raise FileNotFoundError(f"Mapping not found: {mapping_json}")

    with open(mapping_json, "r") as f:
        test2orig = json.load(f)
    test_paths = list(test2orig.keys())

    method_name = "baseline"
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = (base_xai / out_root).resolve()  
    out_dir = out_root / method_name / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    size = 256
    X, Y = make_grid(size)
    rng = np.random.default_rng(args.seed)

    saved = 0
    for img_path in tqdm(test_paths, desc=method_name):
        n_blobs = int(rng.integers(1, 4))  # {1,2,3}
        base = np.zeros((size, size), dtype=np.float32)
        for _ in range(n_blobs):
            cx = float(rng.uniform(0.05, 0.95))
            cy = float(rng.uniform(0.05, 0.95))
            sigma = float(rng.uniform(0.03, 0.08))
            w = float(rng.uniform(0.7, 1.3))
            base += w * gaussian_blob(X, Y, cx, cy, sigma)

        base = base / (base.max() + 1e-8)
        m = float(base.mean())
        alpha = args.target_k / max(m, 1e-8)
        prob = np.clip(alpha * base, 0.0, 1.0).astype(np.float32)

        samp = (rng.random(prob.shape) < prob).astype(np.float32)   # sparse 0/1 map
        heat = cv2.GaussianBlur(samp, (3, 3), sigmaX=0)             # soften a bit
        pt_name = fname_from_path(img_path) + ".pt"
        save_pt(heat, str(out_dir / pt_name))
        saved += 1

    print(f"[{method_name}] saved {saved} .pt to {out_dir} (shape=(256,256), dtype=float32)")
    print(f"Expected active ratio ≈ {args.target_k:.3f} (varies per image due to sampling)")

if __name__ == "__main__":
    main()
