import os, re, json, argparse
from collections import defaultdict
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F

# ---------- Basics ----------
def load_E_from_pt(pt_path: str) -> np.ndarray:
    obj = torch.load(pt_path, map_location='cpu')
    x = obj['attr_norm'] if isinstance(obj, dict) and 'attr_norm' in obj else obj
    x = torch.as_tensor(x)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x.squeeze(0)
    x = x.detach().cpu().numpy().astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    return (x - mn) / (mx - mn) if mx - mn > 1e-12 else np.zeros_like(x, dtype=np.float32)


def resize_to_grid(E: np.ndarray, grid_size: int) -> np.ndarray:
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    H, W = E.shape
    if H == grid_size and W == grid_size:
        return E
    t = torch.from_numpy(E).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(grid_size, grid_size), mode='bilinear', align_corners=False)
    return t.squeeze().numpy()


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)) + 1e-8
    nb = float(np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / (na * nb))


def trapezoid_auc(xs: np.ndarray, ys: np.ndarray) -> float:
    xs = xs.astype(np.float64)
    ys = ys.astype(np.float64)
    rng = xs.max() - xs.min()
    if rng <= 0:
        return 0.0
    xn = (xs - xs.min()) / rng
    return float(np.trapz(ys, xn))


# ---------- Build same-video near-frame pairs using pt2orig.json ----------
def build_same_video_pairs(pt_dir: str,
                           pt2orig: Dict[str, str],
                           max_dt: int) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    
    fake_groups = defaultdict(list)
    real_groups = defaultdict(list)

    for img_path, orig_path in tqdm(pt2orig.items(), desc="grouping by video", leave=False):
       
        norm_orig = orig_path
        if isinstance(orig_path, str):
            if orig_path.strip().lower() in {"", "null", "none"}:
                norm_orig = None
        is_fake = norm_orig is not None

        pt_name = img_path.replace('/', '_').replace('\\', '_') + ".pt"
        pt_path = os.path.join(pt_dir, pt_name)
        if not os.path.isfile(pt_path):
            continue

        group_id = os.path.dirname(img_path)

        fname = os.path.basename(img_path)
        stem, _ = os.path.splitext(fname)
        try:
            frame_idx = int(stem)
        except ValueError:
            continue

        groups_dict = fake_groups if is_fake else real_groups
        groups_dict[group_id].append((frame_idx, pt_path))

    def build_pairs(groups: Dict[str, List[Tuple[int, str]]]) -> List[Tuple[str, str]]:
        pairs = []
        for _, lst in groups.items():
            lst.sort(key=lambda x: x[0])
            n = len(lst)
            for i in range(n):
                for dt in range(1, max_dt + 1):
                    j = i + dt
                    if j >= n:
                        break
                    pairs.append((lst[i][1], lst[j][1]))
        return pairs

    return build_pairs(fake_groups), build_pairs(real_groups)


# ---------- Compute IoU for multiple ks using “rank indices” in one pass ----------
def iou_over_ks_from_sorted_indices(flat_a: np.ndarray,
                                    flat_b: np.ndarray,
                                    idx_a_desc: np.ndarray,
                                    idx_b_desc: np.ndarray,
                                    nks: np.ndarray) -> List[float]:

    ious = []
    for nk in nks:
        if nk <= 0:
            ious.append(0.0)
            continue
        sel_a = idx_a_desc[:nk]
        sel_b = idx_b_desc[:nk]
        inter = np.intersect1d(sel_a, sel_b, assume_unique=False, return_indices=False).size
        denom = 2 * nk - inter
        ious.append(float(inter / max(1, denom)))
    return ious


# ---------- Compute Continuity ----------
def continuity_stats_on_pairs(pairs: List[Tuple[str, str]],
                              grid_size: int,
                              ks: List[int]) -> Dict[str, float]:
    cos_vals = []
    # Pre-convert ks into pixel counts (based on grid_size^2)
    Npix = grid_size * grid_size
    nks = np.array([max(1, int(round(Npix * k / 100.0))) for k in ks], dtype=np.int64)
    iou_buffer = [[] for _ in ks]  # same order as ks

    for pa, pb in tqdm(pairs, desc=f"pairs[{len(pairs)}]", leave=False):
        Ea = load_E_from_pt(pa)
        Eb = load_E_from_pt(pb)
        Va = resize_to_grid(Ea, grid_size)
        Vb = resize_to_grid(Eb, grid_size)

        # Single cosine computation
        va = Va.ravel()
        vb = Vb.ravel()
        cos_vals.append(cos_sim(va, vb))

        # Sort once per image (descending indices)
        idx_a = np.argsort(va)[::-1]
        idx_b = np.argsort(vb)[::-1]

        # Use the indices to compute IoU for all ks
        ious = iou_over_ks_from_sorted_indices(va, vb, idx_a, idx_b, nks)
        for j, iou in enumerate(ious):
            iou_buffer[j].append(iou)

    stats = {
        "N_pairs": len(pairs),
        "cosine_mean": float(np.mean(cos_vals)) if cos_vals else np.nan,
        "cosine_std": float(np.std(cos_vals)) if cos_vals else np.nan,
    }
    iou_means = []
    for k, vals in zip(ks, iou_buffer):
        m = float(np.mean(vals)) if vals else np.nan
        stats[f"IoU@{k}"] = m
        iou_means.append(m)
    stats["IoU_AUC_over_k"] = trapezoid_auc(
        np.array(ks, dtype=np.float64),
        np.array(iou_means, dtype=np.float64)
    )
    return stats


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--methods',
        type=str,
        nargs='+',
        required=True,
        help='e.g., GradCAM IG LRP random_baseline'
    )
    ap.add_argument('--dataset', type=str, default='Celeb-DF-v2')
    ap.add_argument('--base_dir', type=str, default='training/XAI/output')
    ap.add_argument('--grid_size', type=int, default=256)
    ap.add_argument('--ks', type=str, default='5,10,20,30,40,50')
    ap.add_argument('--max_dt', type=int, default=1, help='same-video near frames within Δt')
    args = ap.parse_args()

    ks = [int(s) for s in args.ks.split(',') if s.strip()]
    grid = int(args.grid_size)

    json_path = os.path.join(args.base_dir, 'json', args.dataset, 'pt2orig.json')
    assert os.path.exists(json_path), f"Missing mapping json: {json_path}"
    with open(json_path, 'r') as f:
        pt2orig = json.load(f)

    save_dir = os.path.join(args.base_dir, 'eval_csv')
    os.makedirs(save_dir, exist_ok=True)

    for method in args.methods:
        print(f"\n[Exp4] Method={method} | dataset={args.dataset} | grid={grid} | ks={ks} | Δt<={args.max_dt}")
        pt_dir = os.path.join(args.base_dir, method, args.dataset)
        assert os.path.isdir(pt_dir), f"Attribution dir not found: {pt_dir}"

        pairs_fake, pairs_real = build_same_video_pairs(pt_dir, pt2orig, args.max_dt)
        res_fake = continuity_stats_on_pairs(pairs_fake, grid, ks)
        res_real = continuity_stats_on_pairs(pairs_real, grid, ks)

        rows = [
            {
                'method': method,
                'dataset': args.dataset,
                'layer': 'Fake',
                'pair_type': f'same_video_Δt<={args.max_dt}',
                'grid': grid,
                'ks': ",".join(map(str, ks)),
                **res_fake
            },
            {
                'method': method,
                'dataset': args.dataset,
                'layer': 'Real',
                'pair_type': f'same_video_Δt<={args.max_dt}',
                'grid': grid,
                'ks': ",".join(map(str, ks)),
                **res_real
            },
        ]
        df = pd.DataFrame(rows)
        out_csv = os.path.join(save_dir, f"{method}_exp4.csv")
        df.to_csv(out_csv, index=False)
        print(f"[Exp4:{method}] saved → {out_csv}")
        show_cols = [
            c for c in [
                'method',
                'layer',
                'pair_type',
                'N_pairs',
                'cosine_mean',
                'IoU@5',
                'IoU@10',
                'IoU@20',
                'IoU_AUC_over_k'
            ]
            if c in df.columns
        ]
        print(df[show_cols].to_string(index=False))


if __name__ == '__main__':
    main()
