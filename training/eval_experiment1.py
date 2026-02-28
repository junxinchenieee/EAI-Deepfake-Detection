import os
import torch
import numpy as np
import json
import glob
from tqdm import tqdm
import pandas as pd
import argparse
from PIL import Image
import cv2

def normalize_expl(expl, mode, mask) :
    # Normalize explanation map: decide whether to match the total sum of the mask
    x = expl.astype(np.float32)
    if mode == 'none':
        return x
    if mode == 'match_mask':  
        assert mask is not None, "mask is required for match_mask"
        x = np.maximum(x, 0.0)           # Keep only non-negative contributions
        s = float(x.sum())
        m = float(mask.sum())
        if s < 1e-12 or m < 1e-12:\
            return np.zeros_like(x)      # Return all-zero map in extreme cases
        return x / s * m                 # Scale to match the total mask sum
    raise ValueError(f"Unknown norm mode: {mode}")

def compute_metrics_extended(explanation_map, explanation_map_norm, mask):
    # Compute extended metrics: IoU / Precision / Recall and Top-k soft IoU
    mask = mask.astype(np.float32)
    explanation_map = explanation_map.astype(np.float32)
    explanation_map_norm = np.maximum(explanation_map_norm.astype(np.float32), 0.0)

    # IoU (using normalized explanation map)
    intersection_norm = float((explanation_map_norm * mask).sum())
    union = float(explanation_map_norm.sum() + mask.sum() - intersection_norm)
    iou = float(intersection_norm / union) if union > 0 else 0.0

    # Precision / Recall (based on raw explanation map intensities)
    intersection = float((explanation_map * mask).sum())
    precision = float(intersection / (float(explanation_map.sum()) + 1e-8))
    recall = float(intersection / (float(mask.sum()) + 1e-8))

    def compute_topk_iou_soft(k):
        # Soft Top-k IoU: select top-k% highest pixels (not binarized) and compute IoU with mask
        arr = explanation_map_norm
        H, W = arr.shape
        N = H * W
        K = max(1, int(round(k / 100.0 * N)))

        flat = arr.reshape(-1)
        idx = np.argpartition(flat, -K)[-K:]  # Approximate selection of top-K indices
        topk_vals = np.zeros_like(flat, dtype=np.float32)
        topk_vals[idx] = flat[idx]
        topk_map_soft = topk_vals.reshape(H, W)

        inter = float((topk_map_soft * mask).sum())
        uni = float(topk_map_soft.sum() + mask.sum() - inter)
        return float(inter / uni) if uni > 0 else 0.0

    top10_iou = compute_topk_iou_soft(10)
    top25_iou = compute_topk_iou_soft(25)
    top40_iou = compute_topk_iou_soft(40)
    top55_iou = compute_topk_iou_soft(55)
    top70_iou = compute_topk_iou_soft(70)

    return iou, precision, recall, top10_iou, top25_iou, top40_iou, top55_iou, top70_iou

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate explanation maps against masks.")
    parser.add_argument('--methods', type=str, nargs='+', required=True,
                        help='List of explainability methods to evaluate')
    parser.add_argument('--dataset', type=str, default='Celeb-DF-v2',
                        help='Name of the dataset folder')
    parser.add_argument('--norm', type=str, default='match_mask',
                        choices=['match_mask', 'none'],
                        help='Normalization for explanation maps')
    parser.add_argument('--output_dir', type=str,
                        default="training/XAI/output",
                        help='Path to output root (contains json/, <method>/, eval_csv/)')

    args = parser.parse_args()

    dataset = args.dataset            # Dataset name (for path concatenation)
    methods = args.methods            # List of explanation methods to evaluate
    norm_mode = args.norm             # Normalization mode

    base_dir = os.path.abspath(args.output_dir)
    json_path = os.path.join(base_dir, 'json', dataset, 'pt2mask.json')  # Mapping file: PT to mask

    summary = []  # Summary statistics across methods

    for method in methods:
        print(f"[{method}] evaluating... (norm={norm_mode})")
        pt_dir = os.path.join(base_dir, method, dataset)                 # Directory of explanation maps (.pt)
        save_csv = os.path.join(base_dir, 'eval_csv',dataset, f'{method}_exp1.csv')  # Per-method evaluation CSV

        with open(json_path, 'r') as f:
            pt2mask = json.load(f)  # Load mapping from sample to mask path

        all_pt_paths = glob.glob(os.path.join(pt_dir, '*.pt'))
        total_files = len(all_pt_paths)
        records = []   # Store per-sample metrics
        success = 0
        skipped = 0    # Count skipped samples (missing mapping / read error / shape mismatch)

        for pt_path in tqdm(all_pt_paths, desc=method):
            pt_name = os.path.basename(pt_path)
            if pt_name not in pt2mask:
                skipped += 1
                continue

            mask_path = pt2mask[pt_name]
            if mask_path is None or not os.path.exists(mask_path):
                skipped += 1
                continue

            try:
                explanation_map = torch.load(pt_path).detach().cpu().numpy()  # Load explanation map
            except:
                skipped += 1
                continue

            try:
                # Load and binarize mask, apply dilation to tolerate boundary errors
                mask = np.array(Image.open(mask_path).convert('L')) / 255.0
                mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
            except:
                skipped += 1
                continue

            if mask.shape != explanation_map.shape:
                # Resize explanation map to match mask size using bilinear interpolation
                explanation_map = torch.nn.functional.interpolate(
                    torch.tensor(explanation_map[None, None, ...]),
                    size=mask.shape, mode='bilinear', align_corners=False
                ).squeeze().numpy()

            # Normalize explanation map 
            explanation_map_norm = normalize_expl(explanation_map, norm_mode, mask)

            # Compute evaluation metrics
            iou, precision, recall, top10, top25, top40, top55, top70 = compute_metrics_extended(explanation_map, explanation_map_norm, mask)
            records.append({
                'image': pt_name,
                'IoU': iou,
                'Precision': precision,
                'Recall': recall,
                'Top10_IoU': top10,
                'Top25_IoU': top25,
                'Top40_IoU': top40,
                'Top55_IoU': top55,
                'Top70_IoU': top70,
            })
            success += 1

        # Save per-method detailed results
        df = pd.DataFrame(records)
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
        df.to_csv(save_csv, index=False)

        # Summary statistics
        summary.append({
            'Method': method,
            'Total': total_files,
            'Evaluated': success,
            'Skipped': skipped,
            'Precision': round(df['Precision'].mean(), 4) if success else 0.0,
            'Recall': round(df['Recall'].mean(), 4) if success else 0.0,
            'IoU': round(df['IoU'].mean(), 4) if success else 0.0,
            'Top10_IoU': round(df['Top10_IoU'].mean(), 4) if success else 0.0,
            'Top25_IoU': round(df['Top25_IoU'].mean(), 4) if success else 0.0,
            'Top40_IoU': round(df['Top40_IoU'].mean(), 4) if success else 0.0,
            'Top55_IoU': round(df['Top55_IoU'].mean(), 4) if success else 0.0,
            'Top70_IoU': round(df['Top70_IoU'].mean(), 4) if success else 0.0,
        })

    # Print comparison table to console
    print("\n========== [Mask Alignment] ==========")
    print(pd.DataFrame(summary).to_string(index=False))
    print("========================================")
