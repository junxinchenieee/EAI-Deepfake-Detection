import os, json, argparse
from typing import Tuple, List, Dict
import math
import pandas as pd
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import yaml
from PIL import Image

from test import prepare_testing_data, init_seed, inference
from detectors import DETECTOR


# ================= Utility Functions =================

def parse_ks(s: str) -> Tuple[float, ...]:
    ks = tuple(float(x) for x in s.split(',') if x.strip())
    return ks

def trapezoid_auc(xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    xs = xs.to(torch.float64); ys = ys.to(torch.float64)
    # Deduplicate adjacent equal xs (avoid zero-width intervals if 0 was prepended twice)
    if xs.numel() >= 2:
        keep = torch.ones_like(xs, dtype=torch.bool)
        keep[1:] = xs[1:] != xs[:-1]
        if not bool(keep.all()):
            xs = xs[keep]; ys = ys[:, keep]
    x0, xN = xs.min(), xs.max()
    assert (xN - x0) > 0, "xs range should be greater than 0"
    xn = (xs - x0) / (xN - x0)
    dx = xn[1:] - xn[:-1]
    area = (dx * (ys[:, :-1] + ys[:, 1:]) * 0.5).sum(dim=1)  # [B]
    return area  # torch.float64 [B]

def normalized_trapezoid_auc(xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    # Normalize each curve by its initial value to enable comparison across methods.
    y0 = ys[:, :1].clamp(min=1e-8)
    ys_norm = ys / y0
    return trapezoid_auc(xs, ys_norm)

def gaussian_kernel_1d(size: int, sigma: float, device, dtype):
    coords = torch.arange(size, dtype=dtype, device=device) - (size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    return g / g.sum()

def gaussian_blur_batch(img_bchw: torch.Tensor, ksize: int = 11, sigma: float = 5.0) -> torch.Tensor:
    assert ksize % 2 == 1 and ksize > 1, "ksize must be an odd number and >1"
    B, C, H, W = img_bchw.shape
    device, dtype = img_bchw.device, img_bchw.dtype

    g1d = gaussian_kernel_1d(ksize, sigma, device, dtype)  # [ksize]
    # Replicate kernel per channel: weight shape [C, 1, kH, kW], matching groups=C (depthwise)
    kx = g1d.view(1, 1, 1, -1).repeat(C, 1, 1, 1)  # horizontal kernel
    ky = g1d.view(1, 1, -1, 1).repeat(C, 1, 1, 1)  # vertical kernel
    pad = ksize // 2

    x = img_bchw.contiguous()
    x = F.conv2d(F.pad(x, (pad, pad, 0, 0), mode='reflect'), kx, groups=C)
    x = F.conv2d(F.pad(x, (0, 0, pad, pad), mode='reflect'), ky, groups=C)
    return x

def occlude_batch(img_bchw: torch.Tensor, mask_b1hw: torch.Tensor,
                  mode: str, ksize: int = 11, sigma: float = 5.0,
                  rep_src_bchw: torch.Tensor = None) -> torch.Tensor:
    """mask=1 means the region to be replaced/occluded. mode:
       - 'mean'   : replace with per-image mean
       - 'blur'   : replace with Gaussian-blurred image
       - 'repair' : replace with rep_src_bchw (typically the pristine/original source)
    """
    assert img_bchw.shape[0] == mask_b1hw.shape[0]
    assert img_bchw.shape[-2:] == mask_b1hw.shape[-2:], "img and mask must have the same size"
    if mode == 'mean':
        mean_val = img_bchw.mean(dim=(2,3), keepdim=True)  # [B,C,1,1]
        return img_bchw * (1 - mask_b1hw) + mean_val * mask_b1hw
    elif mode == 'blur':
        blurred = gaussian_blur_batch(img_bchw, ksize=ksize, sigma=sigma)
        return img_bchw * (1 - mask_b1hw) + blurred * mask_b1hw
    elif mode == 'repair':
        assert rep_src_bchw is not None, "occlusion_mode='repair' requires rep_src_bchw (pristine/original tensor)"
        return img_bchw * (1 - mask_b1hw) + rep_src_bchw * mask_b1hw
    else:
        raise ValueError("occlusion_mode only supports 'mean' | 'blur' | 'repair'")

def batch_topk_mask_exact(attrs_b1hw: torch.Tensor, k_percent: float, bottom: bool = False) -> torch.Tensor:
    # Select the top/bottom k% pixels by attribution; return a 0/1 mask (float).
    B, _, H, W = attrs_b1hw.shape
    N = H * W
    k = max(1, int(round(N * (float(k_percent) / 100.0))))
    flat = attrs_b1hw.reshape(B, -1)  # [B,N]
    idx = torch.topk(flat, k, largest=not bottom, sorted=False).indices  # [B,k]
    mask = torch.zeros_like(flat, dtype=torch.bool)
    arange = torch.arange(B, device=attrs_b1hw.device).unsqueeze(1).expand(B, k)
    mask[arange, idx] = True
    return mask.view(B, 1, H, W).to(attrs_b1hw.dtype)

def candidate_pt_names(img_path: str):
    p = img_path.replace('\\', '/')
    p = p.replace('/', '_') + '.pt'
    return [p]

def load_attr_pt(pt_dir: str, img_paths: list, expect_hw: tuple) -> torch.Tensor:
    H, W = expect_hw
    outs = []
    missing = []
    for img_p in img_paths:
        hit = None
        for fn in candidate_pt_names(img_p):
            fp = os.path.join(pt_dir, fn)
            if os.path.exists(fp):
                hit = fp
                break
        if hit is None:
            missing.append(img_p)
            continue

        obj = torch.load(hit, map_location='cpu')
        x = obj['attr_norm'] if isinstance(obj, dict) and 'attr_norm' in obj else torch.as_tensor(obj)
        if x.ndim == 2: x = x.unsqueeze(0)
        assert x.ndim == 3 and x.shape[0] == 1, f"{hit} shape must be [1,H,W], got {tuple(x.shape)}"
        assert tuple(x.shape[-2:]) == (H, W), f"{hit} size mismatch: expected {(H,W)}, got {tuple(x.shape[-2:])}"
        x = x.to(torch.float32)
        xmin, xmax = x.min(), x.max()
        if float(xmax - xmin) > 0:
            x = (x - xmin) / (xmax - xmin)
        outs.append(x)

    if missing:
        raise FileNotFoundError("Missing corresponding .pt files:\n" + "\n".join(f" - {m}" for m in missing[:10]) +
                                (f"\n... total missing {len(missing)}" if len(missing) > 10 else ""))
    return torch.stack(outs, dim=0)  # [B,1,H,W]

def load_fake2orig_json(json_path: str) -> Dict[str, str]:
    with open(json_path, 'r') as f:
        mp = json.load(f)
    return {str(k): str(v) for k, v in mp.items()}

def parse_vec3(s: str) -> List[float]:
    vals = [float(x) for x in s.split(',') if x.strip()]
    assert len(vals) == 3, "expect 3 values like '0.5,0.5,0.5'"
    return vals

def read_image_to_tensor(path: str, size_hw: tuple, mean: List[float], std: List[float], device, dtype) -> torch.Tensor:
    H, W = size_hw
    img = Image.open(path).convert('RGB').resize((W, H), resample=Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0  # HxWx3
    t = torch.from_numpy(arr).permute(2, 0, 1).to(device=device, dtype=dtype)  # [C,H,W]
    mean_t = torch.tensor(mean, device=device, dtype=dtype).view(-1, 1, 1)
    std_t  = torch.tensor(std,  device=device, dtype=dtype).view(-1, 1, 1)
    t = (t - mean_t) / std_t
    return t


# ================= Main =================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector_path', type=str, default='./training/config/detector/xception.yaml')
    ap.add_argument('--test_dataset', type=str, default='Celeb-DF-v2')
    ap.add_argument('--weights_path', type=str, default='./training/weights/xception_best.pth')
    ap.add_argument('--methods', type=str, nargs='+', required=True, help='Support multiple method names')
    ap.add_argument('--output_dir', type=str, default='training/XAI/output')
    ap.add_argument('--ks', type=str, default='0.5,1,1.5,2,2.5,3,3.5,5,10,20,30,40,50,60,70,80,90')
    ap.add_argument('--occlusion_mode', type=str, default='repair',
                    choices=['mean','blur','repair'])
    ap.add_argument('--gauss_ksize', type=int, default=11)
    ap.add_argument('--gauss_sigma', type=float, default=5.0)
    ap.add_argument('--repair_fallback', type=str, default='blur',
                    choices=['blur','mean','self'])
    ap.add_argument('--flip_threshold', type=float, default=0.5)
    ap.add_argument('--use_original_repair', action='store_true')
    ap.add_argument('--orig_map_json', type=str,
                    default='training/XAI/output/json/Celeb-DF-v2/pt2orig.json')
    ap.add_argument('--img_mean', type=str, default='0.5,0.5,0.5')
    ap.add_argument('--img_std',  type=str, default='0.5,0.5,0.5')
    ap.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    # Config & random seed
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/test_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)
    config['test_dataset'] = [args.test_dataset]
    config['weights_path'] = args.weights_path

    init_seed(config)
    device = torch.device(args.device)
    if config.get('cudnn', True): cudnn.benchmark = True

    # Model
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)
    ckpt = torch.load(args.weights_path, map_location=device)
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    print('===> Load model done')

    # DataLoader
    loaders = prepare_testing_data(config)
    loader = loaders[args.test_dataset]
    print('===> Load data done')

    ks = parse_ks(args.ks)
    if args.occlusion_mode == 'blur':
        assert args.gauss_ksize % 2 == 1 and args.gauss_ksize > 1
        assert args.gauss_sigma > 0

    # All image paths (to align with .pt files)
    all_img_paths = loader.dataset.data_dict['image']

    # 创建 eval_csv 和子目录（防止写 CSV 报错）
    eval_root = os.path.join(os.path.abspath(args.output_dir), 'eval_csv')
    os.makedirs(eval_root, exist_ok=True)
    os.makedirs(os.path.join(eval_root, args.test_dataset), exist_ok=True)

    # Whether we need the original image (for repair mode or extra repair curve)
    need_orig = (args.occlusion_mode == 'repair') or args.use_original_repair
    fake2orig = {}
    img_mean = parse_vec3(args.img_mean); img_std = parse_vec3(args.img_std)
    if need_orig:
        assert os.path.isfile(os.path.abspath(args.orig_map_json)), f"Mapping JSON not found: {args.orig_map_json}"
        fake2orig = load_fake2orig_json(args.orig_map_json)
        print(f"===> Loaded mapping: {len(fake2orig)} entries from {args.orig_map_json}")

    summaries: List[Dict] = []

    # ==== Loop over multiple methods ====
    for method in args.methods:
        print(f"\n===> Running Exp2 for method: {method} ====")
        pt_dir = os.path.join(os.path.abspath(args.output_dir), method, args.test_dataset)
        assert os.path.isdir(pt_dir), f"Attribution directory not found: {pt_dir}"
        save_csv = os.path.join(eval_root, args.test_dataset, f'{method}_exp2.csv')

        rows: List[Dict] = []
        idx_base = 0

        for i, batch in tqdm(enumerate(loader), total=len(loader), desc=method):
            imgs = batch['image'].to(device)      # [B,3,H,W]
            B, C, H, W = imgs.shape

            # Binary labels & keep only fake (label==1)
            labels = batch.get('label', None)
            if labels is None:
                raise RuntimeError("Dataset missing label, cannot filter fake samples")
            labels = torch.where(batch['label'] != 0, 1, 0).to(device)  # [B]

            # Current batch file names (must match DataLoader order)
            img_batch_paths = all_img_paths[idx_base: idx_base + B]
            assert len(img_batch_paths) == B, "Number of file names must match batch size"
            idx_base += B

            # Keep only fake sample indices
            keep_idx = (labels == 1).nonzero(as_tuple=True)[0]
            if keep_idx.numel() == 0:
                continue  # Skip entire batch if no fake images

            img_paths_f = [img_batch_paths[int(ix)] for ix in keep_idx.tolist()]
            imgs_f = imgs[keep_idx]
            labels_f = labels[keep_idx]
            Bf, _, Hf, Wf = imgs_f.shape

            # Load corresponding fake sample attribution maps (strict check), move to GPU
            attrs = load_attr_pt(pt_dir, img_paths_f, expect_hw=(Hf, Wf)).to(device)  # [Bf,1,H,W]

            # Baseline probability (original image, fake-class probability)
            base_dict = {'image': imgs_f, 'label': labels_f}
            prob0 = inference(model, base_dict)['prob'].reshape(-1).to(torch.float64)  # [Bf]
            assert prob0.numel() == Bf

            # ===== 关键逻辑：topk=0 时 prob0 < 0.5 的图片全部跳过 =====
            # 这里按照你的要求，阈值固定写死为 0.5；如果想复用 flip_threshold，可以改成 float(args.flip_threshold)
            base_filter_tau = 0.5
            keep_prob_mask = prob0 >= base_filter_tau  # 只要 prob0 >= 0.5 的

            if keep_prob_mask.sum().item() == 0:
                # 这一批 fake 样本的 prob0 全都 < 0.5，直接跳过这个 batch
                continue

            # 只保留 prob0 >= 0.5 的样本
            imgs_f   = imgs_f[keep_prob_mask]
            labels_f = labels_f[keep_prob_mask]
            attrs    = attrs[keep_prob_mask]
            prob0    = prob0[keep_prob_mask]

            # 同步过滤文件名列表
            keep_list = keep_prob_mask.detach().cpu().tolist()
            img_paths_f = [p for p, keep in zip(img_paths_f, keep_list) if keep]

            # 更新 Bf, Hf, Wf
            Bf, _, Hf, Wf = imgs_f.shape

            # Curves: x-axis (deletion percentage), y-axis (probability)
            xs = torch.tensor([0.0] + list(ks), dtype=torch.float64, device=device)
            pgi_curve = torch.empty(Bf, xs.numel(), dtype=torch.float64, device=device)
            pgu_curve = torch.empty(Bf, xs.numel(), dtype=torch.float64, device=device)
            pgi_curve[:, 0] = prob0
            pgu_curve[:, 0] = prob0

            # Whether to prepare repair sources (for occlusion_mode='repair' or extra repair curve)
            rep_src = None
            orig_used_flags = torch.zeros(Bf, dtype=torch.bool, device=device)
            if need_orig:
                orig_tensors = []
                for p_idx, fake_p in enumerate(img_paths_f):
                    op = fake2orig.get(fake_p)
                    if (op is not None) and os.path.isfile(op):
                        t = read_image_to_tensor(op, (Hf, Wf), img_mean, img_std, device, imgs_f.dtype)
                        orig_tensors.append(t); orig_used_flags[p_idx] = True
                    else:
                        # Placeholder; will be replaced by fallback later
                        orig_tensors.append(torch.zeros(3, Hf, Wf, dtype=imgs_f.dtype, device=device))
                rep_src = torch.stack(orig_tensors, 0)  # [Bf,3,H,W]

                # Fallback for samples without available originals
                miss_mask = (~orig_used_flags)
                if miss_mask.any():
                    if args.repair_fallback == 'blur':
                        fb = gaussian_blur_batch(imgs_f, ksize=args.gauss_ksize, sigma=args.gauss_sigma)
                    elif args.repair_fallback == 'mean':
                        fb = imgs_f.mean(dim=(2,3), keepdim=True).expand_as(imgs_f)
                    else:  # 'self'
                        fb = imgs_f
                    rep_src[miss_mask] = fb[miss_mask]

            # Prepare an additional repair curve (only if occlusion_mode != 'repair')
            make_separate_repair = args.use_original_repair and (args.occlusion_mode != 'repair')
            if make_separate_repair:
                repair_curve = torch.empty(Bf, xs.numel(), dtype=torch.float64, device=device)
                repair_curve[:, 0] = prob0

            # Incremental occlusion (PGI / PGU) (+ optional repair control)
            for j, k in enumerate(ks, start=1):
                # PGI: delete important region
                m_imp = batch_topk_mask_exact(attrs, k, bottom=False)
                imgs_imp = occlude_batch(
                    imgs_f, m_imp, args.occlusion_mode,
                    args.gauss_ksize, args.gauss_sigma,
                    rep_src_bchw=rep_src if args.occlusion_mode == 'repair' else None
                )
                prob_imp = inference(model, {'image': imgs_imp, 'label': labels_f})['prob'].reshape(-1).to(torch.float64)
                pgi_curve[:, j] = prob_imp

                # PGU: delete unimportant region
                m_un = batch_topk_mask_exact(attrs, k, bottom=True)
                imgs_un = occlude_batch(
                    imgs_f, m_un, args.occlusion_mode,
                    args.gauss_ksize, args.gauss_sigma,
                    rep_src_bchw=rep_src if args.occlusion_mode == 'repair' else None
                )
                prob_un = inference(model, {'image': imgs_un, 'label': labels_f})['prob'].reshape(-1).to(torch.float64)
                pgu_curve[:, j] = prob_un

                # Separate repair control curve (only when evaluating with mean/blur and we want to plot it)
                if make_separate_repair:
                    imgs_rep = occlude_batch(
                        imgs_f, m_imp, mode='repair',
                        ksize=args.gauss_ksize, sigma=args.gauss_sigma,
                        rep_src_bchw=rep_src
                    )
                    prob_rep = inference(model, {'image': imgs_rep, 'label': labels_f})['prob'].reshape(-1).to(torch.float64)
                    repair_curve[:, j] = prob_rep

            # ===== AUC =====
            auc_pgi_batch = trapezoid_auc(xs, pgi_curve)                  # [Bf]
            auc_pgu_batch = trapezoid_auc(xs, pgu_curve)                  # [Bf]
            auc_pgi_norm_batch = normalized_trapezoid_auc(xs, pgi_curve)  # [Bf]
            auc_pgu_norm_batch = normalized_trapezoid_auc(xs, pgu_curve)  # [Bf]

            if make_separate_repair:
                auc_rep_batch      = trapezoid_auc(xs, repair_curve)
                auc_rep_norm_batch = normalized_trapezoid_auc(xs, repair_curve)
            else:
                auc_rep_batch = torch.full_like(auc_pgi_batch, float('nan'))
                auc_rep_norm_batch = torch.full_like(auc_pgi_batch, float('nan'))

            # ===== Output-Completeness & Compactness =====
            prob_end_pgi = pgi_curve[:, -1]                                # [Bf]
            prob_drop_end = (pgi_curve[:, 0] - prob_end_pgi)               # [Bf]

            tau = float(args.flip_threshold)
            ks_tensor = torch.tensor(list(ks), dtype=torch.float64, device=device)
            below = (pgi_curve[:, 1:] < tau)                                # [Bf, len(ks)]
            has_flip = below.any(dim=1)
            first_idx = below.float().argmax(dim=1)
            k_flip = torch.full((below.size(0),), float('nan'), dtype=torch.float64, device=device)
            k_flip[has_flip] = ks_tensor[first_idx[has_flip]]

            # ===== Record into rows =====
            for b in range(Bf):
                row = {
                    'image': img_paths_f[b],
                    # Raw AUC
                    'auc_pgi': f"{float(auc_pgi_batch[b].item()):.8f}",
                    'auc_pgu': f"{float(auc_pgu_batch[b].item()):.8f}",
                    'auc_repair': f"{float(auc_rep_batch[b].item()):.8f}" if make_separate_repair else "",
                    # Normalized AUC
                    'auc_pgi_norm': f"{float(auc_pgi_norm_batch[b].item()):.8f}",
                    'auc_pgu_norm': f"{float(auc_pgu_norm_batch[b].item()):.8f}",
                    'auc_repair_norm': f"{float(auc_rep_norm_batch[b].item()):.8f}" if make_separate_repair else "",
                    # Completeness & Compactness
                    'prob0': f"{float(pgi_curve[b,0].item()):.8f}",
                    'prob_end_pgi': f"{float(prob_end_pgi[b].item()):.8f}",
                    'prob_drop_end': f"{float(prob_drop_end[b].item()):.8f}",
                    'k_flip': f"{float(k_flip[b].item())}" if has_flip[b].item() else "",
                    'flip': int(has_flip[b].item()),
                    # Store the curves as-is
                    'ks': json.dumps([float(v) for v in xs.tolist()]),
                    'pgi_curve': json.dumps([float(v) for v in pgi_curve[b].tolist()]),
                    'pgu_curve': json.dumps([float(v) for v in pgu_curve[b].tolist()]),
                    'repair_curve': json.dumps([float(v) for v in repair_curve[b].tolist()]) if make_separate_repair else "",
                    'method': method,
                    'dataset': args.test_dataset,
                    'occlusion_mode': args.occlusion_mode,
                    'flip_threshold': args.flip_threshold,
                    'orig_used': int(orig_used_flags[b].item()) if need_orig else 0,
                }
                rows.append(row)

        # Save CSV
        assert len(rows) > 0, f"{method}: No fake samples evaluated (check labels or .pt alignment, or all prob0<0.5)"
        df = pd.DataFrame(rows)
        df.to_csv(save_csv, index=False)

        # ===== Summaries =====
        def mean_std_float(col: str, skip_empty: bool = False):
            s = df[col]
            if skip_empty:
                s = s[s != ""]
            vals = s.astype(float).tolist()
            if len(vals) == 0:
                return float('nan'), float('nan')
            m = float(np.mean(vals))
            sd = float(np.std(vals))
            return m, sd

        PGI_mean, PGI_std = mean_std_float('auc_pgi')
        PGU_mean, PGU_std = mean_std_float('auc_pgu')
        PGI_N_mean, PGI_N_std = mean_std_float('auc_pgi_norm')
        PGU_N_mean, PGU_N_std = mean_std_float('auc_pgu_norm')

        REP_mean, REP_std = (float('nan'), float('nan'))
        REPN_mean, REPN_std = (float('nan'), float('nan'))
        if 'auc_repair' in df.columns and df['auc_repair'].astype(str).str.len().gt(0).any():
            REP_mean,  REP_std  = mean_std_float('auc_repair', skip_empty=True)
            REPN_mean, REPN_std = mean_std_float('auc_repair_norm', skip_empty=True)

        prob0_mean, _ = mean_std_float('prob0')
        prob_end_mean, _ = mean_std_float('prob_end_pgi')
        drop_end_mean, drop_end_std = mean_std_float('prob_drop_end')
        flip_rate = float(df['flip'].astype(int).mean())
        k_flip_mean, k_flip_std = mean_std_float('k_flip', skip_empty=True)

        print(f"===> [{method}] Saved: {save_csv}")
        if not math.isnan(REP_mean):
            print(f"      REPAIR={REP_mean:.3f}±{REP_std:.3f}")
        print(f"      flip_rate={flip_rate:.3f}, k_flip_mean={k_flip_mean:.3f}±{k_flip_std:.3f}")

        summaries.append({
            'method': method,
            'dataset': args.test_dataset,
            # Raw AUC
            'PGI_mean': PGI_mean, 'PGI_std': PGI_std,
            'PGU_mean': PGU_mean, 'PGU_std': PGU_std,
            # Normalized AUC
            'PGI_norm_mean': PGI_N_mean, 'PGI_norm_std': PGI_N_std,
            'PGU_norm_mean': PGU_N_mean, 'PGU_norm_std': PGU_N_std,
            # Completeness & Compactness
            'prob0_mean': prob0_mean,
            'prob_end_pgi_mean': prob_end_mean,
            'prob_drop_end_mean': drop_end_mean, 'prob_drop_end_std': drop_end_std,
            'flip_rate': flip_rate,
            'k_flip_mean': k_flip_mean, 'k_flip_std': k_flip_std,
            # Repair summary (only when the control curve exists)
            'REPAIR_mean': REP_mean, 'REPAIR_std': REP_std,
            'REPAIR_norm_mean': REPN_mean, 'REPAIR_norm_std': REPN_std,
        })

    df = pd.DataFrame(summaries).round(6)
    print("\n========== [Incremental Deletion] ==========")
    print(df.set_index('method').T.to_string())
    print("==============================================")

if __name__ == '__main__':
    main()
