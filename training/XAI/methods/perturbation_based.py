import os
import json
import gc
import torch
import numpy as np
import torch.nn.functional as F

import cv2
from skimage.segmentation import slic

from captum.attr import Occlusion, KernelShap
from lime.lime_image import LimeImageExplainer


class BasePerturbationExplainer:
    """Base class for perturbation-based explanations.
    - Holds model/device and common save utilities.
    - Child classes call save_explanation(attributions, data_dict).
    """
    def __init__(self, model, config=None):
        if not isinstance(model, torch.nn.Module):
            raise ValueError(f"Expected torch.nn.Module, but got {type(model)}")
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.output_dir = config.get('output_dir')
        self.method_name = config.get('method', self.__class__.__name__.lower()) if config else self.__class__.__name__.lower()
        self.pt2mask_dict = {}

    def save_explanation(self, attributions, data_dict):
        """Save [B,C,H,W] attributions as per-image .pt maps.
        - Per-sample min–max normalize, then take channel-wise max -> [H,W].
        """
        dataset_name = data_dict.get('dataset_name', 'unknown_dataset')
        image_paths = data_dict.get('image_paths', [])
        B = attributions.size(0)

        save_dir = os.path.join(self.output_dir, self.method_name, dataset_name)
        os.makedirs(save_dir, exist_ok=True)

        for i in range(B):
            attr = attributions[i]
            image_path = image_paths[i]
            image_key = image_path.replace('/', '_').replace('\\', '_')

            # Normalize to [0,1] to make scales comparable across images.
            attr = attr - attr.min()
            attr = attr / (attr.max() + 1e-8)

            explanation_map = attr.max(0)[0]
            save_path = os.path.join(save_dir, f"{image_key}.pt")
            torch.save(explanation_map.cpu(), save_path)

    @torch.no_grad()
    def forward_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Model forward returning probability/logit vector used by explainers."""
        out = self.model({'image': x}, inference=True)['prob']
        return out


class OcclusionSensitivity(BasePerturbationExplainer):
    """Sliding-window occlusion:
    - Replace patches with a baseline (image mean or zero) and measure score drop.
    """
    def __init__(self, model, config=None):
        super().__init__(model, config)
        cfg = config or {}
        self.method_name = "occlusion"
        self.window = tuple(cfg.get('window', (32, 32)))          # (h, w) of the occlusion patch
        self.stride = tuple(cfg.get('stride', (16, 16)))          # (sh, sw) stride between patches
        self.use_image_mean_baseline = bool(cfg.get('use_image_mean_baseline', True))

    def generate(self, data_dict):
        x = data_dict['image'].to(self.device)
        B, C, H, W = x.shape

        def forward_func(inp):
            return self.forward_prob(inp)

        occl = Occlusion(forward_func)
        attrs = []

        for i in range(B):
            xi = x[i:i+1]
            baseline = float(xi.mean().item()) if self.use_image_mean_baseline else 0.0

            # Captum occlusion across the spatial dims with given window/stride.
            attr = occl.attribute(
                xi,
                sliding_window_shapes=(C, *self.window),
                strides=(C, *self.stride),
                baselines=baseline
            )
            attr = attr.abs()  # keep magnitude
            attrs.append(attr)

        attrs = torch.cat(attrs, dim=0)
        self.save_explanation(attrs, data_dict)
        gc.collect()
        torch.cuda.empty_cache()
        return attrs


class KernelShapAttribution(BasePerturbationExplainer):
    """KernelSHAP over coarse patches:
    - Partition image into non-overlapping patches and estimate Shapley values.
    """
    def __init__(self, model, config=None):
        super().__init__(model, config)
        cfg = config or {}
        self.method_name = "kernelshap"
        self.patch = int(cfg.get('patch', 32))                         # patch size (square)
        self.n_samples = int(cfg.get('n_samples', 200))                # SHAP samples
        self.perturbations_per_eval = int(cfg.get('perturbations_per_eval', 64))
        self.baseline_mode = cfg.get('baseline_mode', 'mean')          # {'mean','zero','const'}
        self.baseline_const = float(cfg.get('baseline_const', 0.0))

    def _make_feature_mask(self, H, W, C=3, device='cpu'):
        """Create feature mask IDs per patch for Captum KernelShap."""
        ph = pw = self.patch
        nh = (H + ph - 1) // ph
        nw = (W + pw - 1) // pw
        mask2d = torch.zeros((H, W), dtype=torch.long, device=device)
        fid = 0
        for i in range(nh):
            for j in range(nw):
                h0, h1 = i * ph, min((i + 1) * ph, H)
                w0, w1 = j * pw, min((j + 1) * pw, W)
                mask2d[h0:h1, w0:w1] = fid
                fid += 1
        mask = mask2d.unsqueeze(0).repeat(C, 1, 1)
        return mask

    def generate(self, data_dict):
        x = data_dict['image'].to(self.device)
        B, C, H, W = x.shape

        def forward_func(inp):
            return self.forward_prob(inp)

        explainer = KernelShap(forward_func)
        attrs = []

        for i in range(B):
            xi = x[i:i+1]

            # Choose baseline strategy.
            if self.baseline_mode == 'mean':
                base_val = float(xi.mean().item())
            elif self.baseline_mode == 'zero':
                base_val = 0.0
            else:
                base_val = self.baseline_const
            baselines = torch.full_like(xi, base_val)

            # Per-channel mask with integer region IDs.
            feature_mask = self._make_feature_mask(H, W, C=C, device=self.device)  # [3,H,W]
            feature_mask = feature_mask.unsqueeze(0)                                # [1,3,H,W]

            attr = explainer.attribute(
                xi,
                baselines=baselines,
                feature_mask=feature_mask.squeeze(0),
                n_samples=self.n_samples,
                perturbations_per_eval=self.perturbations_per_eval,
                show_progress=False
            )  # [1,3,H,W]
            attr = attr.abs()
            attrs.append(attr)

        attrs = torch.cat(attrs, dim=0)  # [B,3,H,W]
        self.save_explanation(attrs, data_dict)
        gc.collect()
        torch.cuda.empty_cache()
        return attrs


class LimeImageAttribution(BasePerturbationExplainer):
    """LIME (image): superpixel segmentation + local surrogate model."""
    def __init__(self, model, config=None):
        super().__init__(model, config)
        cfg = config or {}
        self.method_name = "lime"
        self.num_samples = int(cfg.get('num_samples', 400))     # typical 300–500
        self.batch_size  = int(cfg.get('batch_size', 64))
        self.target_label = int(cfg.get('target_label', 1))     # class index for explanation
        self.resize_to   = tuple(cfg.get('resize_to', (160, 160)))  # (W,H), or None to keep size
        self.n_segments  = int(cfg.get('n_segments', 64))       # typical 50–64
        self.segment_compactness = float(cfg.get('segment_compactness', 10.0))
        self.segment_sigma       = float(cfg.get('segment_sigma', 0.0))
        self.use_image_mean_hide_color = bool(cfg.get('use_image_mean_hide_color', True))
        self.random_state = int(cfg.get('random_state', 0))

        self.explainer = LimeImageExplainer(random_state=self.random_state)

    def _predict_numpy(self, imgs_hw3_list):
        """Adapter for LIME: list[np.ndarray HxWx3(float32 0..1)] -> probs Nx2."""
        arr = np.stack([np.transpose(img.astype(np.float32), (2, 0, 1)) for img in imgs_hw3_list], axis=0)
        t = torch.from_numpy(arr).to(self.device)
        with torch.no_grad():
            p_fake = self.forward_prob(t).detach().flatten()
        p_fake = p_fake.clamp(0.0, 1.0)
        p_real = 1.0 - p_fake
        probs = torch.stack([p_real, p_fake], dim=1)
        return probs.cpu().numpy()

    def _seg_fn(self, img_hw3: np.ndarray):
        """SLIC superpixels used by LIME."""
        return slic(
            img_hw3, n_segments=self.n_segments,
            compactness=self.segment_compactness,
            sigma=self.segment_sigma, start_label=0
        )

    def _lime_attr_single(self, xi: torch.Tensor) -> torch.Tensor:
        """Run LIME for one image and map superpixel weights back to a dense heatmap."""
        _, C, H, W = xi.shape
        img_np = xi[0].detach().permute(1, 2, 0).cpu().numpy()

        # Optional resize for faster LIME; map back later.
        if self.resize_to and (self.resize_to[0] != W or self.resize_to[1] != H):
            small = cv2.resize(img_np, self.resize_to, interpolation=cv2.INTER_AREA)
        else:
            small = img_np

        hide_color = float(small.mean()) if self.use_image_mean_hide_color else 0.0

        exp = self.explainer.explain_instance(
            image=small,
            classifier_fn=self._predict_numpy,
            labels=(self.target_label,),
            hide_color=hide_color,
            num_samples=self.num_samples,
            batch_size=self.batch_size,
            segmentation_fn=self._seg_fn
        )

        segments = exp.segments  # [h,w] superpixel IDs
        weight_small = np.zeros_like(segments, dtype=np.float32)
        for seg_id, w in exp.local_exp[self.target_label]:
            weight_small[segments == seg_id] = w

        # Resize weights back to original (W,H) if needed.
        if weight_small.shape[::-1] != (W, H):
            weight_map = cv2.resize(weight_small, (W, H), interpolation=cv2.INTER_LINEAR)
        else:
            weight_map = weight_small

        # Normalize and expand to 3 channels to match [C,H,W].
        weight_map = weight_map - weight_map.min()
        weight_map = weight_map / (weight_map.max() + 1e-8)
        attr = torch.from_numpy(weight_map).to(xi.device).unsqueeze(0).repeat(C, 1, 1)  # [3,H,W]
        return attr.unsqueeze(0)

    def generate(self, data_dict):
        """Apply LIME per image; concatenate and save."""
        x = data_dict['image'].to(self.device)
        attrs = [self._lime_attr_single(x[i:i+1]) for i in range(x.size(0))]
        attrs = torch.cat(attrs, dim=0)
        self.save_explanation(attrs, data_dict)
        import gc; gc.collect(); torch.cuda.empty_cache()
        return attrs
