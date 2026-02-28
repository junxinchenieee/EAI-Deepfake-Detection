# training/XAI/methods/cam_based.py
import os
import json
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============== Utilities ===============
def _resolve_module_by_name(model: nn.Module, dotted: str):
    if not dotted:
        return None
    cur = model
    for p in dotted.split('.'):
        if not hasattr(cur, p):
            return None
        cur = getattr(cur, p)
    return cur

def _upsample_like(x, ref):
    return F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)

def _minmax01(t: torch.Tensor, eps: float = 1e-8):
    tmin = t.amin(dim=(-2, -1), keepdim=True)
    tmax = t.amax(dim=(-2, -1), keepdim=True)
    return (t - tmin) / (tmax - tmin + eps)


# =============== CAM base (separate from gradients base) ===============
class BaseCAMExplainer:
    def __init__(self, model, config=None):
        if not isinstance(model, torch.nn.Module):
            raise ValueError(f"Expected torch.nn.Module, got {type(model)}")
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.config = config or {}
        self.output_dir = self.config.get('output_dir', './XAI/output/')
        self.method_name = self.config.get('method', self.__class__.__name__.lower())
        self.pt2mask_dict = {}

        # Target layer: prefer config['target_layer']; else auto-pick on first generate.
        tl_name = self.config.get('target_layer', None)
        layer = _resolve_module_by_name(self.model, tl_name) if tl_name else None
        if layer is None:
            self._pending_auto_pick = True
            self.target_layer = None
        else:
            self._pending_auto_pick = False
            self.target_layer = layer

        self._fh = None
        self._bh = None
        self._fmap = None
        self._grad = None

        # If target set, register hooks now.
        if self.target_layer is not None:
            self._register_cam_hooks()

    # ---------- Save (aligned with your existing logic) ----------
    def save_explanation(self, tensor_3ch, data_dict):
        dataset_name = data_dict.get('dataset_name', 'unknown_dataset')
        image_paths = data_dict.get('image_paths')
        B = tensor_3ch.size(0)

        for i in range(B):
            grad = tensor_3ch[i]  # [3,H,W]
            image_path = image_paths[i]
            image_key = image_path.replace('/', '_').replace('\\', '_')

            save_dir = os.path.join(self.output_dir, self.method_name, dataset_name)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{image_key}.pt")

            # Channel-max + [0,1] normalize before saving.
            explanation_map = grad.max(0)[0]
            explanation_map = (explanation_map - explanation_map.min()) / (explanation_map.max() - explanation_map.min() + 1e-8)
            torch.save(explanation_map.cpu(), save_path)

    # ---------- Helpers ----------
    def _forward_prob(self, x):
        return self.model({'image': x}, inference=True)['prob']  # [B,K] or [B,1]

    def _select_score(self, prob):
        # Sum to a scalar to avoid per-sample backward.
        return prob.sum()

    def _save_cam_1ch(self, cam_01, data_dict):
        """Expand 1ch CAM to 3ch to reuse save_explanation."""
        if cam_01.ndim == 3:  # [B,H,W] -> [B,1,H,W]
            cam_01 = cam_01.unsqueeze(1)
        cam_3c = cam_01.repeat(1, 3, 1, 1)  # [B,3,H,W]
        self.save_explanation(cam_3c, data_dict)

    # ---------- Auto-pick + (re)register hooks ----------
    def _auto_pick_target_layer(self, sample_x: torch.Tensor) -> nn.Module:
        active = []
        handles = []

        def make_fhook(name, m):
            def _fh(module, inp, out):
                if isinstance(out, torch.Tensor) and out.ndim == 4:
                    B, C, h, w = out.shape
                    if h >= 2 and w >= 2:
                        active.append((name, m))
            return _fh

        for name, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d):
                handles.append(m.register_forward_hook(make_fhook(name, m)))

        try:
            _ = self.model({'image': sample_x}, inference=True)
        finally:
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass

        if not active:
            raise RuntimeError(
                "Auto-pick failed: no usable Conv2d feature map. "
                "Set config['target_layer'] manually."
            )

        name, layer = active[-1]
        print(f"[CAM] Auto-picked target_layer: {name}")
        return layer

    def _register_cam_hooks(self):
        """(Re)register forward/backward hooks for CAM."""
        # Remove old hooks.
        for h in (getattr(self, "_fh", None), getattr(self, "_bh", None)):
            try:
                if h:
                    h.remove()
            except Exception:
                pass
        self._fh, self._bh = None, None

        if self.target_layer is None:
            return

        def f_hook(module, inp, out):
            self._fmap = out.detach()

        def b_hook(module, grad_in, grad_out):
            self._grad = grad_out[0]  # grad wrt layer output

        self._fh = self.target_layer.register_forward_hook(f_hook)
        if hasattr(self.target_layer, 'register_full_backward_hook'):
            self._bh = self.target_layer.register_full_backward_hook(b_hook)
        else:
            self._bh = self.target_layer.register_backward_hook(b_hook)

    def _maybe_autopick_and_register(self, x: torch.Tensor):
        """Auto-pick & hook on first call to generate()."""
        if getattr(self, "_pending_auto_pick", False) or self.target_layer is None:
            self.target_layer = self._auto_pick_target_layer(x[:1])
            self._pending_auto_pick = False
            self._register_cam_hooks()

    def _cleanup(self):
        try:
            if hasattr(self, "_fh") and self._fh:
                self._fh.remove()
            if hasattr(self, "_bh") and self._bh:
                self._bh.remove()
        except Exception:
            pass
        self._fmap, self._grad = None, None
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    def __del__(self):
        self._cleanup()


# =============== Grad-CAM ===============
class GradCAM(BaseCAMExplainer):
    """w_k = GAP(dY/dA_k); CAM = ReLU(sum_k w_k * A_k)"""
    def generate(self, data_dict):
        x = data_dict['image'].to(self.device).detach()
        x.requires_grad_(True)

        # Auto-pick + hook.
        self._maybe_autopick_and_register(x)

        prob = self._forward_prob(x)
        score = self._select_score(prob)

        self.model.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad.zero_()
        score.backward(retain_graph=True)

        fmap = self._fmap              # [B,C,h,w]
        grad = self._grad              # [B,C,h,w]
        if fmap is None or grad is None:
            raise RuntimeError("Hooks missed fmap/grad. Check target_layer or inference path.")

        weights = grad.mean(dim=(2, 3), keepdim=True)           # [B,C,1,1]
        cam = (weights * fmap).sum(dim=1, keepdim=True)         # [B,1,h,w]
        cam = F.relu(cam)
        cam = _upsample_like(cam, x)                            # [B,1,H,W]

        # Per-sample [0,1] normalize.
        B = cam.size(0)
        cams = []
        for i in range(B):
            m = cam[i, 0]
            m = (m - m.min()) / (m.max() - m.min() + 1e-8)
            cams.append(m)
        cam_01 = torch.stack(cams, dim=0)                      # [B,H,W]

        self._save_cam_1ch(cam_01, data_dict)
        return cam_01


# =============== Grad-CAM++ ===============
class GradCAMPlusPlus(BaseCAMExplainer):
    """
    alpha_k = g^2 / (2*g^2 + sum_{i,j} A_k * g^3 + eps), w_k = sum_{i,j} alpha_k * ReLU(g),
    CAM = ReLU(sum_k w_k * A_k)
    """
    def generate(self, data_dict):
        x = data_dict['image'].to(self.device).detach()
        x.requires_grad_(True)

        # Auto-pick + hook.
        self._maybe_autopick_and_register(x)

        prob = self._forward_prob(x)
        score = self._select_score(prob)

        self.model.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad.zero_()
        score.backward(retain_graph=True)

        fmap = self._fmap  # [B,C,h,w]
        g = self._grad     # [B,C,h,w]
        if fmap is None or g is None:
            raise RuntimeError("Hooks missed fmap/grad. Check target_layer or inference path.")

        eps = 1e-8
        g2 = g * g
        g3 = g2 * g

        # sum_{i,j} A_k * g^3
        sum_A_g3 = (fmap * g3).sum(dim=(2, 3), keepdim=True)        # [B,C,1,1]
        # alpha has shape [B,C,h,w] (denominator broadcast over h,w)
        denom = 2.0 * g2 + sum_A_g3 + eps
        alpha = g2 / denom
        alpha = torch.clamp(alpha, min=0.0)

        # w_k = sum_{i,j} alpha * ReLU(g)
        weights = (alpha * F.relu(g)).sum(dim=(2, 3), keepdim=True)  # [B,C,1,1]
        cam = (weights * fmap).sum(dim=1, keepdim=True)              # [B,1,h,w]
        cam = F.relu(cam)
        cam = _upsample_like(cam, x)

        B = cam.size(0)
        cams = []
        for i in range(B):
            m = cam[i, 0]
            m = (m - m.min()) / (m.max() - m.min() + 1e-8)
            cams.append(m)
        cam_01 = torch.stack(cams, dim=0)

        self._save_cam_1ch(cam_01, data_dict)
        return cam_01


# =============== Score-CAM ===============
class ScoreCAM(BaseCAMExplainer):
    """
    No gradients: upsample & normalize each channel map A_k to mask input; forward score as weight.
    Supports top-k channels and batched evaluation for speed.
    config:
        - scorecam_topk (int|None): keep strongest K channels (by fmap global mean). None → all.
        - scorecam_batch (int): batch size for masked forwards. Default 16.
    """
    def generate(self, data_dict):
        x = data_dict['image'].to(self.device).detach()  # [B,3,H,W]

        # Auto-pick + hook (uses only forward hook).
        self._maybe_autopick_and_register(x)

        # Forward once to cache fmap.
        _ = self._forward_prob(x)
        fmap = self._fmap  # [B,C,h,w]
        if fmap is None:
            raise RuntimeError("Forward hook missed fmap. Check target_layer or inference path.")

        B, C, h, w = fmap.shape
        H, W = x.shape[-2:]

        # Channel selection
        topk = 32
        with torch.no_grad():
            ch_score = fmap.mean(dim=(0, 2, 3))  # [C]
            idx = torch.argsort(ch_score, descending=True)
            if isinstance(topk, int) and 1 <= topk < C:
                idx = idx[:topk]
        C_eff = idx.numel()

        # Build per-channel masks ([0,1], upsampled).
        masks = []
        for c in idx:
            a = fmap[:, c, :, :]                                  # [B,h,w]
            a = F.relu(a)
            a_up = _upsample_like(a.unsqueeze(1), x).squeeze(1)   # [B,H,W]
            a_up = _minmax01(a_up)                                # per-sample normalize
            masks.append(a_up)
        masks = torch.stack(masks, dim=1)                         # [B,C_eff,H,W]

        # Batched forwards to get weights.
        bs = int(self.config.get('scorecam_batch', 16))
        weights = torch.zeros(B, C_eff, device=self.device)

        with torch.no_grad():
            for b in range(B):
                ch_ptr = 0
                while ch_ptr < C_eff:
                    ch_end = min(ch_ptr + bs, C_eff)
                    m = masks[b, ch_ptr:ch_end, :, :]             # [k,H,W]
                    m = m.unsqueeze(1)                            # [k,1,H,W]
                    x_masked = x[b:b+1] * m                       # [k,3,H,W]
                    prob = self._forward_prob(x_masked)           # [k,K] or [k]
                    w = prob
                    weights[b, ch_ptr:ch_end] = w
                    ch_ptr = ch_end

        # Combine: sum_k w_k * mask_k
        cam = torch.zeros(B, 1, H, W, device=self.device)
        for b in range(B):
            m = masks[b] * weights[b].view(-1, 1, 1)             # [C_eff,H,W]
            s = m.sum(dim=0, keepdim=True)                       # [1,H,W]
            cam[b:b+1, 0] = s

        cam = F.relu(cam)

        # Per-sample [0,1] normalize
        cams = []
        for i in range(B):
            m = cam[i, 0]
            m = (m - m.min()) / (m.max() - m.min() + 1e-8)
            cams.append(m)
        cam_01 = torch.stack(cams, dim=0)                        # [B,H,W]

        self._save_cam_1ch(cam_01, data_dict)
        return cam_01


# =============== CAM ===============
class CAM(BaseCAMExplainer):
    """
    Classic CAM: conv feature -> GAP -> Linear; channel weights from classifier.
    If the final head is not GAP+FC, this may not apply.
    config:
        - cam_classifier (str): dotted path to the Linear layer; if empty, auto-pick last nn.Linear.
    """
    def __init__(self, model, config=None):
        super().__init__(model, config)
        # Find classifier layer.
        clf_name = self.config.get('cam_classifier', None)
        classifier = _resolve_module_by_name(self.model, clf_name) if clf_name else None
        if classifier is None:
            # Auto-pick the last Linear
            last_linear = None
            for m in self.model.modules():
                if isinstance(m, nn.Linear):
                    last_linear = m
            classifier = last_linear
        if not isinstance(classifier, nn.Linear):
            raise RuntimeError("CAM requires a Linear classifier. Set config['cam_classifier'].")

        self.classifier = classifier  # nn.Linear(in_features=C, out_features=K)

    def generate(self, data_dict):
        x = data_dict['image'].to(self.device).detach()

        # Auto-pick + hook (need fmap).
        self._maybe_autopick_and_register(x)

        # Forward once to capture fmap from target layer.
        prob = self._forward_prob(x)  # [B,K] or [B,1]
        fmap = self._fmap             # [B,C,h,w]
        if fmap is None:
            raise RuntimeError("Forward hook missed fmap. Check target_layer or inference path.")
        B, C, h, w = fmap.shape

        # Assume classifier takes GAP(fmap): W shape [K,C]
        W = self.classifier.weight    # [K,C]
        # Bias typically ignored in CAM.
        # b = self.classifier.bias

        # Use a generic weight vector (sum over classes). For per-class, pick argmax weights per sample.
        w_all = W.sum(dim=0)          # [C]

        # CAM = sum_c w_c * A_c (on fmap size, then upsample).
        cam = (fmap * w_all.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)  # [B,1,h,w]
        cam = F.relu(cam)
        cam = _upsample_like(cam, x)  # [B,1,H,W]

        cams = []
        for i in range(B):
            m = cam[i, 0]
            m = (m - m.min()) / (m.max() - m.min() + 1e-8)
            cams.append(m)
        cam_01 = torch.stack(cams, dim=0)  # [B,H,W]

        self._save_cam_1ch(cam_01, data_dict)
        return cam_01
