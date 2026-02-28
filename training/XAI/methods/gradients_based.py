import os
import json
import torch
import gc
import torch.nn as nn
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import torch.nn.functional as F
EPS = 1e-6
from captum.attr import (
    IntegratedGradients as IG,
    Saliency as SL,
    InputXGradient as IXG,
    LayerConductance as LC,
    GuidedBackprop as GBP,
    Deconvolution as DC,
    LRP as CAPTUM_LRP,
    NoiseTunnel,
    configure_interpretable_embedding_layer,
    remove_interpretable_embedding_layer,
)


class BaseGradientExplainer:
    """Base class for gradient-based explainers.
    - Holds model/device.
    - Resolves output directory relative to repo root.
    - Provides saving utilities for explanation maps and JSON mappings.
    """
    def __init__(self, model, config=None):
        if not isinstance(model, torch.nn.Module):
            raise ValueError(f"Expected torch.nn.Module, but got {type(model)}")
        self.model = model.eval()
        self.device = next(model.parameters()).device

        # Resolve repository root and output directory (supports relative config path).
        self.repo = Path(__file__).resolve().parents[3]       
        out  = Path(config['output_dir'])
        self.output_dir = (out if out.is_absolute() else (self.repo / out)).resolve()
        
        self.method_name = config.get('method', self.__class__.__name__.lower()) if config else self.__class__.__name__.lower()
        self.pt2mask_dict = {}

    def save_explanation(self, gradients, data_dict):
        """Save per-sample explanation as a single-channel tensor file (.pt).
        - Input gradients: [B, C, H, W]
        - We reduce over channels by max and min-max normalize to [0, 1].
        - File name is derived from image path to ensure uniqueness.
        """
        dataset_name = data_dict.get('dataset_name', 'unknown_dataset')
        image_paths = data_dict.get('image_paths')
        B = gradients.size(0)

        for i in range(B):
            grad = gradients[i]
            image_path = image_paths[i]
            image_key = image_path.replace('/', '_').replace('\\', '_')

            save_dir = os.path.join(self.output_dir, self.method_name, dataset_name)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{image_key}.pt")

            # Channel-wise max projection -> [H, W], then normalize.
            explanation_map = grad.max(0)[0]
            explanation_map = (explanation_map - explanation_map.min()) / (explanation_map.max() - explanation_map.min() + 1e-8)
            # explanation_map = grad.max(0)[0].to(torch.float32)
            torch.save(explanation_map.cpu(), save_path)

        self.save_json_mapping(dataset_name, image_paths)

    def save_json_mapping(self, dataset_name, image_paths):
        for image_path in image_paths:
            image_key = image_path.replace('/', '_').replace('\\', '_') + ".pt"

            if dataset_name in ["FF-DF", "FaceForensics++"]:
                if "manipulated_sequences" in image_path and "/frames/" in image_path:
                    parts = image_path.split('/')
                    id_pair = parts[-2]
                    id1 = id_pair.split('_')[0]
                    img_name = parts[-1]

                    orig_path = (
                        self.repo / "datasets" / "rgb" / "FaceForensics++" /
                        "original_sequences" / "youtube" / "c23" / "frames" /
                        id1 / img_name
                    ).resolve()
                    self.pt2mask_dict[image_path] = orig_path
                else:
                    self.pt2mask_dict[image_path] = None

            elif dataset_name == "Celeb-DF-v2":
                if (
                    "Celeb-DF-v2" in image_path
                    and "Celeb-synthesis" in image_path
                    and "/frames/" in image_path
                ):
                    # .../Celeb-synthesis/frames/id4_id20_0008/230.png
                    parts = image_path.split('/')
                    id_pair = parts[-2]   # "id4_id20_0008"
                    img_name = parts[-1]  # "230.png"

                    id_parts = id_pair.split('_')
                    if len(id_parts) >= 3:
                        orig_dir = f"{id_parts[0]}_{id_parts[2]}"  # id4_0008
                        orig_path = (
                            self.repo / "datasets" / "rgb" / "Celeb-DF-v2" /
                            "Celeb-real" / "frames" / orig_dir / img_name
                        ).resolve()
                        self.pt2mask_dict[image_path] = orig_path
                    else:
                        self.pt2mask_dict[image_path] = None
                else:
                    self.pt2mask_dict[image_path] = None

            else:
                self.pt2mask_dict[image_path] = None

        json_dir = os.path.join(self.output_dir, "json", dataset_name)
        os.makedirs(json_dir, exist_ok=True)
        json_path = os.path.join(json_dir, "pt2orig.json")
        with open(json_path, 'w') as f:
            json.dump(self.pt2mask_dict, f, indent=2, default=str)





class VanillaGradient(BaseGradientExplainer):
    """Vanilla gradients: d(prob)/d(input). Uses abs to get saliency magnitude."""
    def generate(self, data_dict):
        input_tensor = data_dict['image'].clone().detach().requires_grad_(True)
        data_dict = data_dict.copy()
        data_dict['image'] = input_tensor

        output = self.model(data_dict, inference=True)
        prob = output['prob']
        self.model.zero_grad()
        prob.sum().backward()  # accumulate grads over batch

        gradient = input_tensor.grad.detach().abs()
        self.save_explanation(gradient, data_dict)
        return gradient


class IntegratedGradients(BaseGradientExplainer):
    """Captum Integrated Gradients wrapper. Returns |IG|."""
    def generate(self, data_dict):
        input_tensor = data_dict['image'].clone().detach().requires_grad_(True)
        data_dict = data_dict.copy()
        data_dict['image'] = input_tensor

        # Forward function that matches Captum's expected signature.
        def forward_func(x):
            new_dict = data_dict.copy()
            new_dict['image'] = x
            return self.model(new_dict, inference=True)['prob']

        ig = IG(forward_func)  
        attr = ig.attribute(input_tensor, target=None)  # None → use model output directly
        gradient = attr.abs()
        self.save_explanation(gradient, data_dict)
        return gradient

class SmoothGrad(BaseGradientExplainer):
    """Captum SmoothGrad over Saliency. Adds noise and averages gradients."""
    def generate(self, data_dict):
        input_tensor = data_dict['image'].clone().detach().to(self.device).requires_grad_()

        def forward_func(x):
            return self.model({'image': x}, inference=True)['prob']

        saliency = SL(forward_func)
        smoothgrad = NoiseTunnel(saliency)

        # Typical SmoothGrad hyperparameters; abs=True to aggregate magnitude.
        attr = smoothgrad.attribute(
            input_tensor,
            nt_type='smoothgrad',   
            nt_samples=25,           
            stdevs=0.15,            
            abs=True                
        )

        self.save_explanation(attr.abs(), data_dict)
        return attr


class InputXGradient(BaseGradientExplainer):
    """Elementwise product: input * gradient."""
    def generate(self, data_dict):
        input_tensor = data_dict['image'].clone().detach().requires_grad_(True)
        data_dict = data_dict.copy()
        data_dict['image'] = input_tensor

        def forward_func(x):
            new_dict = data_dict.copy()
            new_dict['image'] = x
            return self.model(new_dict, inference=True)['prob']

        explainer = IXG(forward_func)
        attr = explainer.attribute(input_tensor)
        self.save_explanation(attr.abs(), data_dict)
        return attr

class RawModelWrapper(torch.nn.Module):
    """Wrap model(data_dict) → model(x) so Captum methods expecting f(x) can be used."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        data_dict = {'image': x}
        return self.model(data_dict, inference=True)['prob']

class GuidedBackprop(BaseGradientExplainer):
    """Guided Backpropagation (non-negative gradients through ReLU)."""
    def generate(self, data_dict):
        input_tensor = data_dict['image'].clone().detach().requires_grad_(True)
        data_dict = data_dict.copy()
        data_dict['image'] = input_tensor

        wrapped_model = RawModelWrapper(self.model)
        wrapped_model.eval()

        explainer = GBP(wrapped_model)
        attr = explainer.attribute(input_tensor)

        self.save_explanation(attr.abs(), data_dict)
        return attr


class Deconvolution(BaseGradientExplainer):
    """Deconvolution (Zeiler & Fergus) style backprop rules."""
    def generate(self, data_dict):
        input_tensor = data_dict['image'].clone().detach().requires_grad_(True)
        data_dict = data_dict.copy()
        data_dict['image'] = input_tensor

        wrapped_model = RawModelWrapper(self.model)
        wrapped_model.eval()

        explainer = DC(wrapped_model)
        attr = explainer.attribute(input_tensor)

        self.save_explanation(attr.abs(), data_dict)
        return attr


# ===== Zennit-based LRP (EpsilonPlus) =====
try:
    from zennit.attribution import Gradient as ZGradient
    from zennit.composites import EpsilonPlus
    _ZENNIT_OK = True
    _ZENNIT_ERR = None
except Exception as _e:
    _ZENNIT_OK = False
    _ZENNIT_ERR = _e

class LRP(BaseGradientExplainer):
    """
    LRP via Zennit (EpsilonPlus composite).
    Output has the same shape as input; optionally resized to 256x256 for consistency.
    """
    def __init__(self, model, config=None):
        super().__init__(model, config)
        if not _ZENNIT_OK:
            raise ImportError(f"Zennit not available: {_ZENNIT_ERR}\nInstall with: pip install zennit")

        # Wrap to expose logits/probabilities directly to Zennit.
        class _LogitsWrapper(torch.nn.Module):
            def __init__(self, mdl):
                super().__init__()
                self.m = mdl
            def forward(self, x):
                out = self.m({'image': x}, inference=True)
                if isinstance(out, dict):
                    if 'cls' in out:  return out['cls']   # (B,C) logits
                    if 'prob' in out: return out['prob']  # (B,C) or (B,)
                return out

        self._wrapped = _LogitsWrapper(self.model).eval()
        self._comp = EpsilonPlus()

    def generate(self, data_dict):
        x = data_dict['image'].to(self.device).detach().requires_grad_(True)

        # Target selection: argmax per-sample if multi-class; otherwise element-wise attribution.
        with torch.no_grad():
            y = self._wrapped(x)
            target_idx = None if y.dim() == 1 else y.argmax(dim=1)

        # Construct attribution objective (one-hot on argmax class).
        def _attr_output_fn(y_out):
            if y_out.dim() == 2 and target_idx is not None:
                oh = torch.zeros_like(y_out)
                oh.scatter_(1, target_idx.view(-1,1), 1.0)
                return oh
            return torch.ones_like(y_out)

        attributor = ZGradient(self._wrapped, composite=self._comp)
        _, A = attributor(x, _attr_output_fn)  # A: (B,3,H,W)

        # Normalize spatial size for downstream consumers/metrics.
        if A.shape[-2:] != (256, 256):
            A = F.interpolate(A, size=(256, 256), mode='bilinear', align_corners=False)

        self.save_explanation(A, data_dict)
        return A


# ===== Zennit-based DeepTaylor =====
try:
    from zennit.attribution import Gradient as ZGradient
    try:
        from zennit.composites import EpsilonGammaBox as ZBoxComposite
        _ZBOX_NAME = "EpsilonGammaBox"
    except Exception:
        try:
            from zennit.composites import ZBox as ZBoxComposite
            _ZBOX_NAME = "ZBox"
        except Exception:
            from zennit.composites import EpsilonPlus as ZBoxComposite
            _ZBOX_NAME = "EpsilonPlus(fallback)"
    _ZENNIT_OK = True
    _ZENNIT_ERR = None
except Exception as _e:
    _ZENNIT_OK = False
    _ZENNIT_ERR = _e

class DeepTaylor(BaseGradientExplainer):
    """Deep Taylor Decomposition via Zennit box composite.
    - Infers input bounds in either [0,1] or normalized (mean/std) space.
    """
    def __init__(self, model, config=None):
        super().__init__(model, config)
        if not _ZENNIT_OK:
            raise ImportError(f"Zennit not available: {_ZENNIT_ERR}\nInstall with: pip install zennit")

        class _LogitsWrapper(torch.nn.Module):
            def __init__(self, mdl):
                super().__init__()
                self.m = mdl
            def forward(self, x):
                out = self.m({'image': x}, inference=True)
                if isinstance(out, dict):
                    if 'cls' in out:  return out['cls']   # (B,C) logits
                    if 'prob' in out: return out['prob']  # (B,C) or (B,)
                return out

        self._wrapped = _LogitsWrapper(self.model).eval()
        self._box_name = _ZBOX_NAME

    def _infer_bounds(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Infer per-channel minimum/maximum bounds compatible with normalization.
        - If input appears to be in [0,1], use [0,1].
        - Otherwise, assume ImageNet mean/std normalization.
        """
        B, C, H, W = x.shape
        x_min, x_max = float(x.min().item()), float(x.max().item())

        if -0.05 <= x_min and x_max <= 1.05:
            low  = torch.zeros(B, C, H, W, device=x.device, dtype=x.dtype)
            high = torch.ones (B, C, H, W, device=x.device, dtype=x.dtype)
            return low, high

        # ImageNet mean/std bounds in normalized space.
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, C, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, C, 1, 1)
        low_c  = (0.0 - mean) / std          # 1×C×1×1
        high_c = (1.0 - mean) / std
        low  = low_c.expand(B, C, H, W).contiguous()
        high = high_c.expand(B, C, H, W).contiguous()
        return low, high

    def generate(self, data_dict):
        x = data_dict['image'].to(self.device).detach().requires_grad_(True)

        # Select target class via argmax if multi-class.
        with torch.no_grad():
            y = self._wrapped(x)
            target_idx = None if y.dim() == 1 else y.argmax(dim=1)

        def _attr_output_fn(y_out):
            if y_out.dim() == 2 and target_idx is not None:
                oh = torch.zeros_like(y_out)
                oh.scatter_(1, target_idx.view(-1,1), 1.0)
                return oh
            return torch.ones_like(y_out)

        # Box composite bounds.
        low, high = self._infer_bounds(x)
        comp = ZBoxComposite(low=low, high=high)

        attributor = ZGradient(self._wrapped, composite=comp)
        _, A = attributor(x, _attr_output_fn)  # A: B×3×H×W

        # Resize to canonical size for evaluation/visualization.
        if A.shape[-2:] != (256, 256):
            A = F.interpolate(A, size=(256, 256), mode='bilinear', align_corners=False)

        self.save_explanation(A, data_dict)
        return A



# ===== Zennit-based Excitation Backprop =====
try:
    from zennit.composites import ExcitationBackprop as ZExcitationBP
    _ZENNIT_EB_OK = True
except Exception as _e:
    _ZENNIT_EB_OK = False

class ExcitationBP(BaseGradientExplainer):
    """Excitation Backprop via Zennit composite (class-selective top-down signal)."""

    def __init__(self, model, config=None):
        super().__init__(model, config)
        if not _ZENNIT_OK or not _ZENNIT_EB_OK:
            raise ImportError("Zennit ExcitationBackprop requires zennit. Install with: pip install zennit")

        class _LogitsWrapper(torch.nn.Module):
            def __init__(self, mdl):
                super().__init__()
                self.m = mdl
            def forward(self, x):
                out = self.m({'image': x}, inference=True)
                if isinstance(out, dict):
                    if 'cls' in out:  return out['cls']
                    if 'prob' in out: return out['prob']
                return out

        self._wrapped = _LogitsWrapper(self.model).eval()
        self._comp = ZExcitationBP()

    def generate(self, data_dict):
        x = data_dict['image'].to(self.device).detach().requires_grad_(True)

        # Target selection per sample.
        with torch.no_grad():
            y = self._wrapped(x)
            target_idx = None if y.dim() == 1 else y.argmax(dim=1)

        def _attr_output_fn(y_out):
            if y_out.dim() == 2 and target_idx is not None:
                oh = torch.zeros_like(y_out)
                oh.scatter_(1, target_idx.view(-1,1), 1.0)
                return oh
            return torch.ones_like(y_out)

        attributor = ZGradient(self._wrapped, composite=self._comp)
        _, A = attributor(x, _attr_output_fn)  # Bx3xHxW

        if A.shape[-2:] != (256, 256):
            A = F.interpolate(A, size=(256, 256), mode='bilinear', align_corners=False)

        self.save_explanation(A, data_dict)
        return A
