# training/XAI/utils/method_loader.py
from typing import Dict, Any


def _set_method(config: Dict[str, Any], name: str) -> Dict[str, Any]:
    cfg = dict(config or {})
    cfg.setdefault("method", name)
    return cfg


def load_explainer(method_name, model, config):
    n = (method_name or "").lower()

    # ---- gradient-based ----
    from XAI.methods.gradients_based import (
        VanillaGradient,
        IntegratedGradients,
        SmoothGrad,
        InputXGradient,
        GuidedBackprop,
        Deconvolution,
        LRP,
        DeepTaylor,
        ExcitationBP,
    )

    # ---- cam-based ----
    from XAI.methods.cam_based import (
        GradCAM,
        GradCAMPlusPlus,
        ScoreCAM,
        CAM,
    )

    # ---- perturbation-based ----
    from XAI.methods.perturbation_based import (
        OcclusionSensitivity,
        KernelShapAttribution,
        LimeImageAttribution,
    )

    table = {
        # gradients
        "vanillagradient": ("vanillagradient", VanillaGradient),

        "integratedgradients": ("integratedgradients", IntegratedGradients),
        "ig": ("integratedgradients", IntegratedGradients),

        "smoothgrad": ("smoothgrad", SmoothGrad),

        "inputxgradient": ("inputxgradient", InputXGradient),

        "guidedbackprop": ("guidedbackprop", GuidedBackprop),

        "deconvolution": ("deconvolution", Deconvolution),

        "lrp": ("lrp", LRP),

        "deeptaylor": ("deeptaylor", DeepTaylor),

        "excitationbp": ("excitationbp", ExcitationBP),

        # cams
        "gradcam": ("gradcam", GradCAM),

        "gcampp": ("gradcamplusplus", GradCAMPlusPlus),

        "scorecam": ("scorecam", ScoreCAM),

        "cam": ("cam", CAM),

        # perturbation
        "occlusion": ("occlusion", OcclusionSensitivity),

        "kernelshap": ("kernelshap", KernelShapAttribution),

        "lime": ("lime", LimeImageAttribution),
    }

    if n not in table:
        raise NotImplementedError(f"Unsupported method: {method_name}")

    canonical, cls = table[n]
    cfg = _set_method(config, canonical)
    return cls(model, config=cfg)
