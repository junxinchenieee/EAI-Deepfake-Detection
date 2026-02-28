import os
import re
import glob
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

# =========================
# 1) USER SETTINGS
# =========================
FAKE_A_START = "/mnt/datadisk0/deepfakebench/DeepfakeBench/datasets/rgb/FaceForensics++/manipulated_sequences/Deepfakes/c23/frames/480_389/000.png"
FAKE_B_START = "/mnt/datadisk0/deepfakebench/DeepfakeBench/datasets/rgb/FaceForensics++/manipulated_sequences/Deepfakes/c23/frames/623_630/033.png"

XAI_OUT_ROOT = "/mnt/datadisk0/deepfakebench/DeepfakeBench/training/XAI/output"
MASK_ROOT = "/mnt/datadisk0/deepfakebench/DeepfakeBench/training/XAI/masks/FF-DF"

METHOD = "lime"
DATASET = "FF-DF"

OUT_DIR = "output/visual"
os.makedirs(OUT_DIR, exist_ok=True)

# overlay params
ALPHA_HEAT = 0.45
CMAP = "jet"

MASK_ALPHA = 0.35
MASK_COLOR = (0.2, 0.8, 0.2)  # light green

EPS = 1e-8


# =========================
# 2) IO HELPERS
# =========================
def load_rgb01(img_path: str) -> np.ndarray:
    im = Image.open(img_path).convert("RGB")
    return np.asarray(im).astype(np.float32) / 255.0

def save_rgb01(out_path: str, arr01: np.ndarray, dpi: int = 300):
    arr01 = np.clip(arr01, 0.0, 1.0)
    plt.figure(figsize=(5, 5))
    plt.imshow(arr01)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close()

def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < EPS:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


# =========================
# 3) FF++ PATH MAPPING: fake -> real
# =========================
_PAIR_DIR_RE = re.compile(r"^(\d+)_(\d+)$")

def ffpp_root_from_path(any_path: str) -> str:
    token = "/FaceForensics++/"
    return any_path.split(token)[0] + "/FaceForensics++"

def parse_compression(fake_path: str) -> str:
    parts = fake_path.split(os.sep)
    idx = parts.index("Deepfakes")
    return parts[idx + 1]  # c23

def parse_pair_folder(fake_path: str):
    pair_folder = os.path.basename(os.path.dirname(fake_path))
    m = _PAIR_DIR_RE.match(pair_folder)
    src = m.group(1)
    return src

def fake_to_real_path(fake_path: str) -> str:
    root = ffpp_root_from_path(fake_path)
    comp = parse_compression(fake_path)
    src = parse_pair_folder(fake_path)
    frame_file = os.path.basename(fake_path)
    return os.path.join(
        root,
        "original_sequences", "youtube", comp, "frames", src, frame_file
    )


# =========================
# 4) Adjacent frames (no +1 assumption)
# =========================
def list_numeric_pngs(folder: str):
    files = glob.glob(os.path.join(folder, "*.png"))
    items = []
    for p in files:
        m = re.match(r"^(\d+)\.png$", os.path.basename(p))
        if m:
            items.append((int(m.group(1)), p))
    items.sort(key=lambda x: x[0])
    return items

def three_adjacent_frames(fake_start: str):
    folder = os.path.dirname(fake_start)
    start_idx = int(os.path.splitext(os.path.basename(fake_start))[0])
    items = list_numeric_pngs(folder)
    nums = [n for n, _ in items]
    pos = nums.index(start_idx)
    chosen = items[pos:pos + 3]

    fake_paths = [p for _, p in chosen]
    real_paths = [fake_to_real_path(p) for p in fake_paths]
    return fake_paths, real_paths


# =========================
# 5) LIME heatmap loader
# =========================
def flatten_abs_path_to_pt(abs_img_path: str) -> str:
    s = abs_img_path.replace("/", "_")
    if not s.startswith("_"):
        s = "_" + s
    return s + ".pt"

def load_heatmap_pt(img_path: str):
    pt_name = flatten_abs_path_to_pt(img_path)
    pt_path = os.path.join(XAI_OUT_ROOT, METHOD, DATASET, pt_name)
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"LIME pt not found: {pt_path}")
    return torch.load(pt_path, map_location="cpu")

def tensor_to_heatmap01(x: torch.Tensor) -> np.ndarray:
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 3:
        x = x.mean(dim=0)
    x = x.detach().cpu().float().numpy()
    return normalize01(x)


# =========================
# 6) MASK loader (NEW, CORRECT)
# =========================
def flatten_abs_path_to_mask_png(abs_img_path: str) -> str:
    s = abs_img_path.replace("/", "_")
    if not s.startswith("_"):
        s = "_" + s
    return s + ".png"

def load_mask_for_fake(fake_img_path: str) -> np.ndarray:
    mask_name = flatten_abs_path_to_mask_png(fake_img_path)
    mask_path = os.path.join(MASK_ROOT, mask_name)
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    m = Image.open(mask_path).convert("L")
    return np.asarray(m).astype(np.float32) / 255.0


# =========================
# 7) OVERLAYS
# =========================
def overlay_heatmap_on_image(img01, heat01, alpha=ALPHA_HEAT, cmap=CMAP):
    cm = plt.get_cmap(cmap)
    heat_rgb = cm(heat01)[..., :3]
    return np.clip((1 - alpha) * img01 + alpha * heat_rgb, 0, 1)

def overlay_mask_on_image(img01, mask01, color=MASK_COLOR, alpha=MASK_ALPHA):
    color = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    return np.clip(
        img01 * (1 - alpha * mask01[..., None]) + color * (alpha * mask01[..., None]),
        0, 1
    )


# =========================
# 8) MAIN
# =========================
def save_sequence(tag, fake_start, save_real, save_lime, save_mask):
    fake_paths, real_paths = three_adjacent_frames(fake_start)

    for i, fpath in enumerate(fake_paths):
        fake_img = load_rgb01(fpath)
        save_rgb01(os.path.join(OUT_DIR, f"{tag}_fake_{i}.png"), fake_img)

        real_img = None
        if save_real or save_mask:
            real_img = load_rgb01(real_paths[i])
            if save_real:
                save_rgb01(os.path.join(OUT_DIR, f"{tag}_real_{i}.png"), real_img)

        if save_lime:
            heat = tensor_to_heatmap01(load_heatmap_pt(fpath))
            lime_overlay = overlay_heatmap_on_image(fake_img, heat)
            save_rgb01(os.path.join(OUT_DIR, f"{tag}_lime_overlay_{i}.png"), lime_overlay)

        if save_mask:
            mask01 = load_mask_for_fake(fpath)
            masked = overlay_mask_on_image(real_img, mask01)
            save_rgb01(os.path.join(OUT_DIR, f"{tag}_real_mask_overlay_{i}.png"), masked)


if __name__ == "__main__":
    save_sequence("A", FAKE_A_START, save_real=True, save_lime=True, save_mask=True)
    save_sequence("B", FAKE_B_START, save_real=False, save_lime=False, save_mask=False)

    print(f"\n✅ All images saved to: {OUT_DIR}")
