import os
import json
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm 

def fname_from_path(p):
    """Convert a file path to a safe file name."""
    return p.replace("/", "_").replace("\\", "_")

def build_diff_mask(fake_bgr, orig_bgr, eps=0, open_kernel=0):
    """Generate a 0/1 mask based on pixel differences."""
    if eps <= 0:
        m = (fake_bgr != orig_bgr).any(axis=2).astype(np.uint8)
    else:
        diff = np.abs(fake_bgr.astype(np.int16) - orig_bgr.astype(np.int16))
        m = (diff > eps).any(axis=2).astype(np.uint8)
    if open_kernel > 0:
        k = np.ones((open_kernel, open_kernel), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    return m

def main():
    ap = argparse.ArgumentParser(description="Build pixel-diff masks from test→origin mapping.")
    ap.add_argument("--dataset", default="FF-DF")
    ap.add_argument("--eps", type=int, default=5, help="Pixel tolerance (0 = strict match)")
    ap.add_argument("--open_kernel", type=int, default=3, help="Morphological open kernel size (0 = off)")
    ap.add_argument("--mapping_json", default="output/json",
                    help="Path to json file")
    ap.add_argument("--out_root", default="masks",
                    help="Output root dir for masks")
    args = ap.parse_args()

    mapping_json = Path(f"{args.mapping_json}/{args.dataset}/pt2orig.json")

    out_dir = (Path.cwd() / args.out_root / args.dataset).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(mapping_json):
        raise FileNotFoundError(f"Mapping file not found: {mapping_json}")

    with open(mapping_json, "r") as f:
        test2orig = json.load(f)

    pt2mask = {}

    saved = skipped = 0
    for fake_path, orig_path in tqdm(test2orig.items(), desc=f"Building masks for {args.dataset}"):
        pt_name = fname_from_path(fake_path) + ".pt"
        if not orig_path:
            pt2mask[pt_name] = None
            continue
        if not (os.path.exists(fake_path) and os.path.exists(orig_path)):
            skipped += 1
            continue

        fake_img = cv2.imread(fake_path, cv2.IMREAD_COLOR)
        orig_img = cv2.imread(orig_path, cv2.IMREAD_COLOR)
        if fake_img is None or orig_img is None:
            skipped += 1
            continue
        
        if fake_img.shape != orig_img.shape:
            raise ValueError(f"Image size mismatch: {fake_path} ({fake_img.shape}) vs {orig_path} ({orig_img.shape})")

        mask01 = build_diff_mask(fake_img, orig_img, eps=args.eps, open_kernel=args.open_kernel)
        out_path = out_dir / (fname_from_path(fake_path) + ".png")
        cv2.imwrite(str(out_path), mask01 * 255)
        saved += 1

        pt2mask[pt_name] = str(out_path)

    json_out_path = Path(args.mapping_json)/ args.dataset / "pt2mask.json"
    with open(json_out_path, "w") as f:
        json.dump(pt2mask, f, indent=2)

    print(f"[{args.dataset}] Done. Saved {saved} masks, Skipped {skipped}.")
    print(f"pt2mask.json saved to: {json_out_path}")

if __name__ == "__main__":
    main()
