#!/usr/bin/env python3
"""
Convert an RGB image, a depth image (usually uint16 PNG), and a UOC instance-label
image (.npy) into a Jacquard-like sample folder.

Main output folder:
  <sample_id>_RGB.png
  <sample_id>_mask.png                  union mask, binary 0/255
  <sample_id>_perfect_depth.tiff         float32 meters
  <sample_id>_mask_instance_<id>.png     one binary mask per UOC instance id
  <sample_id>_meta.json

Optional per-instance subfolders:
  mask_<id>/
    <sample_id>_RGB.png
    <sample_id>_mask.png                 this instance only, renamed as default mask
    <sample_id>_perfect_depth.tiff
    <sample_id>_meta.json
    optionally <sample_id>_grasps.txt    dummy, for older eval.py only

Recommended newer workflow:
  Use this script only to generate the top-level files, then run eval.py with:
      --custom_no_gt 1 --mask-id <id>

Backward-compatible workflow:
  Add --make_instance_dirs 1 --write_dummy_grasps 1, then run old eval.py on mask_<id>/.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def expand_path(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def to_square_and_resize(arr: np.ndarray, out_size: int = 1024, is_mask: bool = False) -> np.ndarray:
    if arr.ndim == 2:
        h, w = arr.shape
    else:
        h, w = arr.shape[:2]

    side = max(h, w)
    pad_y1 = (side - h) // 2
    pad_y2 = side - h - pad_y1
    pad_x1 = (side - w) // 2
    pad_x2 = side - w - pad_x1

    if arr.ndim == 2:
        padded = np.pad(arr, ((pad_y1, pad_y2), (pad_x1, pad_x2)), mode="constant", constant_values=0)
        img = Image.fromarray(padded)
        resample = Image.NEAREST if is_mask else Image.BILINEAR
        img = img.resize((out_size, out_size), resample=resample)
        return np.array(img)

    padded = np.pad(arr, ((pad_y1, pad_y2), (pad_x1, pad_x2), (0, 0)), mode="constant", constant_values=0)
    img = Image.fromarray(padded)
    img = img.resize((out_size, out_size), resample=Image.BILINEAR)
    return np.array(img)


def write_dummy_jacquard_grasps(path: str, out_size: int = 1024):
    """
    Write a harmless dummy Jacquard-style grasp file for backward compatibility.

    Format used in your data examples:
      x;y;theta_deg;h;w

    This is NOT meaningful ground truth; it is only for old dataset code that insists
    on discovering samples from *_grasps.txt.
    """
    cx = out_size / 2.0
    cy = out_size / 2.0
    with open(path, "w") as f:
        f.write(f"{cx:.4f};{cy:.4f};0.0;20.0;40.0\n")


def save_base_files(out_dir, sample_id, rgb_out, union_out, depth_out):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_rgb_path = out_dir / f"{sample_id}_RGB.png"
    out_mask_path = out_dir / f"{sample_id}_mask.png"
    out_depth_path = out_dir / f"{sample_id}_perfect_depth.tiff"

    Image.fromarray(rgb_out).save(out_rgb_path)
    Image.fromarray(union_out).convert("L").save(out_mask_path)
    Image.fromarray(depth_out.astype(np.float32), mode="F").save(out_depth_path, compression="raw")

    return out_rgb_path, out_mask_path, out_depth_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb", required=True, help="RGB image path (png/jpg)")
    ap.add_argument("--depth", required=True, help="Depth image path, usually uint16 PNG")
    ap.add_argument("--labels", required=True, help="Instance label path (.npy), values 0..N")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--sample_id", default="0_from_rgbd", help="Output sample id prefix")
    ap.add_argument("--out_size", type=int, default=1024, help="Output square size")
    ap.add_argument("--depth_scale_to_m", type=float, default=0.001, help="Meters per input depth unit")

    ap.add_argument("--make_instance_dirs", type=int, default=1, choices=[0, 1],
                    help="If 1, also create mask_<id>/ subfolders for each instance")
    ap.add_argument("--write_dummy_grasps", type=int, default=0, choices=[0, 1],
                    help="If 1, write dummy *_grasps.txt files for old eval.py compatibility")

    args = ap.parse_args()

    rgb_path = expand_path(args.rgb)
    depth_path = expand_path(args.depth)
    labels_path = expand_path(args.labels)
    out_dir = expand_path(args.out_dir)

    os.makedirs(out_dir, exist_ok=True)

    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    depth_raw = np.array(Image.open(depth_path))
    labels = np.load(labels_path)

    if labels.shape != depth_raw.shape or labels.shape != rgb.shape[:2]:
        raise ValueError(f"Shape mismatch: rgb {rgb.shape[:2]}, depth {depth_raw.shape}, labels {labels.shape}")

    instance_ids = [int(x) for x in np.unique(labels) if int(x) != 0]
    union = (labels != 0).astype(np.uint8) * 255
    per_instance = {iid: (labels == iid).astype(np.uint8) * 255 for iid in instance_ids}

    depth_m = depth_raw.astype(np.float32) * float(args.depth_scale_to_m)

    rgb_out = to_square_and_resize(rgb, out_size=args.out_size, is_mask=False)
    union_out = to_square_and_resize(union, out_size=args.out_size, is_mask=True)
    depth_out = to_square_and_resize(depth_m, out_size=args.out_size, is_mask=False).astype(np.float32)
    per_instance_out = {
        iid: to_square_and_resize(mask, out_size=args.out_size, is_mask=True)
        for iid, mask in per_instance.items()
    }

    out_rgb_path, out_mask_path, out_depth_path = save_base_files(
        out_dir, args.sample_id, rgb_out, union_out, depth_out
    )

    instance_files = []
    instance_dir_names = []

    # Save top-level instance masks
    for iid, mask_out in per_instance_out.items():
        p = Path(out_dir) / f"{args.sample_id}_mask_instance_{iid}.png"
        Image.fromarray(mask_out).convert("L").save(p)
        instance_files.append(p.name)

    # Optional backward-compatible mask_<id>/ folders
    if args.make_instance_dirs == 1:
        for iid, mask_out in per_instance_out.items():
            subdir = Path(out_dir) / f"mask_{iid}"
            subdir.mkdir(parents=True, exist_ok=True)

            # same RGB/depth
            Image.fromarray(rgb_out).save(subdir / f"{args.sample_id}_RGB.png")
            Image.fromarray(depth_out.astype(np.float32), mode="F").save(
                subdir / f"{args.sample_id}_perfect_depth.tiff",
                compression="raw",
            )

            # selected instance mask renamed to the default Jacquard mask filename
            Image.fromarray(mask_out).convert("L").save(subdir / f"{args.sample_id}_mask.png")

            if args.write_dummy_grasps == 1:
                write_dummy_jacquard_grasps(subdir / f"{args.sample_id}_grasps.txt", out_size=args.out_size)

            instance_dir_names.append(subdir.name)

    if args.write_dummy_grasps == 1:
        write_dummy_jacquard_grasps(Path(out_dir) / f"{args.sample_id}_grasps.txt", out_size=args.out_size)

    areas = {int(iid): int(np.sum(labels == iid)) for iid in instance_ids}

    meta = {
        "sample_id": args.sample_id,
        "input": {
            "rgb": os.path.abspath(rgb_path),
            "depth": os.path.abspath(depth_path),
            "labels": os.path.abspath(labels_path),
        },
        "output": {
            "RGB": Path(out_rgb_path).name,
            "mask_union": Path(out_mask_path).name,
            "perfect_depth_tiff": Path(out_depth_path).name,
            "mask_instances": instance_files,
            "instance_dirs": instance_dir_names,
            "dummy_grasps_written": bool(args.write_dummy_grasps),
        },
        "instance_ids": instance_ids,
        "instance_pixel_areas_input_res": areas,
        "conversion": {
            "input_resolution": [int(rgb.shape[1]), int(rgb.shape[0])],
            "output_resolution": [int(args.out_size), int(args.out_size)],
            "resize_method": "pad-to-square then resize",
            "depth_scale_to_m": float(args.depth_scale_to_m),
        },
        "recommended_eval_command": (
            "python eval.py --root <this_out_dir> --custom_no_gt 1 --mask-id <instance_id> "
            "--ckp_path <checkpoint> --sam-encoder-type vit_t --no-grasps 5 "
            "--remove_background 0 --apply_mask_to_q 1"
        ),
    }

    meta_path = Path(out_dir) / f"{args.sample_id}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Copy meta into each instance subdir with selected_instance_id for clarity.
    if args.make_instance_dirs == 1:
        for iid in instance_ids:
            sub_meta = dict(meta)
            sub_meta["selected_instance_id"] = iid
            sub_meta["output"] = dict(meta["output"])
            sub_meta["output"]["mask_union"] = f"{args.sample_id}_mask.png"
            with open(Path(out_dir) / f"mask_{iid}" / f"{args.sample_id}_meta.json", "w") as f:
                json.dump(sub_meta, f, indent=2)

    print("Wrote base Jacquard-like files:")
    print(" ", out_rgb_path)
    print(" ", out_mask_path)
    print(" ", out_depth_path)
    print(" ", meta_path)
    for fn in instance_files:
        print(" ", os.path.join(out_dir, fn))
    if instance_dir_names:
        print("Wrote instance subfolders:")
        for d in instance_dir_names:
            print(" ", os.path.join(out_dir, d))


if __name__ == "__main__":
    main()
