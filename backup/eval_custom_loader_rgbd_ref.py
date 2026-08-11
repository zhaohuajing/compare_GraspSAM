#!/usr/bin/env python3
"""eval_custom_jacquard_rgbd.py

Run GraspSAM inference on *Jacquard-like* RGBD+mask samples created from custom RGBD inputs.

Expected files per sample_id (in --root):
  <sample_id>_RGB.png
  <sample_id>_perfect_depth.tiff        (float32 depth in meters)
  <sample_id>_mask.png                  (binary 0/255 union mask)
  <sample_id>_mask_instance_<id>.png    (optional; binary 0/255)

This script supports:
  - --use_crop: crop around the selected mask instance and resize back to 1024
  - --remove_background: zero background pixels in RGB using mask
  - --apply_mask_to_q: multiply q-map by mask before grasp detection

Outputs (in grasp_outputs/run_YYYYMMDD_HHMMSS by default):
  sample_<k>_maps.npz   (q, angle, width)
  sample_<k>_grasps.json
  sample_<k>_output_full.png  (grasps over RGB)
  sample_<k>_mask.png         (mask used)

Notes:
  - This bypasses JacquardDataset and uses the same model forward pass as existing eval scripts.
  - Image normalization defaults to [-1, 1] (img01 -> (img-0.5)/0.5), matching common training pipelines.


Example command:

python3 eval_custom_jacquard_rgbd.py \
  --root ./datasets/sample_scene_ucn/topView/YCB_scene_jaclike \
  --sample_id 0_from_rgbd_demo \
  --ckp_path ./pretrained_checkpoint/mobile_sam.pt \
  --sam-encoder-type vit_t \
  --no-grasps 60 \
  --instance_id 3 \
  --use_crop 1 \
  --remove_background 1 \
  --apply_mask_to_q 1
  
"""

import argparse
import json
import os
import time
import warnings

warnings.filterwarnings(action='ignore')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import numpy as np
import torch
import torch.nn.functional as F

import cv2
from skimage.filters import gaussian

from model.planar_grasp_sam import PlanarGraspSAM
from data.utils.grasp_utils import detect_grasps

from data.utils.grasp_pose_convert_utils import CameraIntrinsics, rectangle_to_pose_topdown

# Ultralytics alias guard (some checkpoints reference ultralytics.yolo)
import sys
try:
    import ultralytics
    import ultralytics.models.yolo
    sys.modules["ultralytics.yolo"] = ultralytics.models.yolo
except Exception as e:
    print("Warning: Ultralytics aliasing failed:", e)

from copy import deepcopy

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, required=True, help='Folder containing <id>_RGB.png/_mask.png/_perfect_depth.tiff')
    ap.add_argument('--sample_id', type=str, default=None, help='Single sample id prefix (without suffix).')
    ap.add_argument('--sample_ids', type=str, default=None, help='Comma-separated sample ids.')

    ap.add_argument('--ckp_path', type=str, required=True)
    ap.add_argument('--sam-encoder-type', dest='sam_encoder_type', type=str, default='vit_t')

    ap.add_argument('--gpu_num', type=int, default=0)
    ap.add_argument('--no-grasps', dest='no_grasps', type=int, default=20)
    ap.add_argument('--q_threshold', type=float, default=0.02)

    ap.add_argument('--instance_id', type=int, default=0, help='0 => use union mask. Otherwise use _mask_instance_<id>.png if present.')
    ap.add_argument('--use_crop', type=int, default=0, help='1 to crop around mask bbox and resize back to 1024')
    ap.add_argument('--crop_margin', type=int, default=20)
    ap.add_argument('--remove_background', type=int, default=0, help='1 to zero RGB background using mask')
    ap.add_argument('--apply_mask_to_q', type=int, default=1, help='1 to multiply q-map by mask before detect_grasps')

    ap.add_argument('--out_dir', type=str, default=None, help='Output directory. Default: grasp_outputs/run_<timestamp>')

    # Intrinsics for converting rectangle->6D
    ap.add_argument('--fx', type=float, default=554.3827128226441)
    ap.add_argument('--fy', type=float, default=554.3827128226441)
    ap.add_argument('--cx', type=float, default=320.0)
    ap.add_argument('--cy', type=float, default=240.0)
    ap.add_argument('--intr_w', type=int, default=640)
    ap.add_argument('--intr_h', type=int, default=480)

    ap.add_argument('--normalize', type=str, default='unit', choices=['minus1_1', 'unit', 'none'],
                    help='RGB normalization: minus1_1 -> (img-0.5)/0.5; unit -> img in [0,1]; none -> uint8/255 then no further.')

    return ap.parse_args()


def setup_model(sam_encoder_type: str, device: torch.device) -> PlanarGraspSAM:
    model = PlanarGraspSAM(sam_encoder_type=sam_encoder_type)
    model = model.to(device)
    return model


def load_checkpoint(model: torch.nn.Module, ckp_path: str):
    print('loading checkpoint from :', os.path.basename(ckp_path))
    ckpt = torch.load(ckp_path, map_location='cpu')

    if isinstance(ckpt, dict) and 'model' in ckpt and isinstance(ckpt['model'], dict):
        sd = ckpt['model']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
        sd = ckpt['state_dict']
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        raise RuntimeError(f'Unexpected checkpoint type: {type(ckpt)}')

    if len(sd) > 0:
        k0 = next(iter(sd.keys()))
        if k0.startswith('module.'):
            sd = {k[7:]: v for k, v in sd.items()}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print('loaded. missing:', len(missing), 'unexpected:', len(unexpected))


def post_process_output(q_img, cos_img, sin_img, width_img):
    # Ensure 1024
    if len(q_img.shape) == 3:
        q_img = F.interpolate(q_img.unsqueeze(0), size=(1024, 1024))[0]
        cos_img = F.interpolate(cos_img.unsqueeze(0), size=(1024, 1024))[0]
        sin_img = F.interpolate(sin_img.unsqueeze(0), size=(1024, 1024))[0]
        width_img = F.interpolate(width_img.unsqueeze(0), size=(1024, 1024))[0]
    elif len(q_img.shape) == 4:
        q_img = F.interpolate(q_img, size=(1024, 1024))[0]
        cos_img = F.interpolate(cos_img, size=(1024, 1024))[0]
        sin_img = F.interpolate(sin_img, size=(1024, 1024))[0]
        width_img = F.interpolate(width_img, size=(1024, 1024))[0]

    width_scale = 512
    q_img = q_img.detach().cpu().numpy().squeeze()
    ang_img = (torch.atan2(sin_img, cos_img) / 2.0).detach().cpu().numpy().squeeze()
    width_img = width_img.detach().cpu().numpy().squeeze() * width_scale

    q_img = gaussian(q_img, 2.0, preserve_range=True)
    ang_img = gaussian(ang_img, 2.0, preserve_range=True)
    width_img = gaussian(width_img, 1.0, preserve_range=True)

    return q_img, ang_img, width_img


def pick_mask_path(root: str, sample_id: str, instance_id: int) -> str:
    if instance_id and instance_id > 0:
        p = os.path.join(root, f'{sample_id}_mask_instance_{instance_id}.png')
        if os.path.exists(p):
            return p
        # fall back to union
    return os.path.join(root, f'{sample_id}_mask.png')


def load_sample(root: str, sample_id: str, instance_id: int):
    rgb_path = os.path.join(root, f'{sample_id}_RGB.png')
    depth_path = os.path.join(root, f'{sample_id}_perfect_depth.tiff')
    mask_path = pick_mask_path(root, sample_id, instance_id)

    if not os.path.exists(rgb_path):
        raise FileNotFoundError(rgb_path)
    if not os.path.exists(depth_path):
        raise FileNotFoundError(depth_path)
    if not os.path.exists(mask_path):
        raise FileNotFoundError(mask_path)

    rgb = cv2.cvtColor(cv2.imread(rgb_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    # TIFF float32 meters
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth = depth.astype(np.float32)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if (not instance_id) or instance_id == 0:
        mask = (mask > 0).astype(np.float32)
    else:
        mask = (mask == instance_id).astype(np.float32)

    # Ensure same size
    H, W = rgb.shape[:2]
    if depth.shape[:2] != (H, W):
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
    if mask.shape[:2] != (H, W):
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.float32)

    return rgb, depth, mask


def normalize_rgb(rgb: np.ndarray, mode: str) -> np.ndarray:
    rgb01 = rgb.astype(np.float32) / 255.0
    if mode == 'unit':
        return rgb01
    if mode == 'minus1_1':
        return (rgb01 - 0.5) / 0.5
    return rgb01


def bbox_from_mask(mask01: np.ndarray, margin: int, H: int, W: int):
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(W, x_max + margin)
    y_max = min(H, y_max + margin)
    return x_min, y_min, x_max, y_max


def apply_crop_and_resize(rgb_norm: np.ndarray, depth: np.ndarray, mask01: np.ndarray, margin: int, target=1024):
    H, W = mask01.shape
    bbox = bbox_from_mask(mask01, margin=margin, H=H, W=W)
    if bbox is None:
        # no crop
        return rgb_norm, depth, mask01, 0, 0, 1.0, 1.0

    x_min, y_min, x_max, y_max = bbox
    rgb_crop = rgb_norm[y_min:y_max, x_min:x_max]
    depth_crop = depth[y_min:y_max, x_min:x_max]
    mask_crop = mask01[y_min:y_max, x_min:x_max]

    crop_h = max(1, y_max - y_min)
    crop_w = max(1, x_max - x_min)

    rgb_rs = cv2.resize(rgb_crop, (target, target), interpolation=cv2.INTER_LINEAR)
    depth_rs = cv2.resize(depth_crop, (target, target), interpolation=cv2.INTER_NEAREST)
    mask_rs = cv2.resize(mask_crop, (target, target), interpolation=cv2.INTER_NEAREST)
    mask_rs = (mask_rs > 0).astype(np.float32)

    scale_x = target / float(crop_w)
    scale_y = target / float(crop_h)

    return rgb_rs, depth_rs, mask_rs, x_min, y_min, scale_x, scale_y


def map_grasps_back(gs, x_min, y_min, scale_x, scale_y):
    # g.center is (y,x)
    for g in gs:
        cy, cx = g.center
        new_cy = y_min + cy / scale_y
        new_cx = x_min + cx / scale_x
        g.center = (new_cy, new_cx)

        # lengths are along image axes in grasp_utils implementation
        if hasattr(g, 'length'):
            g.length /= scale_y
        g.width /= scale_x


def grasp_to_dict_with_pose(g):
    d = {
        # 'x': float(g.center[1]),
        # 'y': float(g.center[0]),
        'x': float(g.center[0]),
        'y': float(g.center[1]),
        'angle': float(g.angle),
        'width_px': float(g.width),
        'length_px': float(getattr(g, 'length', 0.0)),
        'score': float(getattr(g, 'score', 0.0)),
    }
    if getattr(g, 'pos', None) is not None:
        d['pos'] = [float(v) for v in np.asarray(g.pos).reshape(-1)[:3]]
    if getattr(g, 'quat', None) is not None:
        d['quat'] = [float(v) for v in np.asarray(g.quat).reshape(-1)[:4]]
    if getattr(g, 'width_m', None) is not None:
        d['width_m'] = float(g.width_m)
    return d


def main():
    args = parse_args()

    sample_ids = []
    if args.sample_ids:
        sample_ids = [s.strip() for s in args.sample_ids.split(',') if s.strip()]
    elif args.sample_id:
        sample_ids = [args.sample_id]
    else:
        raise SystemExit('Provide --sample_id or --sample_ids')

    device = torch.device(f'cuda:{args.gpu_num}' if torch.cuda.is_available() else 'cpu')

    if args.out_dir is None:
        run_id = time.strftime('%Y%m%d_%H%M%S')
        out_dir = f'grasp_outputs/run_{run_id}'
    else:
        out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Intrinsics must be scaled to current image size
    # The intrinsics provided are for intr_w x intr_h.
    out_size = 1024
    sx = out_size / float(args.intr_w)
    sy = out_size / float(args.intr_h)
    intrinsics = CameraIntrinsics(
        fx=args.fx * sx,
        fy=args.fy * sy,
        cx=args.cx * sx,
        cy=args.cy * sy,
    )

    model = setup_model(args.sam_encoder_type, device)
    load_checkpoint(model, args.ckp_path)
    model.eval()

    for k, sid in enumerate(sample_ids):
        print('-' * 80)
        print('Sample:', sid)
        rgb_u8, depth_m, mask01 = load_sample(args.root, sid, args.instance_id)

        # remove background in *0..1* space (then re-normalize)
        rgb01 = rgb_u8.astype(np.float32) / 255.0
        if args.remove_background:
            rgb01 = rgb01 * mask01[..., None]

        # normalize after background removal
        if args.normalize == 'unit':
            rgb_norm = rgb01
        elif args.normalize == 'minus1_1':
            rgb_norm = (rgb01 - 0.5) / 0.5
        else:
            rgb_norm = rgb01

        x_min = 0
        y_min = 0
        scale_x = 1.0
        scale_y = 1.0

        if args.use_crop:
            rgb_norm, depth_m, mask01, x_min, y_min, scale_x, scale_y = apply_crop_and_resize(
                rgb_norm, depth_m, mask01,
                margin=args.crop_margin,
                target=1024
            )

        # tensors
        img_t = torch.from_numpy(rgb_norm).permute(2, 0, 1).unsqueeze(0).to(device)  # [1,3,H,W]
        mask_t = torch.from_numpy(mask01).unsqueeze(0).unsqueeze(0).to(device)      # [1,1,H,W]

        targets = {'masks': mask_t}
        targets["grasps"] = None

        with torch.no_grad():
            grasp_pred, mask_pred = model.total_forward(imgs=img_t, targets=targets)

        # q_out, ang_out, w_out = post_process_output(
        #     grasp_pred['pos'], grasp_pred['cos'], grasp_pred['sin'], grasp_pred['width']
        # )


        pos_pred, cos_pred, sin_pred, width_pred = grasp_pred
        q_out, ang_out, w_out = post_process_output(
            pos_pred,
            cos_pred,
            sin_pred,
            width_pred
        )


        # Ensure mask aligned to q-map
        mask_for_q = mask01
        if mask_for_q.shape != q_out.shape:
            mask_for_q = cv2.resize(mask_for_q, (q_out.shape[1], q_out.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_for_q = (mask_for_q > 0).astype(np.float32)

        if args.apply_mask_to_q:
            q_used = q_out * mask_for_q
        else:
            q_used = q_out

        # detect grasps
        gs = detect_grasps(q_used, ang_out, width_img=w_out, no_grasps=args.no_grasps)

        # filter by q_threshold
        gs_f = []
        for g in gs:
            cy, cx = g.center
            iy = int(np.clip(cy, 0, q_used.shape[0] - 1))
            ix = int(np.clip(cx, 0, q_used.shape[1] - 1))
            if q_used[iy, ix] >= args.q_threshold:
                gs_f.append(g)
        gs = deepcopy(gs_f)

        # If cropped, map back to full 1024x1024 coordinate system
        if args.use_crop and len(gs) > 0:
            map_grasps_back(gs, x_min=x_min, y_min=y_min, scale_x=scale_x, scale_y=scale_y)

        # Compute 6D poses using *full-resolution* depth
        # If cropped, we need full depth; we can reload full depth & mask (union) to be safe.
        if args.use_crop:
            rgb_u8_full, depth_full, mask_full = load_sample(args.root, sid, args.instance_id)
            depth_for_pose = depth_full
        else:
            depth_for_pose = depth_m

        for g in gs:
            try:
                g.pos, g.quat, g.width_m = rectangle_to_pose_topdown(
                    g,
                    depth_for_pose,
                    intrinsics,
                    grasp_height_offset=0.01,
                )
            except Exception:
                g.pos, g.quat, g.width_m = None, None, None

        # Save maps
        np.savez(
            os.path.join(out_dir, f'sample_{k}_maps.npz'),
            q=q_out,
            angle=ang_out,
            width=w_out,
        )

        # Save grasps json
        grasp_dicts = [grasp_to_dict_with_pose(g) for g in gs]
        with open(os.path.join(out_dir, f'sample_{k}_grasps.json'), 'w') as f:
            json.dump({'sample_id': sid, 'grasps': grasp_dicts}, f, indent=2)

        # Save mask visualization
        cv2.imwrite(os.path.join(out_dir, f'sample_{k}_mask.png'), (mask_for_q * 255).astype(np.uint8))

        # Visualization overlay (full image in 0..1)
        rgb_vis = rgb_u8.astype(np.float32) / 255.0
        import matplotlib.pyplot as plt
        from data.utils.grasp_utils import GraspRectangles

        fig, ax = plt.subplots(1, figsize=(6, 6))
        ax.imshow(rgb_vis)
        for g in gs:
            g.plot(ax, color='red')
        ax.set_title(f'{sid} | grasps={len(gs)}')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'sample_{k}_output_full.png'))
        plt.close(fig)

        # q-map visualization
        fig, ax = plt.subplots(1, figsize=(6, 6))
        ax.imshow(q_used, cmap='jet')
        if args.use_crop:
            for g in gs_f:
                g.plot(ax, color='white')

        else:
            for g in gs:
                g.plot(ax, color='white')
        ax.set_title('Grasp Quality Map (q_used)')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'sample_{k}_qmap.png'))
        plt.close(fig)

        print(f'Output: {out_dir} | grasps: {len(gs)}')


if __name__ == '__main__':
    main()
