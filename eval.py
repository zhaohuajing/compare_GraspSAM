import os
import time
import argparse
import warnings

warnings.filterwarnings(action='ignore')
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.jacquard_data import JacquardDataset
from data.grasp_anything_data import GraspAnythingDataset

from model.planar_grasp_sam import PlanarGraspSAM

import matplotlib.pyplot as plt
from skimage.filters import gaussian
from data.utils.grasp_utils import *

from data.utils.grasp_utils import GraspRectangles, detect_grasps
import matplotlib.pyplot as plt

from data.utils.grasp_pose_convert_utils import (
    CameraIntrinsics,
    rectangle_to_pose_topdown
)



# ------------------------------------------------------------------
# Added: Fix for old Ultralytics checkpoints expecting `ultralytics.yolo`
# ------------------------------------------------------------------
import sys
import types 

try:
    import ultralytics
    import ultralytics.models.yolo
    sys.modules["ultralytics.yolo"] = ultralytics.models.yolo
except Exception as e:
    print("Warning: Ultralytics aliasing failed:", e)


# import random, numpy as np
# random.seed(0)
# np.random.seed(0)
# torch.manual_seed(0)
# torch.cuda.manual_seed_all(0)

import cv2 

# ------------------------------------------------------------------
# Added part ends
# ------------------------------------------------------------------

# sample command: (GraspSAM) root@1ea777512d03:~/graspnet_ws/src/graspsam_ros2/compare_GraspSAM# 
# python eval.py --dataset_name from_rgbd --ckp_path ./pretrained_checkpoint/sam_vit_b_01ec64.pth   --sam-encoder-type vit_b


def post_process_output(q_img, cos_img, sin_img, width_img):
    """
    Post-process the raw output of the GG-CNN, convert to numpy arrays, apply filtering.
    :param q_img: Q output of GG-CNN (as torch Tensors)
    :param cos_img: cos output of GG-CNN
    :param sin_img: sin output of GG-CNN
    :param width_img: Width output of GG-CNN
    :return: Filtered Q output, Filtered Angle output, Filtered Width output
    """
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

    width_scale = 512 # default : 512
    q_img = q_img.data.detach().cpu().numpy().squeeze()
    ang_img = (torch.atan2(sin_img, cos_img) / 2.0).data.detach().cpu().numpy().squeeze()
    width_img = width_img.data.detach().cpu().numpy().squeeze() * width_scale

    q_img = gaussian(q_img, 2.0, preserve_range=True)
    ang_img = gaussian(ang_img, 2.0, preserve_range=True)
    width_img = gaussian(width_img, 1.0, preserve_range=True)

    return q_img, ang_img, width_img

def calculate_iou_match(grasp_q, grasp_angle, ground_truth_bbs, no_grasps=1, grasp_width=None, gs=None): #added gs input
    """
    Calculate grasp success using the IoU (Jacquard) metric (e.g. in https://arxiv.org/abs/1301.3592)
    A success is counted if grasp rectangle has a 25% IoU with a ground truth, and is withing 30 degrees.
    :param grasp_q: Q outputs of GG-CNN (Nx300x300x3)
    :param grasp_angle: Angle outputs of GG-CNN
    :param ground_truth_bbs: Corresponding ground-truth BoundingBoxes
    :param no_grasps: Maximum number of grasps to consider per image.
    :param grasp_width: (optional) Width output from GG-CNN
    :return: success
    """

    if not isinstance(ground_truth_bbs, GraspRectangles):
        gt_bbs = GraspRectangles.load_from_array(ground_truth_bbs)
    else:
        gt_bbs = ground_truth_bbs


    if gs == None:
        gs = detect_grasps(grasp_q, grasp_angle, width_img=grasp_width, no_grasps=no_grasps)

    for g in gs:
        if g.max_iou(gt_bbs) > 0.25:
            return True
    else:
        return False


def setup_model(model_type, sam_encoder_type):
    if model_type == "bs_grasp_sam":
        model = PlanarGraspSAM(sam_encoder_type=sam_encoder_type)

    else:
        raise("please input correct model type")

    return model



#-----------------------------
# Add function for data saving
#----------------------------

def grasp_to_dict(g):
    """
    Convert a GraspSAM grasp (object OR tensor/array) into JSON-safe dict
    """
    # Case 1: Grasp object
    if hasattr(g, "center"):
        return {
            "x": float(g.center[0]),
            "y": float(g.center[1]),
            "angle": float(g.angle),
            "width": float(g.width),
            "score": float(getattr(g, "score", 0.0)),
        }

    # Case 2: Tensor or ndarray [x, y, angle, width, score?]
    import numpy as np
    if hasattr(g, "detach"):
        g = g.detach().cpu().numpy()

    if isinstance(g, (list, tuple, np.ndarray)):
        return {
            "x": float(g[0]),
            "y": float(g[1]),
            "angle": float(g[2]),
            "width": float(g[3]),
            "score": float(g[4]) if len(g) > 4 else 0.0,
        }

    raise TypeError(f"Unsupported grasp type: {type(g)}")

#-----------------------------
# Add function for extracting bounding box from instance mask
#----------------------------

def bbox_from_mask(mask, pad=10):
    """
    mask: HxW, nonzero = object
    pad: pixels of padding around bbox
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(mask.shape[1] - 1, x_max + pad)
    y_max = min(mask.shape[0] - 1, y_max + pad)

    return x_min, y_min, x_max, y_max

#-----------------------------
# Add function for croping RGB, depth, and mask consistently
#----------------------------


def crop_with_bbox(img, bbox):
    """
    img: HxW or HxWxC
    bbox: (x_min, y_min, x_max, y_max)
    """
    x_min, y_min, x_max, y_max = bbox
    return img[y_min:y_max, x_min:x_max]


#-----------------------------
# Added function ends
#----------------------------

def main(args, i=0):
    GPU_NUM = args.gpu_num    
    args.device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')

    #----------------------
    # Added CLI flags
    #----------------------
    # no_grasps = 10
    no_grasps = args.no_grasps
    use_crop = args.use_crop   # <<< toggle here

    #-----------------------
    # Added CameraIntrinsics for later convertion from grasp rectangle to 6D poses
    #-----------------------

    #  RGB-D images in GraspSAM are currently 1024×1024, but the camera info is 640×480: will convert in eval.py

    scale_x = 1024 / 640
    scale_y = 1024 / 480

    intrinsics = CameraIntrinsics(
        fx=554.3827128226441 * scale_x,
        fy=554.3827128226441 * scale_y,
        cx=320.0 * scale_x,
        cy=240.0 * scale_y,
    )


    #-----------------------------
    # Add to save info to log file
    #----------------------------


    # run_id = time.strftime("%Y%m%d_%H%M%S")
    run_id = time.strftime("%Y%m%d") #_%H%M%S")
    out_dir = f"grasp_outputs/run_{run_id}"

    # out_dir = "grasp_outputs/sample4"
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, "note.txt")
    log_f = open(log_path, "a")   # append mode

    log_f.write("\n" + "="*60 + "\n")
    log_f.write(f"New run | encoder={args.sam_encoder_type} | no_grasps={no_grasps}\n")
    log_f.write("="*60 + "\n")

    # Add to save info to log file
    def log_print(*args): 
        msg = " ".join(str(a) for a in args)
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()   # ensure it's written immediately

    #-----------------------------
    # Added part end
    #----------------------------
    
    if args.dataset_name == "jacquard":
        
        train_dataset = JacquardDataset(root=args.root, crop_size=1024, include_mask=True, 
                                        random_rotate=False, random_zoom=False,
                                        start=0.0, end=0.9, seen=True)
        test_dataset = JacquardDataset(root=args.root, crop_size=1024, include_mask=True, 
                                       random_rotate=False, random_zoom=False,   
                                       start=0.9, end=1.0, seen=False)


        # train_dataset = JacquardDataset(root=args.root, crop_size=256, include_mask=True, 
        #                                 random_rotate=False, random_zoom=False,
        #                                 start=0.0, end=0.9, seen=True)
        # test_dataset = JacquardDataset(root=args.root, crop_size=256, include_mask=True, 
        #                                random_rotate=False, random_zoom=False,   
        #                                start=0.9, end=1.0, seen=False)

    elif args.dataset_name == "grasp_anything":
        
        train_dataset = GraspAnythingDataset(root=args.root, include_mask=True, 
                                             random_rotate=False, random_zoom=False,
                                             start=0.0, end=0.9, seen=True)
        test_dataset = GraspAnythingDataset(root=args.root, include_mask=True, 
                                            random_rotate=False, random_zoom=False,
                                            start=0.9, end=1.0, seen=False)

    # Added option to test with local rgbd files

    elif args.dataset_name == "from_rgbd":

        rgbd_root="./datasets/sample_scene_ucn",
        rgbd_pairs = [
            ("./datasets/sample_scene_ucn/from_rgbd-color.png", "./datasets/sample_scene_ucn/from_rgbd-depth.png"),
            # or ("color.png", "0.npy")
        ]

        test_dataset = JacquardDataset(
            root=rgbd_root,
            rgbd_pairs=rgbd_pairs,
            has_gt=False,
            output_size=1024,
            include_mask=True
        )

        depth_available = True
    
    if args.seen_set:
        indices = list(range(test_dataset.__len__()))
        split = int(np.floor(args.split * train_dataset.__len__()))

        test_indices = indices[split:]
        test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)

        # test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=False,
        #                                           num_workers=4, shuffle=False, sampler=test_sampler)

        test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, pin_memory=False,
                                                  num_workers=0, shuffle=False, sampler=test_sampler) # modified num_workers: 4->0
        
        log_print("test_dataset size : {}".format(len(test_indices)))
    else:
        # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=False, 
                                                  # num_workers=4, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, pin_memory=False, 
                                                  num_workers=0, shuffle=False) # modified num_workers: 4->0

    
    # model = setup_model(model_type="bs_grasp_sam", sam_encoder_type="eff_vit_t_w_ad") # commented out to avoid hardcoding encoder type to (non-existing) efficent sam cpk
    # model = setup_model(model_type="bs_grasp_sam",sam_encoder_type="vit_h") #added 
    model = setup_model(model_type="bs_grasp_sam",sam_encoder_type=args.sam_encoder_type)

    model   = model.to(args.device)

    ckp_path = args.ckp_path

    '''
    print("loading checkpoint from : ", ckp_path.split("/")[-1])
    
    # state_dict = torch.load(ckp_path, map_location=args.device) # commented out
    state_dict = torch.load(ckp_path, map_location=args.device, weights_only=True) #added


    if "module." in list(state_dict["model"].keys())[0]:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
                new_state_dict[k[7:]] = v
                
        state_dict = new_state_dict

    model.load_state_dict(state_dict["model"], strict=False) # commented out
    '''

    #--------------------
    # Added check point loading session
    #--------------------

    log_print("loading checkpoint from : ", os.path.basename(ckp_path))

    ckpt = torch.load(ckp_path, map_location="cpu")  # CPU load is safer for big pickles

    # Common patterns:
    # 1) {"model": state_dict, ...}
    # 2) {"state_dict": state_dict, ...}
    # 3) directly a state_dict
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        sd = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        sd = ckpt  # might already be a state_dict-like dict
    else:
        raise RuntimeError(f"Unexpected checkpoint type: {type(ckpt)}")

    # Strip DataParallel "module." prefix if present
    if len(sd) > 0:
        k0 = next(iter(sd.keys()))
        if k0.startswith("module."):
            sd = {k[7:]: v for k, v in sd.items()}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    log_print("loaded. missing:", len(missing), "unexpected:", len(unexpected))

    #--------------------
    # Added part ends
    #--------------------

    
    log_print("-"*80) 
     
    ld = len(test_loader)
    results = {"correct": 0, "failed": 0, 
               "g_loss":0, 
               "g_losses":{
                "p_loss": 0,
                "cos_loss": 0,
                "sin_loss": 0,
                "width_loss": 0,
                },}
    
    model.eval()
    with torch.no_grad():
        
        for idx, data in enumerate(test_loader):

            torch.cuda.empty_cache() # temporally added

            images, masks, grasps, didx, rot, zoom_factor = data
            images = images.to(args.device)    
            masks = masks.to(args.device)
            grasps = [g.to(args.device) for g in grasps]
            # grasps = grasps.to(args.device)

            targets = {}
            targets["masks"] = masks
            targets["grasps"] = grasps 
        
            #--------------------------------------------------
            # Added for custom mask loading and image resizing
            #--------------------------------------------------

            # Load instance mask (scene-level)
            # mask_full = np.load("./datasets/sample_scene_ucn/im_label.npy")  # or pass path via args
            # mask_full = (mask_full == 2).astype(np.uint8)  # 2 for top-down gazebo rgbd cylinder; pick instance 1 if only one detected, or check masks if multiple instances detected


            # -------------------------------------------------
            # Load UCN instance mask and select instance 
            # -------------------------------------------------
            mask_np = np.load("./datasets/sample_scene_ucn/im_label.npy")

            # Select object instance = 2
            mask_np = (mask_np == 2).astype(np.float32)

            # print("Mask values:", mask_np[200])
            print("Mask shape:", mask_np.shape)


            # Resize to model resolution (1024×1024)
            mask_np = cv2.resize(
                mask_np,
                (1024, 1024),
                interpolation=cv2.INTER_NEAREST
            )

            # Convert to tensor [B, 1, H, W]
            mask_tensor_full = torch.from_numpy(mask_np)[None, None].to(args.device)

            print("Mask unique values:", torch.unique(mask_tensor_full))
            # print("Mask values:", mask_tensor_full)

            plt.imshow(mask_np, cmap='gray')
            # plt.title("Binary Mask Instance 2")
            plt.show()



            # bbox = bbox_from_mask(mask_full, pad=20) # used 20
            # if bbox is None:
            #     print("No object found in mask, skipping sample")
            #     continue

            # x_min, y_min, x_max, y_max = bbox

            # # Convert RGB tensor to numpy for cropping
            # rgb_full = images[0].permute(1, 2, 0).cpu().numpy()


            rgb_tensor_full = images            # already normalized correctly
            # mask_tensor_full = masks.float()    # ensure float



            # Depth: load explicitly from file
            depth_full = cv2.imread(
                "./datasets/sample_scene_ucn/from_rgbd-depth.png",
                cv2.IMREAD_UNCHANGED
            ).astype(np.float32)

            depth_full = cv2.resize(
                depth_full,
                (1024, 1024),
                interpolation=cv2.INTER_NEAREST
            )
            # If depth is in mm (Gazebo / ROS often is), convert to meters
            if depth_full.max() > 10.0:
                depth_full *= 0.001


            # Ensure depth matches RGB resolution
            H_full, W_full = rgb_tensor_full.shape[2:]
            if depth_full.shape[0] != H_full or depth_full.shape[1] != W_full:
                depth_full = cv2.resize(
                    depth_full,
                    (W_full, H_full),
                    interpolation=cv2.INTER_NEAREST
                )



            if use_crop:
                # Resize crop to model input size
                TARGET_SIZE = 1024 #384

                # rgb_crop = crop_with_bbox(rgb_full, bbox)
                # mask_crop = crop_with_bbox(mask_full, bbox)

                # depth_crop = crop_with_bbox(depth_full, bbox)

                # # Optional: depth crop if you use depth
                # # if depth_available:

            
                # rgb_crop_resized = cv2.resize(
                #     rgb_crop, (TARGET_SIZE, TARGET_SIZE),
                #     interpolation=cv2.INTER_LINEAR
                # )

                # mask_crop_resized = cv2.resize(
                #     mask_crop, (TARGET_SIZE, TARGET_SIZE),
                #     interpolation=cv2.INTER_NEAREST
                # )

                # depth_crop_resized = cv2.resize(
                #     depth_crop, (TARGET_SIZE, TARGET_SIZE),
                #     interpolation=cv2.INTER_NEAREST
                # )

                # # mask_crop = mask_crop.astype(np.float32)
                # # mask_crop = (mask_crop > 0).astype(np.float32)

                # # mask_crop_tensor = torch.from_numpy(mask_crop)[None, None].to(args.device)

                # # rgb_crop_tensor = torch.from_numpy(rgb_crop_resized).float().permute(2, 0, 1)
                # # rgb_crop_tensor = rgb_crop_tensor.unsqueeze(0).to(args.device)

                # rgb_input = rgb_crop_resized
                # mask_input = mask_crop_resized
                # depth_input = depth_crop_resized

                # ---------- find bounding box from mask ----------
                mask_np = mask_tensor_full[0, 0].cpu().numpy()

                ys, xs = np.where(mask_np > 0)

                if len(xs) == 0:
                    print("Warning: empty mask, skipping crop.")
                    rgb_tensor = rgb_tensor_full
                    mask_tensor = mask_tensor_full
                    depth_crop = depth_full
                    x_min = 0
                    y_min = 0
                    scale_x = 1.0
                    scale_y = 1.0
                else:
                    x_min, x_max = xs.min(), xs.max()
                    y_min, y_max = ys.min(), ys.max()

                    # Add margin
                    margin = 20
                    x_min = max(0, x_min - margin)
                    x_max = min(W_full, x_max + margin)
                    y_min = max(0, y_min - margin)
                    y_max = min(H_full, y_max + margin)

                    # ---------- crop tensors ----------
                    rgb_crop = rgb_tensor_full[:, :, y_min:y_max, x_min:x_max]
                    mask_crop = mask_tensor_full[:, :, y_min:y_max, x_min:x_max]
                    depth_crop = depth_full[y_min:y_max, x_min:x_max]

                    # ---------- resize tensors ----------
                    rgb_tensor = F.interpolate(
                        rgb_crop,
                        size=(TARGET_SIZE, TARGET_SIZE),
                        mode='bilinear',
                        align_corners=False
                    )

                    mask_tensor = F.interpolate(
                        mask_crop,
                        size=(TARGET_SIZE, TARGET_SIZE),
                        mode='nearest'
                    )

                    depth_crop = cv2.resize(
                        depth_crop,
                        (TARGET_SIZE, TARGET_SIZE),
                        interpolation=cv2.INTER_NEAREST
                    )

                    scale_x = TARGET_SIZE / (x_max - x_min)
                    scale_y = TARGET_SIZE / (y_max - y_min)

                depth_input = depth_crop


            else:
                # No crop: use full image & full mask
                # x_min, y_min = 0, 0
                # x_max, y_max = mask_full.shape[1], mask_full.shape[0]

                # rgb_input = rgb_full
                # mask_input = mask_full

                depth_input = depth_full

                rgb_tensor = rgb_tensor_full
                mask_tensor = mask_tensor_full
                depth_input = depth_full
                x_min = 0
                y_min = 0
                scale_x = 1.0
                scale_y = 1.0

            # mask_input = mask_input.astype(np.float32)

            # mask_input = (mask_input > 0).astype(np.float32)

            # # rgb_input = rgb_input / 255.0
            # # rgb_input = (rgb_input - 0.5) / 0.5


            # mask_tensor = torch.from_numpy(mask_input)[None, None].to(args.device)
            # rgb_tensor = torch.from_numpy(rgb_input).float().permute(2, 0, 1)
            # rgb_tensor = rgb_tensor.unsqueeze(0).to(args.device)

            targets = {
                "masks": mask_tensor
            }


            #------------------------------
            # Added lines end
            #------------------------------


            if idx == 0: # added for cuda usage inspection
                log_print("Allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
                log_print("Reserved :", torch.cuda.memory_reserved() / 1024**2, "MB")
                log_print("Total    :", torch.cuda.get_device_properties(0).total_memory / 1024**2, "MB")

                # input()
                

            #-------------------------------------
            # Commented out original lines with training loss
            #-------------------------------------
            '''    
            # grasp_pred, mask_pred = model.total_forward(imgs=images, targets=targets)    # commented out
            grasp_pred, mask_pred = model.total_forward(imgs=rgb_tensor,targets=targets)
            #     "masks": mask_crop_tensor   # shape [B, 1, H, W], float {0,1}
            # }) # replaced

            lossd = model.compute_loss(grasp_pred, mask_pred, targets, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0)

            loss = lossd["g_loss"]
            results['g_loss'] += loss.item() / ld
            for ln, l in lossd['g_losses'].items():
                if ln not in results['g_losses']:
                    results['g_losses'][ln] = 0
                results['g_losses'][ln] += l.item() / ld

            q_out, ang_out, w_out = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                        lossd['pred']['sin'], lossd['pred']['width'])

            '''

            #-------------------------------------
            # Added new lines to run inference mode without loss
            #-------------------------------------

            (grasp_pred, mask_pred) = model.total_forward(
                imgs=rgb_tensor,
                targets=targets
            )

            # grasp_pred is a list:
            # [pos, cos, sin, width]
            pos_pred, cos_pred, sin_pred, width_pred = grasp_pred

            q_out, ang_out, w_out = post_process_output(
                pos_pred,
                cos_pred,
                sin_pred,
                width_pred
            )

            #-----------------------------------------
            # Apply binary mask to suppress background [add args control later]
            #---------------------------------------------
            mask_np_resized = mask_tensor[0,0].cpu().numpy()

            q_out *= mask_np_resized
            w_out *= mask_np_resized


            #------------------------------
            # Added lines end
            #------------------------------


            if args.seen_set:
                success = calculate_iou_match(q_out, ang_out, 
                                          train_dataset.get_gtbb(didx, rot, zoom_factor), 
                                          no_grasps=1, 
                                          grasp_width=w_out)
            else:

                #------------------------------
                # Added to force scalar rot/zoom_factor/didx before calling get_gtbb
                #------------------------------

                # ---- make dataloader outputs scalar (batch_size=1 expected) ----
                

                def _scalar(x):
                    # tensor -> python scalar
                    if isinstance(x, torch.Tensor):
                        return x.item() if x.numel() == 1 else x[0].item()
                    # list/tuple -> first element
                    if isinstance(x, (list, tuple)):
                        return _scalar(x[0])
                    # numpy scalar -> python scalar
                    try:
                        import numpy as np
                        if isinstance(x, np.ndarray):
                            return x.item() if x.size == 1 else x.reshape(-1)[0].item()
                    except Exception:
                        pass
                    return x

                didx_s = int(_scalar(didx))
                rot_s = float(_scalar(rot))
                zoom_s = float(_scalar(zoom_factor))

                # then use these
                gtbb = test_dataset.get_gtbb(didx_s, rot_s, zoom_s)


                #------------------------------
                # Add a one-off visualization for one sample
                #------------------------------

                # out_dir = "grasp_outputs"
                os.makedirs(out_dir, exist_ok=True)    


                gs = detect_grasps(q_out, ang_out, width_img=w_out, no_grasps=no_grasps)
                gt = gtbb  # already a GraspRectangles

                #--------------------------
                # Added to map grasps back to full image coordinates
                #--------------------------

                # if use_crop:
                #     scale_x = (x_max - x_min) / TARGET_SIZE
                #     scale_y = (y_max - y_min) / TARGET_SIZE

                #     for g in gs:
                #         cx, cy = g.center
                #         # g.center = (
                #         #     x_min + cx * scale_x,
                #         #     y_min + cy * scale_y
                #         # )
                #         g.center = (
                #             x_min + g.center[0] / scale_x,
                #             y_min + g.center[1] / scale_y
                #         )
                #         g.length *= scale_x
                #         g.width  *= scale_y

                if use_crop and len(gs) > 0:

                    crop_w = (x_max - x_min)
                    crop_h = (y_max - y_min)

                    scale_back_x = crop_w / TARGET_SIZE
                    scale_back_y = crop_h / TARGET_SIZE

                    for g in gs:
                        cx, cy = g.center

                        new_cx = x_min + cx * scale_back_x
                        new_cy = y_min + cy * scale_back_y

                        g.center = (new_cx, new_cy)

                        # length = grasp rectangle length direction
                        g.length *= scale_back_x
                        g.width  *= scale_back_y


                # for g in gs:
                #     g.center[0] = x_min + g.center[0] * scale_x
                #     g.center[1] = y_min + g.center[1] * scale_y
                #     g.length *= scale_x
                #     g.width *= scale_y

                # for g in gs:
                #     cx, cy = g.center

                #     new_cx = x_min + cx * scale_x
                #     new_cy = y_min + cy * scale_y

                #     g.center = (new_cx, new_cy)
                #     g.length = g.length * scale_x
                #     g.width  = g.width  * scale_y

                # optional debug print    
                # if idx == 0: 
                #     for g in gs[:3]:
                #         log_print(
                #             f"Mapped grasp: center=({g.center[0]:.1f}, {g.center[1]:.1f}), "
                #             f"angle={np.degrees(g.angle):.1f} deg, width={g.width:.1f}px"
                #         )

                #--------------------------
                # Added for output inspection
                #--------------------------
                log_print("idx = ", idx)
                log_print("q_out:", type(q_out), q_out.shape, q_out.min(), q_out.max())
                log_print("ang_out:", ang_out.shape,
                          np.degrees(ang_out.min()), np.degrees(ang_out.max()), "deg")

                log_print("Detected grasps:", len(gs))

                #--------------------------
                # Added for output visualization
                #--------------------------

                # added for result inspection
                log_print(f"Detected {len(gs)} grasps:")
                for i, g in enumerate(gs):
                    log_print(f"  Grasp {i}: center=({g.center[0]:.1f}, {g.center[1]:.1f}), "
                          f"angle={np.degrees(g.angle):.1f} deg, width={g.width:.1f}px")

                # --------------------------
                # Visualize grasp rectangles
                # --------------------------

                fig, ax = plt.subplots(1)
                # ax.imshow(images[0].permute(1,2,0).cpu().numpy(), cmap='gray')
                # rgb_vis = rgb_full # images[0].permute(1, 2, 0).cpu().numpy()
                rgb_vis = images[0].permute(1, 2, 0).cpu().numpy()
                # rgb_vis = np.clip(rgb_vis, 0, 1)  # IMPORTANT for imshow

                # If normalized to [-1, 1]
                rgb_vis = (rgb_vis - rgb_vis.min()) / (rgb_vis.max() - rgb_vis.min() + 1e-6)

                # Or if ImageNet normalized, undo normalization explicitly
                rgb_vis = np.clip(rgb_vis, 0.0, 1.0)

                ax.imshow(rgb_vis)


                gt.plot(ax, color='green')
                for g in gs:
                    g.plot(ax, color='red')

                plt.show()
                plt.savefig(os.path.join(out_dir, f"sample_{idx}.png"))
                plt.close()


                fig, ax = plt.subplots(1)
                plt.imshow(mask_np, cmap='gray')
                plt.title("Binary Mask Instance 2")
                plt.show()
                plt.savefig(os.path.join(out_dir, f"sample_{idx}_masks.png"))
                plt.close()


                '''
                # --------------------------
                # Visualize grasp angle map
                # --------------------------

                fig, ax = plt.subplots(1, figsize=(6, 6))
                # im = ax.imshow(
                #     np.degrees(ang_out),
                #     # cmap='Greys', #'hsv',
                #     # vmin=-90,
                #     # vmax=90
                # )

                ax.imshow(rgb_vis)
                im = ax.imshow(np.degrees(ang_out), 
                          cmap='hsv', 
                          alpha=0.5,
                          vmin=-90, vmax=90)



                ax.set_title("Grasp Angle Map (degrees)")
                plt.colorbar(im, ax=ax)
                plt.savefig(os.path.join(out_dir, f"sample_{idx}_angle.png"))
                plt.close()


                '''
                # --------------------------
                # Visualize grasp quality map
                # --------------------------

                fig, ax = plt.subplots(1, figsize=(6, 6))

                # IMPORTANT: clamp visualization range so low values are visible
                # im = ax.imshow(
                #     q_out,
                #     # cmap='Greys', #'jet',
                #     vmin=0.0,
                #     vmax=max(0.05, np.percentile(q_out, 99.5))
                # )

                # ax.imshow(rgb_vis)
                im = ax.imshow(q_out, 
                          cmap='jet', 
                          alpha=0.9,
                          vmin=0.0,
                          vmax=np.percentile(q_out, 99.5),
                          # vmax=1,
                          )


                for g in gs:
                    g.plot(ax, color='white')

                ax.set_title("Grasp Quality Map (q_out)")
                plt.colorbar(im, ax=ax) #, fraction=0.046)

                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"sample_{idx}_qmap.png"))
                plt.close()


                #-----------------------------
                # Add data saving for grasps
                #-----------------------------

                # Save dense grasp maps per sample

                np.savez(
                    os.path.join(out_dir, f"sample_{idx}_maps.npz"),
                    q=q_out,
                    angle=ang_out,
                    width=w_out
                )

                # Convert grasps to a clean array and save

                grasp_array = np.array([
                    [
                        g.center[0],     # x (px)
                        g.center[1],     # y (px)
                        g.angle,         # rad
                        g.width          # px
                    ]
                    for g in gs
                ], dtype=np.float32)


                #  Save grasp as npy and json
                np.save(
                    os.path.join(out_dir, f"sample_{idx}_grasps.npy"),
                    grasp_array
                )

                #  Save as human-readable JSON
                # grasp_list = [
                #     {
                #         "x": float(g.center[0]),
                #         "y": float(g.center[1]),
                #         "angle_rad": float(g.angle),
                #         "angle_deg": float(np.degrees(g.angle)),
                #         "width_px": float(g.width)
                #     }
                #     for g in gs
                # ]

                json_path = os.path.join(out_dir, f"sample_{idx}_grasps.json")

                # with open(os.path.join(out_dir, f"sample_{idx}_grasps.json"), "w") as f:
                #     json.dump(grasp_list, f, indent=2)


                # out_json = {
                #     "sample_id": idx,
                #     "num_grasps": len(grasps),
                #     "grasps": []
                # }

                # for g in grasps:
                #     out_json["grasps"].append([
                #         float(g.center[0]),
                #         float(g.center[1]),
                #         float(g.angle),
                #         float(g.width),
                #         float(g.score)
                #     ])

                # with open(json_path, "w") as f:
                #     json.dump(out_json, f, indent=2)

                grasp_dicts = []
                for g in gs:
                    grasp_dicts.append(grasp_to_dict(g))

                with open(json_path, "w") as f:
                    json.dump(grasp_dicts, f, indent=2)


                # -----------------------------
                # Convert grasp rectangle to 6D poses
                # -----------------------------

                for g in gs:
                    try:
                        pos, quat, width_m = rectangle_to_pose_topdown(
                            g,
                            depth_input, # depth_image
                            intrinsics,
                            grasp_height_offset=0.01,  # optional 1cm lift
                        )

                        # Metric width filtering (recommended)
                        if not (0.02 <= width_m <= 0.08):
                            continue

                        log_print(
                            f"6D grasp: pos={pos}, "
                            f"yaw={np.rad2deg(g.angle):.1f}°, "
                            f"width={width_m:.3f} m"
                        )

                    except ValueError as e:
                        continue


                #------------------------------
                # Added part ends
                #------------------------------

                # if args.dataset_name == "jacquard" or args.dataset_name == "grasp_anything":

                success = calculate_iou_match(q_out, ang_out, 
                                              # test_dataset.get_gtbb(didx, rot, zoom_factor), 
                                              gtbb,
                                              # no_grasps=1, 
                                              no_grasps=no_grasps, 
                                              grasp_width=w_out,
                                              gs=gs)
            
            if success:
                results["correct"] += 1
            else:
                results["failed"] += 1
            
            success_rate = 100 * results["correct"] / (results["correct"] + results["failed"])
            
            
            log_print("success rate : {:.2f}% | correct : {},  failed : {}".format(success_rate, results["correct"], results["failed"]))

    log_f.close()


    return 100 * results["correct"] / (results["correct"] + results["failed"])



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--gpu-num", type=int, default=0, help="gpu id number")
    parser.add_argument("--seen-set", action="store_true", help="seen set")
    parser.add_argument("--dataset_name", type=str, default="jacquard", help="dataset name")
    parser.add_argument("--batch-size", type=int, default=1)
    
    parser.add_argument("--split", type=float, default=0.01)
    parser.add_argument("--root", type=str, help="dataset root")
    parser.add_argument("--ckp_path", type=str, help="ckp_path")
    parser.add_argument("--no-grasps", type=int, default=5, help="Top-K grasps to evaluate")
    parser.add_argument("--use_crop", type=bool, default=False, help="Enable mask-based crop before inference")


    # Added to avoid hard-coding encode type
    parser.add_argument(
        "--sam-encoder-type",
        type=str,
        default="vit_h",
        help="SAM backbone type (vit_h, vit_l, vit_b, vit_t, eff_vit_t, eff_vit_t_w_ad)"
    )


    args = parser.parse_args()
    exp_name = time.strftime('%c', time.localtime(time.time()))

    main(args)
