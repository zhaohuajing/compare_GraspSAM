# GraspSAM – Modified & Validated Implementation

This repository contains a **working and validated implementation of GraspSAM** for planar grasp detection on the Jacquard dataset, with support for **SAM ViT-B** and **MobileSAM (ViT-T)** backbones.

The codebase has been updated to:
- run on modern PyTorch / CUDA
- support multiple SAM encoder types
- handle Jacquard sample subsets
- run within 6 GB GPU memory (RTX 2060 tested)

---

## Current Status

**Working features**
- End-to-end evaluation on Jacquard samples
- Support for:
  - `vit_b` (official SAM)
  - `vit_t` (MobileSAM, recommended)
- Correct GT grasp parsing and rotation handling
- Top-K grasp evaluation (`no_grasps`)
- Visual verification (GT vs predicted grasps)

**Known limitations**
- Grasp head is **untrained**
- Results are stochastic due to random augmentation
- Metrics reported are **sanity checks**, not final performance

---

## Environment

Tested on:
- Ubuntu 20.04+
- Python 3.8
- PyTorch ≥ 2.0
- CUDA 12.x
- GPU: RTX 2060 (6 GB)

---

## Checkpoints

Download checkpoints manually:

### SAM ViT-B
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth   -O pretrained_checkpoint/sam_vit_b_01ec64.pth
```

### MobileSAM (recommended)
```bash
# example filename
pretrained_checkpoint/mobile_sam.pt
```

---

## Dataset Structure (Jacquard Samples)

Expected structure:
```
Jacquard_Samples/
└── Samples/
    └── <sample_id>/
        ├── *_RGB.png
        ├── *_mask.png
        ├── *_grasps.txt
```

You can evaluate:
- the full `Samples` directory, or
- a single `<sample_id>` folder

---

## Running Evaluation

### MobileSAM (recommended, low memory)

```bash
python eval.py   --root ./datasets/Jacquard_Samples/Samples/<sample_id>/   --ckp_path ./pretrained_checkpoint/mobile_sam.pt   --sam-encoder-type vit_t
```

### SAM ViT-B

```bash
python eval.py   --root ./datasets/Jacquard_Samples/Samples/<sample_id>/   --ckp_path ./pretrained_checkpoint/sam_vit_b_01ec64.pth   --sam-encoder-type vit_b
```

---

## Important Parameters

### `no_grasps`
- Number of top grasp candidates evaluated per image
- This is **Top-K grasp success**, not Top-1

Typical values:
```python
no_grasps = 10   # recommended for sanity checks
no_grasps = 20   # more forgiving
```

A grasp is counted as **successful if ANY of the K candidates**:
- overlaps GT above IoU threshold
- matches angle threshold

---

## Interpreting Results

- Success rates vary between runs due to:
  - random rotation
  - random cropping
  - random zoom
- This is **expected behavior** for an untrained grasp head
- MobileSAM (`vit_t`) gives more stable behavior than ViT-B

---

## Visualization

- **Green**: predicted grasps  
- **Red**: ground-truth grasps  

Visualization confirms:
- coordinate frames are correct
- rotation logic is correct
- evaluation is meaningful

---

## Next Steps

- Train the grasp head
- Add deterministic evaluation mode
- Benchmark MobileSAM vs SAM backbones
- Integrate into COMPARE evaluation pipeline

---

## Notes

This implementation focuses on **correctness and reproducibility**, not final performance.
