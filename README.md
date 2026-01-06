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


---
---




# GraspSAM
Sangjun Noh, Jongwon Kim, Dongwoo Nam, Seunghyeok Back, Raeyong kang, Kyoobin Lee

This repository contains source codes for the paper "GraspSAM: When Segment Anything Model meets Grasp Detection." (ICRA 2025)
[[ArXiv]](https://arxiv.org/abs/2409.12521) [[Project Website]](https://gistailab.github.io/graspsam/)


## Getting Started

### Environment Setup

Tested on Titan RTX with python 3.8x, pytorch 2.0.1, torchvision 0.15.2, CUDA 11.7 

1. Download source codes
```
git clone https://github.com/gist-ailab/GraspSAM.git
cd GraspSAM
```

2. Set up a python environment
```
conda create -n GraspSAM python=3.8
conda activate GraspSAM
pip install -r requirements.txt
```

### Download grasp detection benchmarks
1. Download `Jacquard` dataset at [[Jacquard]](https://jacquard.liris.cnrs.fr/)
2. Download `Grasp-Anything` dataset at [[Grasp-Anything]](https://github.com/Fsoft-AIC/Grasp-Anything)
3. Extract the downloaded datasets and organize the folders as follows
```
GraspSAM
└── datasets
       ├── Jacqurd_Dataset
       │     └──Jacquard_Dataset_0
       │     └──...
       │     └──Jacquard_Dataset_11
       └── Grasp-Anything
             └──grasp_label_positive
             └──grasp_label_negative
             └──image
             └──mask
             └──scene_description
       
```
### Download pretrained checkpoints for SAM families 
1. Download `Efficient SAM` checkpoint at [Efficient SAM](https://github.com/yformer/EfficientSAM.git)
2. Download `Mobile SAM` checkpoint at [Mobile SAM](https://github.com/ChaoningZhang/MobileSAM.git)
3. Make the pretrained_checkpoint folder and move the downloaded checkpoints to the folder
```
GraspSAM
└── pretrained_checkpoint
       ├── mobile_sam.pt
       ├── efficient_sam
             └──efficient_sma_vitt.pt
             └──... 
```

## Train & Evaluation
### Train on Jacquard
```
python train.py --root {JACQUARD_ROOT} --save --sam-encoder-type {BACKBONE_TYPE}
```

### Train on Grasp-Anything
```
python train.py --root {GRASP_ANYTHING_ROOT} --save --sam-encoder-type {BACKBONE_TYPE}
```

### Evaluation on Jacquard

```
python eval.py --root {JACQUARD_ROOT} --ckp_path {CKP_PATH}
```

### Evaluation on Grasp-Anything
```
python eval.py --root {GRASP_ANYTHING_ROOT} --ckp_path {CKP_PATH}
```


## License

The source code of this repository is released only for academic use. See the [license](./LICENSE.md) file for details.

## Notes

The codes of this repository are built upon the following open sources. Thanks to the authors for sharing the code!
- SAM families : [Efficient SAM](https://github.com/yformer/EfficientSAM.git) and [Mobile SAM](https://github.com/ChaoningZhang/MobileSAM.git) 
- Adapter for using SAM encoder [Rein](https://github.com/w1oves/Rein.git)
- Learnable tokens are modified from [sam-hq](https://github.com/SysCV/sam-hq.git)
- Grasp detection benchmarks : [[Jacquard]](https://jacquard.liris.cnrs.fr/) and [[Grasp-Anything]](https://github.com/Fsoft-AIC/Grasp-Anything)



## Citation
If you use our work in a research project, please cite our work:
```@article{noh2024graspsam,
  title={GraspSAM: When Segment Anything Model Meets Grasp Detection},
  author={Noh, Sangjun and Kim, Jongwon and Nam, Dongwoo and Back, Seunghyeok and Kang, Raeyoung and Lee, Kyoobin},
  journal={arXiv preprint arXiv:2409.12521},
  year={2024}
}
```
