
# Towards Continual Test-Time Adaptation of Vision-Language Models in Open-Vocabulary Semantic Segmentation


## Requirements 
- [Python 3.10.13](https://www.python.org/)
- [PyTorch 2.1.2](https://pytorch.org/)
- [MMSegmentation 1.2.2](https://github.com/open-mmlab/mmsegmentation)


## Getting Started
### Step 1: Requirements
To run DAF, please install the following packages, and conda environment:

```bash
conda create -n daf python==3.10.13
conda activate daf
pip install "numpy<2"
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/rocm5.6
pip install -U openmim
mim install -r requirements.txt
```

> **Note:** All experiments were conducted on a single AMD Instinct MI200 GPU with ROCm 5.6.

---
### Step 2: Prepare Datasets

We evaluate DAF on PASCAL VOC 2012, using two class configurations:

- **PASCAL VOC 20** – The 20 foreground categories (background excluded).
- **PASCAL VOC 21** – The 20 foreground categories plus a challenging background label.

Please follow the [MMSeg data preparation document](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md) to download and pre-process the datasets. Please note that we only use the validation split of each dataset.

Additionally, inspired by [ImageNet-C](https://github.com/hendrycks/robustness), we generate 15 corruption types (e.g., noise, blur, weather, compression) *on-the-fly* at test time, allowing us to effectively evaluate each adaptation method's robustness to diverse distribution shifts. 


### SegEarth-OV base model on this branch

This branch uses **SegEarth-OV (CLIP ViT-B/16)** as the shared OVSS base model
for TENT, SAR, CoTTA, MLMP, CLIPArTT, WATT, and DAF/METHOD. The adaptation
algorithms are unchanged; only their common OVSS backbone is replaced.

The SegEarth-OV path includes the reference Q-Q/K-K/V-V attention, frozen JBU
feature upsampling, class-token fusion, and dataset-specific inference settings.
The official `xclip_jbu_one_million_aid.ckpt` is downloaded on first use. To
use an existing local copy instead:

```bash
export SEGEARTH_UPSAMPLER_CKPT=/path/to/xclip_jbu_one_million_aid.ckpt
```

`--token_merge` is intentionally unsupported for SegEarth-OV because JBU
requires the regular ViT patch grid.

### TMPA-compatible remote-sensing evaluation

DAF can directly reuse the remote-sensing dataset layout used by
[dawndaa/TMPA](https://github.com/dawndaa/TMPA). For these dataset IDs, DAF
follows the TMPA preprocessing/evaluation protocol rather than DAF's generic
dataset geometry:

- RGB input, matching TMPA's `PIL.convert("RGB")`
- square resize to `448 x 448` by default
- sliding-window crops of `224 x 224` with stride `112`
- CLIP normalization equivalent to TMPA's torchvision normalization
- deterministic file order
- prediction restored to the original image resolution before metric computation

Supported TMPA dataset IDs are:
`openearthmap`, `loveda`, `isaid`, `potsdam`, `uavid`, `udd5`,
`vaihingen`, `vdd`, `whu_aerial`, `whu_sat`, `inria`, `xbd`,
`chn6-cug`, `deepglobe`, `massachusetts`, `spacenet`, and `wbs_si`.

The 15 standard ImageNet-C corruptions can be selected with the shortcut
`--corruptions_list imagenet_c`. They are applied on-the-fly in RGB space.
Use `--corruption_severity 1..5` to control severity.

Example SAR continual evaluation on LoveDA:

```bash
python main.py \
  --adapt \
  --method sar \
  --reset_mode continual \
  --dataset loveda \
  --data_dir /path/to/TMPA/data_root \
  --corruptions_list imagenet_c \
  --corruption_severity 5 \
  --tmpa_resolution 448 \
  --tmpa_crop_size 224 \
  --tmpa_crop_stride 112 \
  --ovss_type segearth \
  --ovss_backbone ViT-B/16 \
  --lr 1e-4 \
  --steps 1 \
  --batch-size 1 \
  --save_dir .save/loveda/sar/
```

For a clean run, use `--corruptions_list original`. The TMPA-specific
`--tmpa_*` options are independent of DAF's generic
`--init_resize/--patch_size/--patch_stride` options.


---
### Step 3: Perform Adaptation

DAF includes multiple adaptation baselines, including:

- **TENT** – Entropy minimization over visual encoder LayerNorm parameters.
- **SAR** – Sharpness-Aware and Reliable entropy minimization using SAM and reliability filtering.
- **CoTTA** – Continual test-time adaptation with an EMA teacher and stochastic restoration.
- **MLMP** – Multi-level multi-prompt optimization with entropy, diversity, and cross-modal anchor consistency losses.

The following example runs MLMP adaptation on PASCAL VOC 20:

```python
python main.py \
    --adapt \
    --method mlmp \
    --loss_ent True --lamb_ent 1.0 \
    --loss_div True --lamb_div 2.0 \
    --loss_cmac True --lamb_cmac 0.5 \
    --module_safs True --alpha_safs 0.5 \
    --prompt_dir prompts.yaml \
    --vision_outputs -1 -2 -3 -4 -5 -6 -7 -8 -9 \
    --alpha_cls 1.0 \
    --ovss_type segearth \
    --ovss_backbone ViT-B/16 \
    --token_merge False --merge_type algm \
    --algm_layers 1 7 --algm_threshold 0.8 --algm_window_size 2 2 \
    --save_dir .save/PascalVOC20Dataset/dafm/ \
    --data_dir /path/to/VOCdevkit/VOC2012/ \
    --dataset PascalVOC20Dataset \
    --workers 4 \
    --init_resize 224 224 \
    --patch_size 224 224 \
    --patch_stride 112 \
    --corruptions_list gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression \
    --lr 1e-3 \
    --optimizer sgd  \
    --steps 1 \
    --batch-size 8 \
    --trials 1 \
    --seed "$SEED" \
    --plot_loss \
    --class_extensions \
    --reset_mode continual \
```

To use TENT instead, change `--method mlmp` to `--method tent` and remove the MLMP-specific arguments (`--vision_outputs`, `--alpha_cls`, `--loss_feat_cons`, `--feat_cons_type`). To evaluate on PASCAL VOC 21, change `--dataset PascalVOC20Dataset` to `--dataset PascalVOC21Dataset` and update `--save_dir` accordingly.

---
## License

This source code is released under the MIT license, which can be found [here](LICENCE). This project integrates elements from the following repositories; we gratefully acknowledge the authors for making their work open-source:
- [MLMP](https://github.com/dosowiechi/MLMP) (MIT licensed)
- [TENT](https://github.com/DequanWang/tent) (MIT licensed)

- [SegEarth-OV](https://github.com/likyoo/SegEarth-OV) (base OVSS model used on `feature/segearth-ov-baselines`)
