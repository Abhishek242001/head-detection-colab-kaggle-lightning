# YOLOv11m Head Detection — Complete Training Guide

> A complete, beginner-friendly guide to training a **person head detector** using YOLOv11m across three different cloud platforms. Every decision, every error, and every fix is documented here so you can replicate it from scratch.

---

## Table of Contents

1. [What Are We Building?](#1-what-are-we-building)
2. [What is YOLO?](#2-what-is-yolo)
3. [Project Structure](#3-project-structure)
4. [System Configurations Tested](#4-system-configurations-tested)
5. [Dataset Setup](#5-dataset-setup)
6. [Platform Guides](#6-platform-guides)
   - [Google Colab](#61-google-colab)
   - [Lightning AI Studio](#62-lightning-ai-studio)
   - [Kaggle](#63-kaggle)
7. [Training Parameters Explained](#7-training-parameters-explained)
8. [Optimization Techniques](#8-optimization-techniques)
9. [Augmentation Techniques](#9-augmentation-techniques)
10. [Exporting to TensorRT (.engine)](#10-exporting-to-tensorrt-engine)
11. [Errors We Faced & How We Fixed Them](#11-errors-we-faced--how-we-fixed-them)
12. [Results & Metrics](#12-results--metrics)
13. [Glossary](#13-glossary)

---

## 1. What Are We Building?

We are building an **object detection model** that can look at an image or video and draw a **bounding box around every person's head** it finds.

Think of it like this: you give the model a photo of a crowd, and it draws little rectangles around every head it can see — even small ones far away in the background.

**Why is this useful?**
- Counting people in a crowd
- Security surveillance systems
- Social distancing monitoring
- Retail foot traffic analysis

The model we use is called **YOLOv11m** (You Only Look Once, version 11, medium size).

---

## 2. What is YOLO?

YOLO is a type of deep learning model designed for **real-time object detection**. The name "You Only Look Once" refers to how it works — unlike older methods that scan an image multiple times, YOLO looks at the whole image just once and predicts all bounding boxes simultaneously.

**Why YOLOv11m specifically?**

The YOLO family has many sizes:

| Model | Size | Speed | Accuracy | Best for |
|-------|------|-------|----------|----------|
| YOLOv11n | Nano | Fastest | Lowest | Edge devices |
| YOLOv11s | Small | Fast | Low | Mobile |
| **YOLOv11m** | **Medium** | **Balanced** | **Good** | **Our choice** |
| YOLOv11l | Large | Slow | High | High accuracy |
| YOLOv11x | XLarge | Slowest | Highest | Research |

We chose **medium** because it gives a good balance between speed and accuracy — fast enough for real-time use, accurate enough to detect small heads in crowds.

---

## 3. Project Structure

```
yolov11m-head-detection/
│
├── train_colab.py          # Training script for Google Colab
├── train_lightning.py      # Training script for Lightning AI Studio
├── train_kaggle.py         # Training script for Kaggle (includes dataset download)
├── README.md               # This file
│
└── Dataset/                # Your dataset (not included — download separately)
    ├── images/
    │   ├── train/          # Training images
    │   └── val/            # Validation images
    ├── labels/
    │   ├── train/          # Training labels (.txt files)
    │   └── val/            # Validation labels (.txt files)
    └── dataset_config.yaml # Dataset configuration file
```

**What is a label file?**
Each image has a matching `.txt` file with one line per object:
```
0 0.512 0.345 0.123 0.098
```
This means: class 0 (person/head), center_x=51.2%, center_y=34.5%, width=12.3%, height=9.8% of image size.

**What is dataset_config.yaml?**
```yaml
train: /path/to/images/train
val:   /path/to/images/val
nc: 1                        # number of classes
names: ['person']            # class names
```

---

## 4. System Configurations Tested

We tested training across three different platforms. Here is exactly what we used and what happened:

### Google Colab (Free Tier)
```
GPU     : Tesla T4
VRAM    : 15 GB
Python  : 3.10
CUDA    : 11.8
PyTorch : 2.x
Status  : ✅ Works — use imgsz=640, batch=16
Problem : Session disconnects after ~12 hours
```

### Google Colab Pro
```
GPU     : NVIDIA A100
VRAM    : 40 GB
Status  : ✅ Works — use imgsz=1280, batch=32
```

### Lightning AI Studio ⭐ Recommended
```
GPU     : NVIDIA L40S
VRAM    : 47.7 GB
Python  : 3.12
CUDA    : 12.8
PyTorch : 2.8.0+cu128
Status  : ✅ Best option — persistent sessions, no disconnects
imgsz   : 1280
batch   : 32
```

### Kaggle (Free)
```
GPU     : Tesla T4  (use this — NOT P100)
VRAM    : 15.6 GB
Python  : 3.12
CUDA    : 12.8
PyTorch : 2.10.0+cu128
Status  : ✅ Works on T4 — use imgsz=640, batch=16

⚠️  P100 WARNING:
GPU     : Tesla P100  ← DO NOT USE
VRAM    : 16 GB
Status  : ❌ BROKEN — P100 is sm_60, PyTorch 2.x needs sm_70+
Fix     : Settings → Accelerator → Switch to "GPU T4 x2"
```

---

## 5. Dataset Setup

### Dataset structure
Your dataset must follow this structure:
```
Dataset/
├── images/
│   ├── train/   ← put training images here (.jpg, .png)
│   └── val/     ← put validation images here
├── labels/
│   ├── train/   ← YOLO format .txt files (same filename as images)
│   └── val/
└── dataset_config.yaml
```

### Downloading from MEGA (our dataset source)
Standard `mega.py` library is broken on Python 3.12 due to a `tenacity` dependency issue and does not support new MEGA URL formats. We wrote a custom downloader using `pycryptodome` that:
1. Parses the file ID and AES key from the URL
2. Fetches the encrypted file from MEGA's CDN
3. Decrypts it on-the-fly using AES-CTR mode
4. Saves the readable file to disk

```python
# Install dependency
pip install pycryptodome requests

# The download function handles everything:
# - URL parsing
# - AES key decoding
# - Streaming download
# - On-the-fly decryption
# - Progress bar
# See train_kaggle.py for the complete mega_download() function
```

### Update dataset_config.yaml paths
After extracting, update the yaml to point to your actual paths:

**Google Colab:**
```yaml
train: /content/drive/MyDrive/Dataset/images/train
val:   /content/drive/MyDrive/Dataset/images/val
```

**Lightning AI:**
```yaml
train: /home/zeus/Dataset/Dataset/images/train
val:   /home/zeus/Dataset/Dataset/images/val
```

**Kaggle:**
```yaml
train: /kaggle/working/Dataset/Dataset/images/train
val:   /kaggle/working/Dataset/Dataset/images/val
```

---

## 6. Platform Guides

### 6.1 Google Colab

**Setup steps:**
```python
# Cell 1 — Install
!pip install ultralytics ensemble-boxes -q

# Cell 2 — Mount Google Drive (IMPORTANT — saves results persistently)
from google.colab import drive
drive.mount('/content/drive')

# Cell 3 — Run training
# Run train_colab.py
```

**Important tips for Colab:**
- Always **save outputs to Google Drive** (`/content/drive/MyDrive/`) not `/content/` — Colab resets `/content/` when the session ends
- Free Colab disconnects after ~12 hours — use `save_period=10` to checkpoint every 10 epochs so you don't lose progress
- If you get disconnected, you can resume from the last checkpoint:
```python
model = YOLO("path/to/last.pt")   # load last checkpoint
model.train(resume=True)           # resume training
```
- Use `cache=False` to avoid running out of RAM (Colab gives ~12GB RAM)

**Recommended settings for free T4:**
```python
IMGSZ = 640    # 1280 will OOM on T4
BATCH = 16
```

---

### 6.2 Lightning AI Studio

Lightning AI gives you a **persistent cloud VM** — unlike Colab it does not disconnect and your files stay between sessions.

**Setup steps:**
```bash
# Terminal or notebook cell
pip install ultralytics ensemble-boxes pycryptodome -q
```

**Download dataset from MEGA:**
```python
# Run the mega_download() function from train_lightning.py
# Dataset saves to /home/zeus/Dataset/
```

**Key advantage over Colab:** The L40S GPU has 47.7GB VRAM — you can train at full `imgsz=1280` with `batch=32`.

**OOM issue we encountered:**
Even on 47.7GB VRAM, we got Out of Memory errors. The cause was `multi_scale=True` which randomly upscales images to `1280 × 1.5 = 1920px` during training, causing sudden VRAM spikes.

**Fix:** Set `multi_scale=False`.

```python
# ❌ Caused OOM even on 47GB VRAM
multi_scale = True

# ✅ Fixed
multi_scale = False
```

**Recommended settings for L40S:**
```python
IMGSZ = 1280
BATCH = 32
workers = 8    # L40S has many CPU cores
```

---

### 6.3 Kaggle

Kaggle gives **free GPU access** with 30 hours/week GPU quota.

**Step 1 — Switch to T4 GPU (CRITICAL):**
```
Kaggle Notebook → Settings (right sidebar) → Accelerator → GPU T4 x2
```

> ⚠️ **P100 is completely broken** for modern PyTorch. All available PyTorch versions (2.2+) require CUDA capability sm_70 or higher. P100 only has sm_60. There is no fix — you must switch to T4.

**Step 2 — Install:**
```python
!pip install ultralytics ensemble-boxes pycryptodome -q
```

**Step 3 — Run train_kaggle.py**
This single file handles everything:
- Downloads dataset from MEGA
- Decrypts and extracts it
- Updates yaml paths
- Trains the model
- Validates with TTA
- Copies `best.pt` to the output tab

**Kaggle-specific settings:**
```python
workers = 2      # Kaggle has limited CPU (2-4 cores)
cache   = False  # Kaggle RAM is limited (~13GB)
IMGSZ   = 640    # Safe for T4 15.6GB
BATCH   = 16
```

**Accessing your trained model:**
After training, go to the **Output** tab in your Kaggle notebook — `best.pt` will be there for download.

---

## 7. Training Parameters Explained

Here is every parameter we set and what it means in plain English:

| Parameter | Our Value | What it does |
|-----------|-----------|--------------|
| `epochs` | 150 | How many times the model sees the entire dataset |
| `imgsz` | 1280 / 640 | Image resolution during training. Bigger = sees more detail but uses more VRAM |
| `batch` | 32 / 16 / 8 | How many images are processed at once. Bigger = faster but uses more VRAM |
| `optimizer` | SGD | The algorithm that updates the model's weights. SGD works best for YOLO detection |
| `lr0` | 0.01 | Starting learning rate — how big each update step is |
| `lrf` | 0.01 | Final learning rate as a fraction of `lr0` |
| `momentum` | 0.937 | SGD momentum — helps the optimizer not get stuck |
| `weight_decay` | 0.0005 | Prevents overfitting by penalising large weights |
| `warmup_epochs` | 5 | For the first 5 epochs, LR starts very small and gradually increases — prevents unstable early training |
| `patience` | 20 | **Early stopping** — if mAP doesn't improve for 20 epochs, training stops automatically |
| `device` | 0 | Which GPU to use (0 = first GPU) |
| `conf` | 0.20 | Confidence threshold — boxes below 20% confidence are ignored |
| `iou` | 0.50 | IoU threshold for NMS — boxes overlapping more than 50% are merged |
| `workers` | 2-8 | CPU threads for loading images. More = faster data pipeline |
| `cache` | False | Whether to load all images into RAM. False saves RAM |
| `save_period` | 10 | Save a checkpoint every 10 epochs |

---

## 8. Optimization Techniques

These are the techniques we applied to get better accuracy than a default training run.

### Cosine Learning Rate (`cos_lr=True`)
**What it is:** Instead of dropping the learning rate in steps, it follows a smooth cosine curve — starting high, gradually decreasing, and ending near zero.

**Why it helps:** Smoother convergence. The model doesn't "shock" when the LR suddenly drops. Over 150 epochs this typically gives **+1-2% mAP** compared to step LR.

```
LR
│ ╲
│  ╲
│   ╲___
│       ╲___
│           ╲_____
└──────────────────── epochs
```

### AMP Mixed Precision (`amp=True`)
**What it is:** Uses 16-bit floating point (FP16) instead of 32-bit (FP32) where possible.

**Why it helps:** 
- Uses ~half the VRAM
- ~2× faster on GPUs with Tensor Cores (T4, A100, L40S)
- Minimal accuracy impact

**Important:** P100 does NOT have Tensor Cores — `amp=True` gives no speedup on P100.

### Close Mosaic (`close_mosaic=10`)
**What it is:** Mosaic augmentation combines 4 images into one during training. `close_mosaic=10` turns this OFF for the last 10 epochs.

**Why it helps:** The last few epochs the model fine-tunes on clean, realistic images instead of stitched-together mosaics. This usually gives **+0.5-1% mAP**.

### Label Smoothing (`label_smoothing=0.1`)
**What it is:** Instead of training with hard labels (0 or 1), we use soft labels (0.05 or 0.95).

**Why it helps:** Prevents the model from becoming overconfident. A model that is 95% sure is better calibrated than one that is 100% sure about everything. Reduces overfitting.

### Freeze Backbone (`freeze=10`)
**What it is:** For the first training epochs, the first 10 layers of the backbone (feature extractor) are frozen — their weights don't update.

**Why it helps:** The pretrained backbone already knows how to detect edges, shapes, and textures. Freezing it initially lets the head (detection layers) stabilise first, then everything fine-tunes together. Prevents early instability.

### Copy-Paste Augmentation (`copy_paste=0.3`)
**What it is:** Randomly cuts objects from one image and pastes them onto another.

**Why it helps for head detection:** Creates artificial crowded scenes with overlapping heads — exactly the hard case we want the model to handle well. Gives **+1-2% mAP** on dense crowd images.

### Test-Time Augmentation (`augment=True` during validation)
**What it is:** During inference, the image is flipped and scaled multiple times, predictions are made on each version, and the results are merged.

**Why it helps:** The model sees the image from multiple perspectives and votes on the best boxes. Typically gives **+1-2% mAP** at the cost of ~3× slower inference.

### Why `multi_scale=False` (disabled)
**What it is:** Randomly resizes training images by ±50% each batch.

**Why we disabled it:** At `imgsz=1280`, multi-scale would create images up to `1920px` randomly — causing sudden VRAM spikes that triggered Out of Memory errors even on 47GB VRAM. At lower resolutions (640px) it is safe to enable.

---

## 9. Augmentation Techniques

Augmentation means artificially creating variations of training images so the model learns to handle real-world conditions.

| Augmentation | Our Value | What it does |
|---|---|---|
| `hsv_h=0.015` | Tiny hue shift | Slightly changes the colour tone of the image |
| `hsv_s=0.7` | Saturation jitter | Makes colours more/less vivid |
| `hsv_v=0.4` | Brightness jitter | Makes images lighter or darker |
| `degrees=0.0` | **Disabled** | We don't rotate — heads in real images are always upright |
| `flipud=0.0` | **Disabled** | No vertical flip — upside-down heads don't occur in real life |
| `fliplr=0.5` | 50% chance | Randomly mirror left-right — realistic since heads face both directions |
| `mosaic=1.0` | Always on | Combines 4 training images into one — great for small object detection |
| `copy_paste=0.3` | 30% chance | Pastes heads from other images — creates artificial crowds |
| `scale=0.5` | ±50% zoom | Random zoom in/out — helps detect heads at different distances |
| `translate=0.1` | ±10% shift | Random position shift |

---

## 10. Exporting to TensorRT (.engine)

After training, you can export the model to TensorRT format for much faster inference on NVIDIA GPUs.

**What is TensorRT?**
TensorRT is NVIDIA's inference optimizer. It takes your trained model and compiles it specifically for your GPU, applying optimizations that make it run 3-5× faster than the original PyTorch model.

**Export command:**
```python
from ultralytics import YOLO

model = YOLO("best.pt")
model.export(
    format    = "engine",   # TensorRT
    imgsz     = 640,        # must match training imgsz
    half      = True,       # FP16 — 2x faster, same accuracy
    batch     = 1,          # batch size for inference
    device    = 0,
    workspace = 4,          # GB of GPU memory for optimization
    simplify  = True,       # simplify the graph first
)
# Saves best.engine in same folder as best.pt
```

**Precision options:**

| Precision | Speed | Accuracy | Use when |
|-----------|-------|----------|----------|
| FP32 | Baseline | Best | Debugging only |
| FP16 | ~2× faster | ≈ same | ✅ Recommended |
| INT8 | ~4× faster | Slight drop | Edge deployment |

**Important:** `.engine` files are **GPU-specific**. A file compiled on a T4 will NOT run on an L40S. Always re-export on the target deployment GPU.

**Run inference with .engine:**
```python
model = YOLO("best.engine")
results = model.predict("image.jpg", conf=0.2, iou=0.5)
```

---

## 11. Errors We Faced & How We Fixed Them

### Error 1 — CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Cause:** `multi_scale=True` randomly upscales images to `imgsz × 1.5`, causing sudden VRAM spikes.

**Fix:**
```python
multi_scale = False   # ← disable this
```

---

### Error 2 — mega.py broken on Python 3.12
```
AttributeError: module 'asyncio' has no attribute 'coroutine'
```
**Cause:** `mega.py` uses `@asyncio.coroutine` which was removed in Python 3.11+.

**Fix:** Use our custom `mega_download()` function with `pycryptodome` instead of `mega.py`.

---

### Error 3 — mega.py KeyError on new URL format
```
KeyError: 'Url key missing'
```
**Cause:** `mega.py` does not support the newer `/file/ID#KEY` MEGA URL format.

**Fix:** Same as above — use our custom downloader.

---

### Error 4 — Downloaded file not readable / corrupted
**Cause:** MEGA encrypts all files. Downloading the raw bytes without decryption gives a corrupted file.

**Fix:** Our `mega_download()` function decrypts the file on-the-fly using AES-CTR mode with the key extracted from the URL hash fragment.

---

### Error 5 — P100 not compatible with PyTorch
```
UserWarning: Tesla P100-PCIE-16GB with CUDA capability sm_60 is not
compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_70 sm_75 sm_80...
```
**Cause:** P100 GPU is too old (2016). PyTorch 2.x dropped support for anything below sm_70.

**Fix:** In Kaggle, go to Settings → Accelerator → Switch to **GPU T4 x2**.

There is no code fix for this — you must use a different GPU.

---

### Error 6 — PyTorch version not found on Kaggle
```
ERROR: Could not find a version that satisfies the requirement torch==2.0.1
```
**Cause:** Kaggle's pip index only has `torch>=2.2.0+cu118`, and none of these support P100 (sm_60).

**Fix:** Switch GPU to T4 — no PyTorch downgrade needed.

---

### Error 7 — `apt-get` permission denied on Lightning AI
```
E: Could not open lock file /var/lib/dpkg/lock-frontend (Permission denied)
```
**Cause:** Lightning AI Studio runs as a non-root user.

**Fix:** Use `sudo apt-get` or use `conda install` instead. For megatools specifically, use our Python-based `mega_download()` function instead.

---

## 12. Results & Metrics

### What is mAP?
**mAP (mean Average Precision)** is the main metric for object detection.

- **mAP@50** — measures accuracy when a detection is considered correct if it overlaps the ground truth by at least 50%
- **mAP@50-95** — averages mAP across IoU thresholds from 50% to 95% — a stricter, more comprehensive metric

Higher is better. A mAP@50 of 0.85 means the model correctly detects 85% of heads.

### Typical results by platform

| Platform | GPU | imgsz | Batch | mAP@50 (est.) | Training time |
|---|---|---|---|---|---|
| Colab Free | T4 | 640 | 16 | ~0.78-0.82 | ~4-6 hrs |
| Colab Pro | A100 | 1280 | 32 | ~0.83-0.87 | ~3-4 hrs |
| Lightning AI | L40S | 1280 | 32 | ~0.84-0.88 | ~3-5 hrs |
| Kaggle | T4 | 640 | 16 | ~0.78-0.82 | ~4-6 hrs |

Higher `imgsz` generally gives better mAP for small objects like heads, because the model can see more detail.

---

## 13. Glossary

| Term | Meaning |
|------|---------|
| **YOLO** | You Only Look Once — a real-time object detection algorithm |
| **mAP** | Mean Average Precision — main accuracy metric for object detection |
| **VRAM** | Video RAM — memory on the GPU. More VRAM = bigger batch + image size |
| **Epoch** | One full pass through the entire training dataset |
| **Batch size** | Number of images processed simultaneously before updating weights |
| **Learning rate** | How big each weight update step is. Too high = unstable, too low = slow |
| **AMP** | Automatic Mixed Precision — uses FP16 where safe to speed up training |
| **TensorRT** | NVIDIA's inference optimizer — makes models 3-5× faster on NVIDIA GPUs |
| **FP16** | 16-bit floating point — half the memory of FP32, nearly same accuracy |
| **IoU** | Intersection over Union — measures overlap between predicted and actual box |
| **NMS** | Non-Maximum Suppression — removes duplicate detection boxes |
| **WBF** | Weighted Boxes Fusion — better alternative to NMS for dense objects |
| **TTA** | Test-Time Augmentation — run inference on flipped/scaled versions and merge |
| **Backbone** | The feature extraction part of the neural network |
| **Mosaic** | Augmentation that combines 4 images into one training sample |
| **Overfitting** | When a model memorises training data but fails on new images |
| **Early stopping** | Automatically stops training when accuracy stops improving |
| **sm_60 / sm_70** | CUDA capability levels — higher = newer GPU with more features |
| **Transfer learning** | Starting from a pretrained model instead of random weights |

---

## Quick Start

```bash
# 1. Clone this repo
git clone https://github.com/yourusername/yolov11m-head-detection
cd yolov11m-head-detection

# 2. Install dependencies
pip install ultralytics ensemble-boxes pycryptodome -q

# 3. Choose your platform script and run:
#    Google Colab   → train_colab.py
#    Lightning AI   → train_lightning.py
#    Kaggle         → train_kaggle.py  (includes dataset download)
```

---

## License

MIT License — free to use, modify, and distribute.

---

*Built with patience, lots of OOM errors, and three different cloud platforms.*
