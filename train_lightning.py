# ═════════════════════════════════════════════════════════════════════════════
#  YOLOv11m — Head Detection  |  Lightning AI Studio Edition
#  Target class : person (head)
#  Environment  : Lightning AI Studio
#
#  System tested on:
#    GPU    : NVIDIA L40S
#    VRAM   : 47.7 GB
#    Python : 3.12
#    CUDA   : 12.8
#    PyTorch: 2.8.0+cu128
#
#  Run order:
#    Step 1 : pip install ultralytics ensemble-boxes pycryptodome -q
#    Step 2 : Run mega_download.py  (downloads + decrypts dataset from MEGA)
#    Step 3 : Run this file
# ═════════════════════════════════════════════════════════════════════════════

import torch, gc
from ultralytics import YOLO
from pathlib import Path

# ── Sanity checks ─────────────────────────────────────────────────────────────
print("="*60)
print("  YOLOv11m Head Detection — Lightning AI Studio")
print("="*60)
print(f"PyTorch  : {torch.__version__}")
print(f"CUDA     : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU      : {torch.cuda.get_device_name(0)}")
    print(f"VRAM     : {vram:.1f} GB")
print("="*60)

gc.collect()
torch.cuda.empty_cache()

# ── Config ────────────────────────────────────────────────────────────────────
# Dataset downloaded via mega_download.py → extracted to /home/zeus/Dataset/
DATASET_YAML  = "/home/zeus/Dataset/Dataset/dataset_config.yaml"
MODEL_WEIGHTS = "yolo11m.pt"
PROJECT_NAME  = "head_detection"
RUN_NAME      = "yolo11m_lightning"

EPOCHS        = 150
IMGSZ         = 1280               # L40S has 47GB — can handle 1280
BATCH         = 32                 # safe for L40S at 1280px
OPTIMIZER     = "SGD"
LR0           = 0.01
LRF           = 0.01
MOMENTUM      = 0.937
WEIGHT_DECAY  = 0.0005
WARMUP_EPOCHS = 5.0
PATIENCE      = 20
DEVICE        = 0
CONF_THRESH   = 0.20
IOU_THRESH    = 0.50

print(f"\n  Config → imgsz={IMGSZ}  batch={BATCH}  GPU=L40S\n")

# ── Load model ────────────────────────────────────────────────────────────────
model = YOLO(MODEL_WEIGHTS)

# ── Train ─────────────────────────────────────────────────────────────────────
results = model.train(
    data             = DATASET_YAML,
    epochs           = EPOCHS,
    imgsz            = IMGSZ,
    batch            = BATCH,
    optimizer        = OPTIMIZER,
    lr0              = LR0,
    lrf              = LRF,
    momentum         = MOMENTUM,
    weight_decay     = WEIGHT_DECAY,
    warmup_epochs    = WARMUP_EPOCHS,
    patience         = PATIENCE,
    device           = DEVICE,
    project          = PROJECT_NAME,
    name             = RUN_NAME,

    # Optimizations
    cos_lr           = True,
    amp              = True,
    multi_scale      = False,          # ⚠️ disabled — spikes VRAM even on 48GB at 1280px
    close_mosaic     = 10,
    label_smoothing  = 0.1,
    freeze           = 10,

    # Augmentations
    hsv_h            = 0.015,
    hsv_s            = 0.7,
    hsv_v            = 0.4,
    degrees          = 0.0,
    translate        = 0.1,
    scale            = 0.5,
    shear            = 0.0,
    perspective      = 0.0,
    flipud           = 0.0,
    fliplr           = 0.5,
    mosaic           = 1.0,
    mixup            = 0.0,
    copy_paste       = 0.3,

    save             = True,
    save_period      = 10,
    plots            = True,
    verbose          = True,
    workers          = 8,              # L40S has plenty of CPU cores
    cache            = False,
)

# ── Validate ──────────────────────────────────────────────────────────────────
best_weights = Path(PROJECT_NAME) / RUN_NAME / "weights" / "best.pt"
model_best   = YOLO(str(best_weights))

val_std = model_best.val(data=DATASET_YAML, imgsz=IMGSZ, batch=BATCH,
                          conf=CONF_THRESH, iou=IOU_THRESH, device=DEVICE, plots=True)
val_tta = model_best.val(data=DATASET_YAML, imgsz=IMGSZ, batch=BATCH,
                          conf=CONF_THRESH, iou=IOU_THRESH, device=DEVICE, augment=True)

# Checkpoint ensemble
last_weights = Path(PROJECT_NAME) / RUN_NAME / "weights" / "last.pt"
if last_weights.exists():
    ensemble = YOLO([str(best_weights), str(last_weights)])
    val_ens  = ensemble.val(data=DATASET_YAML, imgsz=IMGSZ, batch=BATCH,
                             conf=CONF_THRESH, iou=IOU_THRESH, device=DEVICE)
    print(f"  mAP@50 (ensemble) : {val_ens.box.map50:.4f}")

print(f"\n{'='*60}")
print(f"  Training complete!")
print(f"  mAP@50  (std) : {val_std.box.map50:.4f}")
print(f"  mAP@50  (TTA) : {val_tta.box.map50:.4f}")
print(f"  mAP@50-95     : {val_tta.box.map:.4f}")
print(f"  Best weights  : {best_weights}")
print(f"{'='*60}")

# ── Export TensorRT .engine ───────────────────────────────────────────────────
# Uncomment to export after training:
# model_best.export(
#     format    = "engine",
#     imgsz     = IMGSZ,
#     half      = True,      # FP16
#     batch     = 1,
#     device    = DEVICE,
#     workspace = 8,         # GB — L40S can handle 8GB workspace
#     simplify  = True,
# )
# print("✅ TensorRT engine exported")
