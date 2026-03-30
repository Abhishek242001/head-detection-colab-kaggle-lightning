# ═════════════════════════════════════════════════════════════════════════════
#  YOLOv11m — Head Detection  |  Google Colab Edition
#  Target class : person (head)
#  Environment  : Google Colab (Free T4 / Colab Pro A100)
#
#  System tested on:
#    GPU    : Tesla T4 (Free) / A100 (Colab Pro)
#    VRAM   : 15 GB (T4) / 40 GB (A100)
#    Python : 3.10
#    CUDA   : 11.8
#
#  ⚠️  IMPORTANT — Run cells in order:
#    Cell 1 : Install dependencies
#    Cell 2 : Mount Google Drive
#    Cell 3 : This training script
# ═════════════════════════════════════════════════════════════════════════════

# ── CELL 1 : Install ─────────────────────────────────────────────────────────
# !pip install ultralytics ensemble-boxes -q

# ── CELL 2 : Mount Google Drive ──────────────────────────────────────────────
# from google.colab import drive
# drive.mount('/content/drive')

# ── CELL 3 : Training ────────────────────────────────────────────────────────
import torch, gc
from ultralytics import YOLO
from pathlib import Path

# ── Sanity checks ─────────────────────────────────────────────────────────────
print("="*60)
print("  YOLOv11m Head Detection — Google Colab")
print("="*60)
print(f"PyTorch  : {torch.__version__}")
print(f"CUDA     : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU      : {torch.cuda.get_device_name(0)}")
    print(f"VRAM     : {vram:.1f} GB")
print("="*60)

# Clear VRAM before starting
gc.collect()
torch.cuda.empty_cache()

# ── Config ────────────────────────────────────────────────────────────────────
# ⚠️  Update this path to where your dataset is in Google Drive
DATASET_YAML  = "/content/drive/MyDrive/Dataset/dataset_config.yaml"
MODEL_WEIGHTS = "yolo11m.pt"           # auto-downloaded on first run
PROJECT_NAME  = "/content/drive/MyDrive/head_detection"   # save to Drive so it persists
RUN_NAME      = "yolo11m_colab"

EPOCHS        = 150
OPTIMIZER     = "SGD"
LR0           = 0.01
LRF           = 0.01
MOMENTUM      = 0.937
WEIGHT_DECAY  = 0.0005
WARMUP_EPOCHS = 5.0
PATIENCE      = 20                     # early stop if no improvement for 20 epochs
DEVICE        = 0
CONF_THRESH   = 0.20
IOU_THRESH    = 0.50

# Auto-select batch + imgsz based on VRAM
# Colab free = T4 (15GB), Colab Pro = A100 (40GB)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
if vram >= 40:
    IMGSZ, BATCH = 1280, 32            # A100 Colab Pro
elif vram >= 14:
    IMGSZ, BATCH = 640,  16            # T4 Free Colab
else:
    IMGSZ, BATCH = 640,  8             # fallback

print(f"\n  Auto config → imgsz={IMGSZ}  batch={BATCH}\n")

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
    cos_lr           = True,           # smooth cosine LR decay
    amp              = True,           # mixed precision (FP16) — 2x faster on T4/A100
    multi_scale      = False,          # disabled — causes VRAM spikes
    close_mosaic     = 10,             # disable mosaic for last 10 epochs
    label_smoothing  = 0.1,            # reduces overconfidence
    freeze           = 10,             # freeze backbone first 10 layers

    # Augmentations (tuned for head detection)
    hsv_h            = 0.015,
    hsv_s            = 0.7,
    hsv_v            = 0.4,
    degrees          = 0.0,            # heads are upright — no rotation
    translate        = 0.1,
    scale            = 0.5,
    shear            = 0.0,
    perspective      = 0.0,
    flipud           = 0.0,            # no vertical flip for heads
    fliplr           = 0.5,
    mosaic           = 1.0,
    mixup            = 0.0,
    copy_paste       = 0.3,            # good for crowded scenes

    save             = True,
    save_period      = 10,
    plots            = True,
    verbose          = True,
    workers          = 2,              # Colab has limited CPU
    cache            = False,          # Colab RAM is limited
)

# ── Validate ──────────────────────────────────────────────────────────────────
best_weights = Path(PROJECT_NAME) / RUN_NAME / "weights" / "best.pt"
model_best   = YOLO(str(best_weights))

val_std = model_best.val(data=DATASET_YAML, imgsz=IMGSZ, batch=BATCH,
                          conf=CONF_THRESH, iou=IOU_THRESH, device=DEVICE, plots=True)
val_tta = model_best.val(data=DATASET_YAML, imgsz=IMGSZ, batch=BATCH,
                          conf=CONF_THRESH, iou=IOU_THRESH, device=DEVICE, augment=True)

print(f"\n{'='*60}")
print(f"  Training complete!")
print(f"  mAP@50  (std) : {val_std.box.map50:.4f}")
print(f"  mAP@50  (TTA) : {val_tta.box.map50:.4f}")
print(f"  mAP@50-95     : {val_tta.box.map:.4f}")
print(f"  Best weights  : {best_weights}")
print(f"{'='*60}")

# ── Export TensorRT engine (optional) ────────────────────────────────────────
# model_best.export(format="engine", imgsz=IMGSZ, half=True, device=DEVICE)
