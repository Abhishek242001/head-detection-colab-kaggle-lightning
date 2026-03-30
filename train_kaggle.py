# ═════════════════════════════════════════════════════════════════════════════
#  YOLOv11m — Head Detection  |  Kaggle Edition
#  Target class : person (head)
#  Environment  : Kaggle Notebook
#
#  System tested on:
#    GPU    : Tesla T4  (recommended — use T4 x2 in Kaggle settings)
#    VRAM   : 15.6 GB
#    Python : 3.12
#    CUDA   : 12.8
#    PyTorch: 2.10.0+cu128
#
#  ⚠️  GPU WARNING:
#    P100 is NOT compatible with PyTorch 2.x (sm_60 too old).
#    In Kaggle → Settings → Accelerator → select "GPU T4 x2"
#
#  Run order:
#    Cell 1 : !pip install ultralytics ensemble-boxes pycryptodome -q
#    Cell 2 : Run mega_download section below (downloads dataset)
#    Cell 3 : Run training section below
# ═════════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════
#  PART A — Download & decrypt dataset from MEGA
# ════════════════════════════════════════════════════════════
import re, struct, binascii, json, os, requests, zipfile, tarfile, shutil
from Crypto.Cipher import AES

MEGA_URL    = "https://mega.nz/file/VVpRTZhR#1KgcatlbE51aLc6K9Gjj2ToCKj0pD1YVqdQ47QSRzYw"
DEST_DIR    = "/kaggle/working/"
EXTRACT_DIR = "/kaggle/working/Dataset/"

def mega_download(url, dest=DEST_DIR):
    match = re.search(r'file/([^#]+)#(.+)', url)
    if not match:
        raise ValueError("Invalid MEGA URL")
    file_id, file_key = match.group(1), match.group(2)

    def b64d(s):
        s = s.replace('-', '+').replace('_', '/')
        s += '=' * (-len(s) % 4)
        return binascii.a2b_base64(s)

    def a32_to_bytes(a): return struct.pack('>%dI' % len(a), *a)
    def bytes_to_a32(b):
        count = len(b) // 4
        return struct.unpack('>%dI' % count, b[:count*4])

    def decrypt_attr(attr_bytes, key):
        cipher = AES.new(a32_to_bytes(key[:4]), AES.MODE_CBC, iv=b'\x00'*16)
        dec = cipher.decrypt(attr_bytes)
        try:    return json.loads(dec[4:].split(b'\x00')[0])
        except: return {}

    key_raw = bytes_to_a32(b64d(file_key))
    k  = [key_raw[0]^key_raw[4], key_raw[1]^key_raw[5],
          key_raw[2]^key_raw[6], key_raw[3]^key_raw[7]]
    iv = key_raw[4:6] + (0, 0)

    resp   = requests.post("https://g.api.mega.co.nz/cs", params={"id":1},
                           json=[{"a":"g","g":1,"p":file_id}], timeout=30)
    data   = resp.json()[0]
    dl_url, size = data['g'], data['s']
    attr   = decrypt_attr(b64d(data['at']), k)
    fname  = attr.get('n', file_id)
    outpath = os.path.join(dest, fname)

    if os.path.exists(outpath) and os.path.getsize(outpath) == size:
        print(f"⏭️  Already exists: {fname}")
        return outpath

    print(f"Downloading : {fname}  ({size/1e6:.0f} MB)")
    cipher = AES.new(a32_to_bytes(k), AES.MODE_CTR, initial_value=a32_to_bytes(iv), nonce=b'')
    downloaded = 0
    with requests.get(dl_url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(outpath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                f.write(cipher.decrypt(chunk))
                downloaded += len(chunk)
                print(f"\r  {downloaded/1e6:.0f}/{size/1e6:.0f} MB ({downloaded/size*100:.1f}%)", end="")
    print(f"\n✅ Saved: {outpath}")
    return outpath

def extract_dataset(filepath, extract_to=EXTRACT_DIR):
    os.makedirs(extract_to, exist_ok=True)
    if filepath.endswith(".zip"):
        with zipfile.ZipFile(filepath, 'r') as z:
            z.extractall(extract_to)
    elif filepath.endswith((".tar.gz", ".tgz")):
        with tarfile.open(filepath, "r:gz") as t:
            t.extractall(extract_to)
    print(f"✅ Extracted to: {extract_to}")

def update_yaml(yaml_path, train_path, val_path):
    import yaml
    with open(yaml_path, "w") as f:
        yaml.dump({"train": train_path, "val": val_path,
                   "nc": 1, "names": ["person"]}, f, default_flow_style=False)
    print(f"✅ YAML updated: {yaml_path}")

# Run download + extract + yaml update
saved = mega_download(MEGA_URL)
extract_dataset(saved)

YAML_PATH  = "/kaggle/working/Dataset/Dataset/dataset_config.yaml"
TRAIN_PATH = "/kaggle/working/Dataset/Dataset/images/train"
VAL_PATH   = "/kaggle/working/Dataset/Dataset/images/val"
update_yaml(YAML_PATH, TRAIN_PATH, VAL_PATH)


# ════════════════════════════════════════════════════════════
#  PART B — Train
# ════════════════════════════════════════════════════════════
import torch, gc
from ultralytics import YOLO
from pathlib import Path

print("="*60)
print("  YOLOv11m Head Detection — Kaggle")
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

MODEL_WEIGHTS = "yolo11m.pt"
PROJECT_NAME  = "/kaggle/working/head_detection"
RUN_NAME      = "yolo11m_kaggle"
EPOCHS        = 150
OPTIMIZER     = "SGD"
LR0, LRF      = 0.01, 0.01
MOMENTUM      = 0.937
WEIGHT_DECAY  = 0.0005
WARMUP_EPOCHS = 5.0
PATIENCE      = 20
DEVICE        = 0
CONF_THRESH   = 0.20
IOU_THRESH    = 0.50

# Auto batch + imgsz by VRAM
vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
if vram >= 40:   IMGSZ, BATCH = 1280, 32
elif vram >= 20: IMGSZ, BATCH = 1280, 16
elif vram >= 14: IMGSZ, BATCH = 640,  16   # T4
else:            IMGSZ, BATCH = 640,  8

print(f"\n  Auto config → imgsz={IMGSZ}  batch={BATCH}\n")

model   = YOLO(MODEL_WEIGHTS)
results = model.train(
    data=YAML_PATH, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
    optimizer=OPTIMIZER, lr0=LR0, lrf=LRF, momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY, warmup_epochs=WARMUP_EPOCHS,
    patience=PATIENCE, device=DEVICE, project=PROJECT_NAME, name=RUN_NAME,
    cos_lr=True, amp=True, multi_scale=False, close_mosaic=10,
    label_smoothing=0.1, freeze=10,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1,
    scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5,
    mosaic=1.0, mixup=0.0, copy_paste=0.3,
    save=True, save_period=10, plots=True, verbose=True,
    workers=2, cache=False,
)

best_weights = Path(PROJECT_NAME) / RUN_NAME / "weights" / "best.pt"
model_best   = YOLO(str(best_weights))

val_std = model_best.val(data=YAML_PATH, imgsz=IMGSZ, batch=BATCH,
                          conf=CONF_THRESH, iou=IOU_THRESH, device=DEVICE, plots=True)
val_tta = model_best.val(data=YAML_PATH, imgsz=IMGSZ, batch=BATCH,
                          conf=CONF_THRESH, iou=IOU_THRESH, device=DEVICE, augment=True)

# Copy best.pt to output tab
shutil.copy(str(best_weights), "/kaggle/working/best.pt")

print(f"\n{'='*60}")
print(f"  Training complete!")
print(f"  mAP@50  (std) : {val_std.box.map50:.4f}")
print(f"  mAP@50  (TTA) : {val_tta.box.map50:.4f}")
print(f"  mAP@50-95     : {val_tta.box.map:.4f}")
print(f"  Best weights  : /kaggle/working/best.pt")
print(f"{'='*60}")

# ── Export TensorRT .engine ───────────────────────────────────────────────────
# model_best.export(format="engine", imgsz=IMGSZ, half=True, batch=1,
#                   device=DEVICE, workspace=4, simplify=True)
