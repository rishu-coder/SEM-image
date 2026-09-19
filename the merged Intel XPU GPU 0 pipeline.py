r"""
Merged Intel-XPU hybrid SEM pipeline.

This is the unified, runnable PyTorch version of:
1. MobileNetV2 hybrid classification-regression pipeline, and
2. Combined Intel-XPU CNN/ViT benchmark.

It preserves the MobileNetV2 scientific formulation and publication outputs,
while adding reproducible multi-model benchmarking. It targets Intel XPU GPU 0
explicitly (torch.device("xpu:0")) and fails clearly instead of silently using CPU.

Default models are selected for the user's Intel Core Ultra 7 / Intel Arc system:
- MobileNetV2       lightweight primary CNN
- EfficientNet-B0   efficient comparison CNN
- ResNet-50         conventional residual CNN
- ViT-B/16          transformer comparison (optional and slower)

Every architecture uses exactly the same:
- train/validation/test CSV rows
- preprocessing and conservative SEM augmentation
- six-level stress classification head
- continuous standardised log10(stress) regression head
- stress-to-property monotonic interpolation
- loss weights, optimisation stages and evaluation metrics
- 600 dpi PNG and vector PDF publication figures

Default CSV:
C:\Users\rt4\Documents\ML\CNN\code\data\csv_manifests\sem_dataset_all.csv

Development run (CNN only at 160 px):
python combined_xpu_cnn_vit_sem_benchmark_v3.py --models mobilenet_v2 efficientnet_b0 --image-size 160 --batch-size 16 --head-epochs 5 --fine-tune-epochs 10

Final benchmark (ViT automatically enforces 224 px and batch <= 4):
python combined_xpu_cnn_vit_sem_benchmark_v3.py --models mobilenet_v2 efficientnet_b0 resnet50 vit_b_16 --image-size 224 --batch-size 8 --head-epochs 10 --fine-tune-epochs 25

Recommended Intel Arc 140T GPU 0 smoke test:
python merged_xpu0_sem_hybrid_benchmark.py --xpu-index 0 --models mobilenet_v2 --image-size 160 --batch-size 8 --head-epochs 1 --fine-tune-epochs 1 --num-workers 0

Recommended full run on GPU 0:
python merged_xpu0_sem_hybrid_benchmark.py --xpu-index 0 --models mobilenet_v2 efficientnet_b0 resnet50 vit_b_16 --image-size 224 --batch-size 4 --head-epochs 10 --fine-tune-epochs 25 --num-workers 0 --amp-dtype float16
"""

from __future__ import annotations

import argparse
import io
import contextlib
import copy
import json
import math
import os
import random
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageFile, UnidentifiedImageError
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

SEED = 42
ImageFile.LOAD_TRUNCATED_IMAGES = True
STRESS_LEVELS = np.array([100, 200, 400, 1000, 2000, 6000], dtype=float)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PROPERTY_TABLE = pd.DataFrame({
    "stress_kpa": STRESS_LEVELS,
    "void_ratio": [0.90, 0.80, 0.70, 0.58, 0.49, 0.38],
    "permeability_m_s": [3e-10, 1.5e-10, 8e-11, 3e-11, 1.5e-11, 5e-12],
    "cv_m2_s": [1e-7, 5e-8, 3e-8, 1.5e-8, 8e-9, 3e-9],
    "mv_m2_kN": [3e-4, 2.5e-4, 2e-4, 1.5e-4, 1e-4, 6e-5],
})
PROPERTY_NAMES = ["void_ratio", "permeability_m_s", "cv_m2_s", "mv_m2_kN"]
PROPERTY_LABELS = ["Void ratio, e", "Permeability, k (m/s)",
                   "Coefficient of consolidation, cᵥ (m²/s)",
                   "Coefficient of volume compressibility, mᵥ (m²/kN)"]
LOG_PROPERTY_INDICES = {1, 2, 3}

MODEL_REGISTRY = {
    "mobilenet_v2": {"family": "CNN", "label": "MobileNetV2"},
    "efficientnet_b0": {"family": "CNN", "label": "EfficientNet-B0"},
    "resnet50": {"family": "CNN", "label": "ResNet-50"},
    "densenet121": {"family": "CNN", "label": "DenseNet-121"},
    "convnext_tiny": {"family": "Modern CNN", "label": "ConvNeXt-Tiny"},
    "vit_b_16": {"family": "Vision Transformer", "label": "ViT-B/16"},
}


@dataclass
class Config:
    csv_path: str
    output_dir: str
    model_keys: Tuple[str, ...]
    image_size: int = 160
    batch_size: int = 32
    head_epochs: int = 5
    fine_tune_epochs: int = 10
    head_lr: float = 1e-3
    backbone_lr: float = 1e-5
    fine_tune_head_lr: float = 1e-4
    weight_decay: float = 1e-4
    dropout: float = 0.30
    classification_weight: float = 1.0
    regression_weight: float = 1.0
    patience: int = 5
    warmup_epochs: int = 2
    unfreeze_blocks: int = 1
    num_workers: int = 2
    prefetch_factor: int = 2
    amp_dtype: str = "float16"
    use_compile: bool = False
    use_cache: bool = True
    balance_training: bool = True
    pretrained: bool = True
    seed: int = SEED
    xpu_index: int = 0


def seed_all(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if hasattr(torch, "xpu"):
        try: torch.xpu.manual_seed_all(seed)
        except Exception: pass
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    try: torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception: pass


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed); np.random.seed(worker_seed)


def select_device(xpu_index=0):
    """Select Intel XPU explicitly and verify that the requested adapter works."""
    if not hasattr(torch, "xpu"):
        raise RuntimeError(
            "This PyTorch build has no Intel XPU support. Install an XPU-enabled "
            "PyTorch build, then run this verification command: "
            "python -c 'import torch; print(torch.xpu.is_available())'"
        )
    if not torch.xpu.is_available():
        raise RuntimeError(
            "Intel XPU is not available to PyTorch. Check the Intel graphics driver "
            "and your XPU-enabled PyTorch installation."
        )
    count = torch.xpu.device_count()
    if xpu_index < 0 or xpu_index >= count:
        raise RuntimeError(f"Requested xpu:{xpu_index}, but PyTorch reports {count} XPU device(s).")
    device = torch.device(f"xpu:{xpu_index}")
    torch.xpu.set_device(device)
    try:
        name = torch.xpu.get_device_name(xpu_index)
    except Exception:
        name = "Intel XPU"
    # Small allocation catches unusable/runtime-mismatched drivers before training.
    probe = torch.zeros(1, device=device)
    del probe
    torch.xpu.synchronize(device)
    print(f"Device: {device} ({name}); XPU count={count}")
    return device


def plot_style():
    sns.set_theme(style="ticks", context="paper")
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.labelsize": 11, "axes.titlesize": 11, "legend.fontsize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.linewidth": 1,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600})


def save_fig(fig, stem):
    fig.savefig(Path(stem).with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(Path(stem).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def validate_csv(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    if "original_split" in df and "split" not in df:
        df = df.rename(columns={"original_split": "split"})
    required = ["image_path", "split", "stress_kpa"]
    missing = [c for c in required if c not in df]
    if missing: raise ValueError(f"Missing CSV columns: {missing}")
    df["split"] = df.split.astype(str).str.strip().str.lower().replace({"validation": "val"})
    df["stress_kpa"] = pd.to_numeric(df.stress_kpa, errors="coerce")
    if df.stress_kpa.isna().any(): raise ValueError("Invalid stress_kpa values.")
    invalid = sorted(set(df.stress_kpa.astype(int)) - set(STRESS_LEVELS.astype(int)))
    if invalid: raise ValueError(f"Unsupported stress levels: {invalid}")

    valid, rejected = [], []
    for index, row in df.iterrows():
        path = Path(str(row.image_path)); reason = None; fmt = None; width = height = None
        if not path.is_file(): reason = "file_not_found"
        else:
            try:
                raw = path.read_bytes()
                if not raw:
                    raise OSError("empty image file")
                with Image.open(io.BytesIO(raw)) as image:
                    fmt = image.format
                    image.load()
                    rgb = image.convert("RGB")
                    width, height = rgb.size
                    # Force complete pixel access now, before DataLoader workers start.
                    rgb.tobytes()
                    if width < 2 or height < 2:
                        reason = "invalid_dimensions"
            except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
                reason = f"unreadable: {type(exc).__name__}: {exc}"
        if reason is None: valid.append(index)
        else: rejected.append({"row_index": index, "image_path": str(path), "split": row.split,
                               "stress_kpa": row.stress_kpa, "reason": reason,
                               "width": width, "height": height, "format": fmt})
    pd.DataFrame(rejected, columns=["row_index", "image_path", "split", "stress_kpa",
        "reason", "width", "height", "format"]).to_csv(Path(output_dir)/"rejected_images.csv", index=False)
    if rejected: print(f"Excluded {len(rejected)} invalid images. See rejected_images.csv")
    df = df.loc[valid].copy().reset_index(drop=True)
    if df.empty: raise ValueError("No readable images remain.")
    df = df.drop(columns=[c for c in PROPERTY_NAMES if c in df], errors="ignore")
    df = df.merge(PROPERTY_TABLE, on="stress_kpa", validate="many_to_one")
    df["log10_stress"] = np.log10(df.stress_kpa)
    return df


def split_df(df):
    train, val, test = (df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    if min(len(train), len(val), len(test)) == 0: raise ValueError("train, val and test must be non-empty.")
    unseen = (set(val.stress_kpa.astype(int)) | set(test.stress_kpa.astype(int))) - set(train.stress_kpa.astype(int))
    if unseen: raise ValueError(f"Validation/test levels absent from training: {sorted(unseen)}")
    for name, frame in [("train", train), ("val", val), ("test", test)]:
        print(f"{name:5s}: {len(frame):5d} images; levels={sorted(frame.stress_kpa.astype(int).unique())}")
    return train, val, test


def balance_df(df, seed):
    n = int(df.groupby("stress_kpa").size().max())
    pieces = [g.sample(n, replace=len(g)<n, random_state=seed) for _, g in df.groupby("stress_kpa")]
    return pd.concat(pieces).sample(frac=1, random_state=seed).reset_index(drop=True)


def dataset_summary(frames, output_dir):
    combined = pd.concat(frames)
    summary = combined.groupby(["split", "stress_kpa"]).size().reset_index(name="number_of_images")
    summary.to_csv(Path(output_dir)/"dataset_summary.csv", index=False)
    pivot = summary.pivot(index="stress_kpa", columns="split", values="number_of_images").fillna(0)
    fig, ax = plt.subplots(figsize=(6.6,3.8)); pivot.plot(kind="bar", ax=ax, color=sns.color_palette("colorblind",3))
    ax.set(xlabel="Effective stress (kPa)", ylabel="SEM images", title="Dataset composition")
    ax.legend(frameon=False); ax.tick_params(axis="x", rotation=0); ax.grid(False); sns.despine(ax=ax)
    fig.tight_layout(); save_fig(fig, Path(output_dir)/"figure_01_dataset_distribution")


def open_rgb_image(path, retries=3):
    """Read an image into memory, fully decode it, and return an independent RGB copy."""
    path = Path(path)
    last_error = None
    for attempt in range(retries):
        try:
            raw = path.read_bytes()
            if not raw:
                raise OSError("empty image file")
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                rgb = image.convert("RGB")
                return rgb.copy()
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(0.05 * (attempt + 1))
    raise RuntimeError(f"Unable to decode image after {retries} attempts: {path}; {type(last_error).__name__}: {last_error}")


class ResizePad:
    def __init__(self, size): self.size = int(size)
    def __call__(self, image):
        image = image.convert("RGB"); w,h = image.size; scale=min(self.size/w,self.size/h)
        image=image.resize((max(1,round(w*scale)),max(1,round(h*scale))),Image.Resampling.BILINEAR)
        canvas=Image.new("RGB",(self.size,self.size),(0,0,0))
        canvas.paste(image,((self.size-image.width)//2,(self.size-image.height)//2)); return canvas


def transforms_for(size):
    norm=transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)
    train=transforms.Compose([ResizePad(size), transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(), transforms.RandomRotation(5, interpolation=InterpolationMode.BILINEAR),
        transforms.RandomAffine(0,translate=(.02,.02),interpolation=InterpolationMode.BILINEAR),
        transforms.ColorJitter(contrast=.05),transforms.ToTensor(),norm])
    evaluation=transforms.Compose([ResizePad(size),transforms.ToTensor(),norm])
    return train,evaluation


def cache_name(path,size):
    import hashlib
    key=f"{Path(path).resolve()}|{Path(path).stat().st_mtime_ns}|{size}"
    return hashlib.sha1(key.encode()).hexdigest()+".pt"


class SEMDataset(Dataset):
    def __init__(self,df,encoder,scaler,size,training,cache_dir=None):
        self.df=df.reset_index(drop=True); self.encoder=encoder; self.scaler=scaler
        self.training=training; self.size=size; self.cache_dir=cache_dir
        self.train_t,self.eval_t=transforms_for(size)
        if cache_dir: Path(cache_dir).mkdir(parents=True,exist_ok=True)
    def __len__(self): return len(self.df)
    def _eval_image(self,path):
        cp=Path(self.cache_dir)/cache_name(path,self.size) if self.cache_dir else None
        if cp and cp.exists():
            try: return torch.load(cp,map_location="cpu",weights_only=True)
            except TypeError: return torch.load(cp,map_location="cpu")
            except Exception: cp.unlink(missing_ok=True)
        image = open_rgb_image(path)
        tensor = self.eval_t(image)
        if cp:
            tmp=cp.with_suffix(".tmp"); torch.save(tensor,tmp); tmp.replace(cp)
        return tensor
    def __getitem__(self,index):
        row=self.df.iloc[index]; path=str(row.image_path)
        if self.training:
            image = open_rgb_image(path)
            tensor = self.train_t(image)
        else:
            tensor = self._eval_image(path)
        class_id=int(self.encoder.transform([int(row.stress_kpa)])[0])
        # Manual one-feature scaling avoids repeated sklearn calls and feature-name warnings.
        scaled=(float(row.log10_stress)-float(self.scaler.mean_[0]))/float(self.scaler.scale_[0])
        return {"image":tensor,"class_id":torch.tensor(class_id),
                "log_stress":torch.tensor(scaled,dtype=torch.float32),"index":torch.tensor(index)}


def preflight_dataset(dataset, split_name, output_dir):
    """Decode every dataset item once in the main process for deterministic failures."""
    failures = []
    print(f"Preflight decoding {split_name}: {len(dataset)} images")
    for index in range(len(dataset)):
        try:
            item = dataset[index]
            if item["image"].shape != (3, dataset.size, dataset.size):
                raise ValueError(f"unexpected tensor shape {tuple(item['image'].shape)}")
        except Exception as exc:
            row = dataset.df.iloc[index]
            failures.append({"split": split_name, "dataset_index": index,
                             "image_path": row.image_path, "stress_kpa": row.stress_kpa,
                             "reason": f"{type(exc).__name__}: {exc}"})
    if failures:
        report = Path(output_dir) / "runtime_decode_failures.csv"
        pd.DataFrame(failures).to_csv(report, index=False)
        raise RuntimeError(f"{len(failures)} images failed runtime decoding. See {report}")


def make_loaders(frames,encoder,scaler,config,device):
    cache=Path(config.output_dir)/"resized_cache" if config.use_cache else None
    datasets=[SEMDataset(frames[0],encoder,scaler,config.image_size,True),
              SEMDataset(frames[1],encoder,scaler,config.image_size,False,cache/"val" if cache else None),
              SEMDataset(frames[2],encoder,scaler,config.image_size,False,cache/"test" if cache else None)]
    marker = Path(config.output_dir) / f"preflight_{config.image_size}px.ok"
    if not marker.exists():
        preflight_dataset(datasets[0], "train", config.output_dir)
        preflight_dataset(datasets[1], "val", config.output_dir)
        preflight_dataset(datasets[2], "test", config.output_dir)
        marker.write_text("ok", encoding="utf-8")
    common={"batch_size":config.batch_size,"num_workers":config.num_workers,
            "worker_init_fn":seed_worker,"persistent_workers":config.num_workers>0}
    if config.num_workers>0: common["prefetch_factor"]=config.prefetch_factor
    if device.type=="cuda": common["pin_memory"]=True
    gen=torch.Generator().manual_seed(config.seed)
    return datasets,[DataLoader(datasets[0],shuffle=True,generator=gen,**common),
                     DataLoader(datasets[1],shuffle=False,**common),DataLoader(datasets[2],shuffle=False,**common)]


def create_backbone(key,pretrained):
    try:
        if key=="mobilenet_v2":
            net=models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
            return net.features,net.last_channel,nn.AdaptiveAvgPool2d(1)
        if key=="efficientnet_b0":
            net=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            dim=net.classifier[-1].in_features; net.classifier=nn.Identity(); return net,dim,nn.Identity()
        if key=="resnet50":
            net=models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            dim=net.fc.in_features; net.fc=nn.Identity(); return net,dim,nn.Identity()
        if key=="densenet121":
            net=models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
            dim=net.classifier.in_features; net.classifier=nn.Identity(); return net,dim,nn.Identity()
        if key=="convnext_tiny":
            net=models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
            dim=net.classifier[-1].in_features; net.classifier[-1]=nn.Identity(); return net,dim,nn.Identity()
        if key=="vit_b_16":
            net=models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT if pretrained else None)
            dim=net.heads.head.in_features; net.heads=nn.Identity(); return net,dim,nn.Identity()
    except Exception as exc:
        if pretrained:
            warnings.warn(f"Pretrained {key} unavailable: {exc}. Retrying without weights.")
            return create_backbone(key,False)
        raise
    raise KeyError(key)


class HybridModel(nn.Module):
    def __init__(self,key,num_classes,dropout,pretrained):
        super().__init__(); self.key=key
        self.backbone,self.feature_dim,self.pool=create_backbone(key,pretrained)
        self.shared=nn.Sequential(nn.LayerNorm(self.feature_dim),nn.Linear(self.feature_dim,512),nn.GELU(),
            nn.Dropout(dropout),nn.Linear(512,256),nn.GELU(),nn.Dropout(dropout*.7))
        self.class_head=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Dropout(dropout*.5),nn.Linear(128,num_classes))
        self.reg_head=nn.Sequential(nn.Linear(256+num_classes,128),nn.GELU(),nn.Dropout(dropout*.5),nn.Linear(128,1))
    def forward(self,x):
        f=self.backbone(x); f=self.pool(f)
        if f.ndim>2: f=f.flatten(1)
        shared=self.shared(f); logits=self.class_head(shared); probs=torch.softmax(logits,1)
        stress=self.reg_head(torch.cat([shared,probs],1)).squeeze(1)
        return {"logits":logits,"log_stress":stress}


def set_stage(model,stage,blocks):
    for p in model.parameters(): p.requires_grad=False
    for module in (model.shared,model.class_head,model.reg_head):
        for p in module.parameters(): p.requires_grad=True
    if stage=="fine_tune":
        children=list(model.backbone.children())
        for child in children[-max(1,blocks):]:
            for p in child.parameters(): p.requires_grad=True
        for module in model.backbone.modules():
            if isinstance(module,nn.modules.batchnorm._BatchNorm):
                module.eval()
                for p in module.parameters(): p.requires_grad=False


def amp_context(config,device):
    if device.type not in {"xpu","cuda"} or config.amp_dtype=="float32": return contextlib.nullcontext()
    dtype=torch.float16 if config.amp_dtype=="float16" else torch.bfloat16
    return torch.autocast(device_type=device.type,dtype=dtype)


def scaler_for(config,device):
    enabled=device.type in {"xpu","cuda"} and config.amp_dtype=="float16"
    try: return torch.amp.GradScaler(device.type,enabled=enabled)
    except TypeError: return torch.amp.GradScaler(enabled=enabled)


def optimizer_for(model,config,stage):
    if stage=="head": return optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config.head_lr,weight_decay=config.weight_decay)
    back,heads=[],[]
    for name,p in model.named_parameters():
        if p.requires_grad: (back if name.startswith("backbone.") else heads).append(p)
    return optim.AdamW([{"params":back,"lr":config.backbone_lr},{"params":heads,"lr":config.fine_tune_head_lr}],weight_decay=config.weight_decay)


def scheduler_for(opt,epochs,warmup):
    def factor(epoch):
        if warmup and epoch<warmup: return (epoch+1)/warmup
        progress=(epoch-warmup)/max(1,epochs-warmup); return .5*(1+math.cos(math.pi*np.clip(progress,0,1)))
    return optim.lr_scheduler.LambdaLR(opt,factor)


def epoch_run(model,loader,config,device,opt=None,scaler=None):
    training=opt is not None; model.train(training)
    if training:
        for m in model.backbone.modules():
            if isinstance(m,nn.modules.batchnorm._BatchNorm): m.eval()
    sums={"loss":0.,"class_loss":0.,"reg_loss":0.,"correct":0,"n":0}
    for batch in loader:
        images=batch["image"].to(device,non_blocking=True); classes=batch["class_id"].to(device); stress=batch["log_stress"].to(device)
        if training: opt.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training),amp_context(config,device):
            out=model(images); cl=nn.functional.cross_entropy(out["logits"],classes)
            rl=nn.functional.huber_loss(out["log_stress"],stress,delta=1.0)
            loss=config.classification_weight*cl+config.regression_weight*rl
        if training:
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
            scaler.step(opt); scaler.update()
        n=len(images); sums["loss"]+=loss.item()*n; sums["class_loss"]+=cl.item()*n
        sums["reg_loss"]+=rl.item()*n; sums["correct"]+=(out["logits"].argmax(1)==classes).sum().item(); sums["n"]+=n
    n=max(1,sums["n"])
    return {"loss":sums["loss"]/n,"class_loss":sums["class_loss"]/n,
            "regression_loss":sums["reg_loss"]/n,"accuracy":sums["correct"]/n}


def train_stage(model,train_loader,val_loader,config,device,model_dir,stage,epochs):
    set_stage(model,stage,config.unfreeze_blocks); opt=optimizer_for(model,config,stage)
    scheduler=scheduler_for(opt,epochs,config.warmup_epochs); scaler=scaler_for(config,device)
    best=float("inf"); best_state=None; wait=0; history=[]
    for epoch in range(1,epochs+1):
        start=time.perf_counter(); tr=epoch_run(model,train_loader,config,device,opt,scaler); va=epoch_run(model,val_loader,config,device)
        scheduler.step(); seconds=time.perf_counter()-start
        history.append({"stage":stage,"epoch":epoch,"seconds":seconds,"learning_rate":max(g["lr"] for g in opt.param_groups),
                        **{f"train_{k}":v for k,v in tr.items()},**{f"val_{k}":v for k,v in va.items()}})
        print(f"{stage:9s} {epoch:03d}/{epochs}: train={tr['loss']:.4f}, acc={tr['accuracy']:.3f}; val={va['loss']:.4f}, acc={va['accuracy']:.3f}; {seconds:.1f}s")
        if va["loss"]<best-1e-7:
            best=va["loss"]; wait=0; best_state=copy.deepcopy(model.state_dict()); torch.save(best_state,model_dir/f"best_{stage}.pth")
        else:
            wait+=1
            if wait>=config.patience: print(f"Early stopping {stage}."); break
    if best_state is not None: model.load_state_dict(best_state)
    return history


def predict(model,loader,config,device):
    model.eval(); out={"true":[],"prob":[],"pred_stress":[],"index":[]}
    with torch.no_grad():
        for batch in loader:
            with amp_context(config,device): pred=model(batch["image"].to(device,non_blocking=True))
            out["true"].append(batch["class_id"].numpy()); out["prob"].append(torch.softmax(pred["logits"],1).float().cpu().numpy())
            out["pred_stress"].append(pred["log_stress"].float().cpu().numpy()); out["index"].append(batch["index"].numpy())
    return {k:np.concatenate(v) for k,v in out.items()}


def interpolate_properties(stress):
    stress=np.clip(np.asarray(stress,float),STRESS_LEVELS.min(),STRESS_LEVELS.max()); x=np.log10(STRESS_LEVELS); xp=np.log10(stress)
    result=np.zeros((len(stress),4)); result[:,0]=np.interp(xp,x,PROPERTY_TABLE.void_ratio)
    for i,col in enumerate(PROPERTY_NAMES[1:],1): result[:,i]=10**np.interp(xp,x,np.log10(PROPERTY_TABLE[col]))
    return result


def safe_r2(y,p): return np.nan if np.unique(y).size<2 else r2_score(y,p)


def metrics_table(true_stress,pred_stress,true_props,pred_props,label):
    rows=[]; ys=[true_stress]+[true_props[:,i] for i in range(4)]; ps=[pred_stress]+[pred_props[:,i] for i in range(4)]
    for name,y,p,log_flag in zip(["stress_kpa"]+PROPERTY_NAMES,ys,ps,[True,False,True,True,True]):
        row={"model":label,"target":name,"physical_r2":safe_r2(y,p),"physical_rmse":np.sqrt(mean_squared_error(y,p)),
             "physical_mae":mean_absolute_error(y,p),"bias":float(np.mean(p-y)),
             "relative_rmse_percent":float(100*np.sqrt(np.mean(((p-y)/y)**2))),
             "log10_r2":np.nan,"log10_rmse":np.nan,"log10_mae":np.nan,"multiplicative_factor":np.nan}
        if log_flag:
            ly,lp=np.log10(y),np.log10(np.clip(p,1e-30,None)); mae=mean_absolute_error(ly,lp)
            row.update({"log10_r2":safe_r2(ly,lp),"log10_rmse":np.sqrt(mean_squared_error(ly,lp)),"log10_mae":mae,"multiplicative_factor":10**mae})
        rows.append(row)
    return pd.DataFrame(rows)


def model_plots(history,true_ids,pred_ids,classes,true_stress,pred_stress,true_props,pred_props,metrics,model_dir,label):
    h=pd.DataFrame(history); h["global_epoch"]=np.arange(1,len(h)+1); h.to_csv(model_dir/"training_history.csv",index=False)
    fig,axes=plt.subplots(1,3,figsize=(10.2,3.2))
    axes[0].plot(h.global_epoch,h.train_loss,label="Train"); axes[0].plot(h.global_epoch,h.val_loss,label="Validation")
    axes[1].plot(h.global_epoch,h.train_accuracy,label="Train"); axes[1].plot(h.global_epoch,h.val_accuracy,label="Validation")
    axes[2].plot(h.global_epoch,h.train_regression_loss,label="Train"); axes[2].plot(h.global_epoch,h.val_regression_loss,label="Validation")
    for ax,title,ylabel in zip(axes,["(a) Total loss","(b) Classification","(c) Regression"],["Loss","Accuracy","Huber loss"]):
        ax.set(xlabel="Epoch",ylabel=ylabel,title=title); ax.legend(frameon=False); ax.grid(False); sns.despine(ax=ax)
    fig.suptitle(label); fig.tight_layout(); save_fig(fig,model_dir/"learning_curves")

    cm=confusion_matrix(true_ids,pred_ids,labels=np.arange(len(classes))); norm=cm/np.maximum(cm.sum(1,keepdims=True),1)
    fig,axes=plt.subplots(1,2,figsize=(9,3.9)); sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",square=True,xticklabels=classes,yticklabels=classes,cbar=False,ax=axes[0])
    sns.heatmap(norm,annot=True,fmt=".2f",cmap="Blues",vmin=0,vmax=1,square=True,xticklabels=classes,yticklabels=classes,ax=axes[1])
    axes[0].set(title="(a) Counts",xlabel="Predicted",ylabel="Measured"); axes[1].set(title="(b) Normalised",xlabel="Predicted",ylabel="Measured")
    fig.suptitle(label); fig.tight_layout(); save_fig(fig,model_dir/"confusion_matrix")

    fig,ax=plt.subplots(figsize=(4.8,4)); ax.scatter(true_stress,pred_stress,s=25,alpha=.7,color="#0072B2",edgecolors="white")
    lo,hi=min(true_stress.min(),pred_stress.min()),max(true_stress.max(),pred_stress.max()); ax.plot([lo,hi],[lo,hi],"k--")
    ax.set_xscale("log"); ax.set_yscale("log"); row=metrics[metrics.target=="stress_kpa"].iloc[0]
    ax.text(.04,.96,f"log-R² = {row.log10_r2:.3f}\nFactor = {row.multiplicative_factor:.2f}×",transform=ax.transAxes,va="top",bbox=dict(boxstyle="round",fc="white"))
    ax.set(xlabel="Measured stress (kPa)",ylabel="Predicted stress (kPa)",title=f"{label}: continuous stress"); ax.grid(False); sns.despine(ax=ax); fig.tight_layout(); save_fig(fig,model_dir/"stress_parity")

    fig,axes=plt.subplots(2,2,figsize=(8.2,7)); axes=axes.ravel(); colors=sns.color_palette("colorblind",4)
    for i,ax in enumerate(axes):
        y,p=true_props[:,i],pred_props[:,i]; ax.scatter(y,p,s=24,alpha=.7,color=colors[i],edgecolors="white")
        lo,hi=min(y.min(),p.min()),max(y.max(),p.max()); ax.plot([lo,hi],[lo,hi],"k--")
        if i in LOG_PROPERTY_INDICES: ax.set_xscale("log"); ax.set_yscale("log")
        row=metrics[metrics.target==PROPERTY_NAMES[i]].iloc[0]; ax.text(.04,.96,f"R²={row.physical_r2:.3f}\nRMSE={row.physical_rmse:.2e}",transform=ax.transAxes,va="top",bbox=dict(boxstyle="round",fc="white"))
        ax.set(xlabel="Assigned",ylabel="Estimated",title=f"({chr(97+i)}) {PROPERTY_LABELS[i]}"); ax.grid(False); sns.despine(ax=ax)
    fig.suptitle(label); fig.tight_layout(); save_fig(fig,model_dir/"property_parity")


def train_one(key,frames,encoder,scaler,config,device,root):
    label=MODEL_REGISTRY[key]["label"]; model_dir=root/key; model_dir.mkdir(parents=True,exist_ok=True); seed_all(config.seed)
    _,loaders=make_loaders(frames,encoder,scaler,config,device); model=HybridModel(key,len(encoder.classes_),config.dropout,config.pretrained).to(device)
    train_model=model
    if config.use_compile and hasattr(torch,"compile"):
        try: train_model=torch.compile(model); print(f"torch.compile enabled for {key}")
        except Exception as exc: warnings.warn(f"compile skipped for {key}: {exc}")
    start=time.perf_counter(); history=train_stage(train_model,loaders[0],loaders[1],config,device,model_dir,"head",config.head_epochs)
    history+=train_stage(train_model,loaders[0],loaders[1],config,device,model_dir,"fine_tune",config.fine_tune_epochs); minutes=(time.perf_counter()-start)/60
    torch.save({"model_key":key,"model_state":model.state_dict(),"config":asdict(config)},model_dir/"final_model.pth")
    raw=predict(model,loaders[2],config,device); order=np.argsort(raw["index"]); probs=raw["prob"][order]; true_ids=raw["true"][order]; pred_ids=probs.argmax(1)
    scaled=raw["pred_stress"][order].reshape(-1,1); pred_stress=np.clip(10**scaler.inverse_transform(scaled).ravel(),STRESS_LEVELS.min(),STRESS_LEVELS.max())
    test=frames[2]; true_stress=test.stress_kpa.to_numpy(float); true_props=test[PROPERTY_NAMES].to_numpy(float); pred_props=interpolate_properties(pred_stress)
    metrics=metrics_table(true_stress,pred_stress,true_props,pred_props,key); metrics.to_csv(model_dir/"test_metrics.csv",index=False)
    class_metrics={"model":key,"accuracy":accuracy_score(true_ids,pred_ids),"balanced_accuracy":balanced_accuracy_score(true_ids,pred_ids),"macro_f1":f1_score(true_ids,pred_ids,average="macro",zero_division=0)}
    pd.DataFrame([class_metrics]).to_csv(model_dir/"classification_metrics.csv",index=False)
    with open(model_dir/"classification_report.json","w") as f: json.dump(classification_report(true_ids,pred_ids,target_names=[str(int(x)) for x in encoder.classes_],output_dict=True,zero_division=0),f,indent=2)
    results=pd.DataFrame({"image_path":test.image_path,"true_stress_kpa":true_stress,"pred_class_stress_kpa":encoder.inverse_transform(pred_ids),"pred_stress_kpa":pred_stress,"confidence":probs.max(1)})
    for i,level in enumerate(encoder.classes_): results[f"probability_{int(level)}_kpa"]=probs[:,i]
    for i,name in enumerate(PROPERTY_NAMES): results[f"true_{name}"]=true_props[:,i]; results[f"pred_{name}"]=pred_props[:,i]
    results.to_csv(model_dir/"test_predictions.csv",index=False)
    model_plots(history,true_ids,pred_ids,[str(int(x)) for x in encoder.classes_],true_stress,pred_stress,true_props,pred_props,metrics,model_dir,label)
    return {"model":key,"label":label,"family":MODEL_REGISTRY[key]["family"],"parameters":sum(p.numel() for p in model.parameters()),"training_minutes":minutes,
            "accuracy":class_metrics["accuracy"],"balanced_accuracy":class_metrics["balanced_accuracy"],"macro_f1":class_metrics["macro_f1"],
            **{f"{row.target}_r2":row.physical_r2 for _,row in metrics.iterrows()},**{f"{row.target}_rmse":row.physical_rmse for _,row in metrics.iterrows()},
            "mean_r2":metrics.physical_r2.mean(),"mean_relative_rmse_percent":metrics.relative_rmse_percent.mean()}


def combined_plots(summary,output_dir):
    order=summary.sort_values("mean_r2",ascending=False).model.tolist(); palette=dict(zip(order,sns.color_palette("colorblind",len(order))))
    fig,axes=plt.subplots(1,2,figsize=(9.3,4)); long=summary.melt(id_vars="model",value_vars=["accuracy","balanced_accuracy","macro_f1"],var_name="metric",value_name="score")
    sns.barplot(data=long,x="metric",y="score",hue="model",hue_order=order,palette=palette,ax=axes[0]); axes[0].set(ylim=(0,1),title="(a) Stress classification",xlabel=None)
    ranked=summary.sort_values("mean_r2"); axes[1].barh(ranked.model,ranked.mean_r2,color=[palette[m] for m in ranked.model]); axes[1].set(title="(b) Mean property R²",xlabel="Mean R²")
    for ax in axes: ax.grid(False); sns.despine(ax=ax)
    axes[0].legend(frameon=False); fig.tight_layout(); save_fig(fig,Path(output_dir)/"comparison_01_classification_regression")

    rmse_cols=[f"{x}_rmse" for x in ["stress_kpa"]+PROPERTY_NAMES]; matrix=summary[["model"]+rmse_cols].copy()
    for col in rmse_cols: matrix[col]=matrix[col]/matrix[col].min()
    matrix=matrix.set_index("model"); matrix.columns=["Stress","Void ratio","k","cᵥ","mᵥ"]
    fig,ax=plt.subplots(figsize=(7.4,4.2)); sns.heatmap(matrix,annot=True,fmt=".2f",cmap="YlOrRd",vmin=1,cbar_kws={"label":"RMSE / best RMSE"},ax=ax)
    ax.set(title="Normalised prediction error, lower is better",xlabel=None,ylabel=None); fig.tight_layout(); save_fig(fig,Path(output_dir)/"comparison_02_rmse_heatmap")

    fig,ax=plt.subplots(figsize=(6.8,4)); ax.scatter(summary.training_minutes,summary.mean_r2,s=80,c=[palette[m] for m in summary.model])
    for _,row in summary.iterrows(): ax.annotate(row.model,(row.training_minutes,row.mean_r2),xytext=(5,4),textcoords="offset points")
    ax.set(xlabel="Training time (min)",ylabel="Mean R²",title="Performance and computational cost"); ax.grid(False); sns.despine(ax=ax); fig.tight_layout(); save_fig(fig,Path(output_dir)/"comparison_03_performance_cost")


def predict_best(summary,config,encoder,scaler,device,image_path,output_dir):
    key=summary.sort_values("mean_r2",ascending=False).iloc[0].model; ck=torch.load(Path(output_dir)/key/"final_model.pth",map_location=device)
    model=HybridModel(key,len(encoder.classes_),config.dropout,False).to(device); model.load_state_dict(ck["model_state"]); model.eval(); _,et=transforms_for(config.image_size)
    image = open_rgb_image(image_path)
    tensor=et(image).unsqueeze(0).to(device)
    with torch.no_grad(),amp_context(config,device): out=model(tensor)
    probs=torch.softmax(out["logits"],1).float().cpu().numpy()[0]; stress=float(np.clip(10**scaler.inverse_transform(out["log_stress"].float().cpu().numpy().reshape(-1,1))[0,0],100,6000)); props=interpolate_properties([stress])[0]
    result={"best_model":key,"image_path":str(Path(image_path).resolve()),"predicted_class_stress_kpa":float(encoder.inverse_transform([probs.argmax()])[0]),"confidence":float(probs.max()),"continuous_stress_kpa":stress}
    result.update({name:float(props[i]) for i,name in enumerate(PROPERTY_NAMES)}); result.update({f"probability_{int(level)}_kpa":float(probs[i]) for i,level in enumerate(encoder.classes_)})
    pd.DataFrame([result]).to_csv(Path(output_dir)/"unknown_image_prediction.csv",index=False); print(json.dumps(result,indent=2))


def main(config,predict_image=None):
    seed_all(config.seed); plot_style(); output=Path(config.output_dir); output.mkdir(parents=True,exist_ok=True); device=select_device(config.xpu_index)
    # Resolve model-specific input requirements before creating transforms, caches,
    # datasets or models. Torchvision ViT-B/16 uses a fixed 224-pixel input.
    if "vit_b_16" in config.model_keys:
        if config.image_size != 224:
            print(
                f"ViT-B/16 selected: changing shared image size "
                f"from {config.image_size} to 224 pixels."
            )
            config.image_size = 224
        if config.batch_size > 16:
            print(
                f"ViT-B/16 selected: reducing shared batch size "
                f"from {config.batch_size} to 16 for Intel XPU stability."
            )
            config.batch_size = 16

    # Rewrite the effective configuration after automatic adjustments so the
    # publication record matches the settings actually used.
    with open(output / "benchmark_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    print(
        f"Effective benchmark settings: image_size={config.image_size}, "
        f"batch_size={config.batch_size}, models={list(config.model_keys)}"
    )
    df=validate_csv(config.csv_path,output); frames=split_df(df); dataset_summary(frames,output)
    training=balance_df(frames[0],config.seed) if config.balance_training else frames[0]; frames=(training,frames[1],frames[2])
    encoder=LabelEncoder().fit(STRESS_LEVELS.astype(int)); scaler=StandardScaler().fit(training[["log10_stress"]].to_numpy(dtype=np.float64)); joblib.dump(encoder,output/"stress_encoder.joblib"); joblib.dump(scaler,output/"stress_scaler.joblib")
    rows=[]
    for key in config.model_keys:
        print(f"\n{'='*72}\n{MODEL_REGISTRY[key]['label']}\n{'='*72}"); rows.append(train_one(key,frames,encoder,scaler,config,device,output))
        if device.type=="xpu":
            try: torch.xpu.empty_cache()
            except Exception: pass
        elif device.type=="cuda": torch.cuda.empty_cache()
    summary=pd.DataFrame(rows).sort_values("mean_r2",ascending=False).reset_index(drop=True); summary.insert(0,"rank",np.arange(1,len(summary)+1)); summary.to_csv(output/"model_comparison_summary.csv",index=False); combined_plots(summary,output)
    print("\nFinal ranking\n",summary[["rank","model","family","accuracy","macro_f1","mean_r2","training_minutes"]].to_string(index=False))
    if predict_image: predict_best(summary,config,encoder,scaler,device,predict_image,output)
    print(f"\nCompleted: {output.resolve()}")


def parse_args():
    p=argparse.ArgumentParser(description="Combined Intel-XPU CNN and ViT SEM benchmark")
    p.add_argument("--csv",default=r"C:\Users\rt4\Documents\ML\CNN\code\data\csv_manifests\sem_dataset_all.csv")
    p.add_argument("--output-dir",default=r"C:\Users\rt4\Documents\ML\CNN\code\Combined_XPU_CNN_ViT_Results")
    p.add_argument("--models",nargs="+",choices=list(MODEL_REGISTRY),default=["mobilenet_v2","efficientnet_b0","resnet50","vit_b_16"])
    p.add_argument("--image-size",type=int,default=224); p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--head-epochs",type=int,default=5); p.add_argument("--fine-tune-epochs",type=int,default=10)
    p.add_argument("--head-lr",type=float,default=1e-3); p.add_argument("--backbone-lr",type=float,default=1e-5); p.add_argument("--fine-tune-head-lr",type=float,default=1e-4)
    p.add_argument("--weight-decay",type=float,default=1e-4); p.add_argument("--dropout",type=float,default=.30)
    p.add_argument("--classification-weight",type=float,default=1.0); p.add_argument("--regression-weight",type=float,default=1.0)
    p.add_argument("--patience",type=int,default=5); p.add_argument("--warmup-epochs",type=int,default=2); p.add_argument("--unfreeze-blocks",type=int,default=1)
    p.add_argument("--num-workers",type=int,default=0); p.add_argument("--prefetch-factor",type=int,default=2)
    p.add_argument("--xpu-index",type=int,default=0,help="Intel XPU adapter index; use 0 for Intel Arc GPU 0")
    p.add_argument("--amp-dtype",choices=["float16","bfloat16","float32"],default="float16"); p.add_argument("--compile",action="store_true")
    p.add_argument("--no-cache",action="store_true"); p.add_argument("--no-balance",action="store_true"); p.add_argument("--no-pretrained",action="store_true"); p.add_argument("--seed",type=int,default=SEED); p.add_argument("--predict-image",default=None)
    a=p.parse_args(); config=Config(csv_path=a.csv,output_dir=a.output_dir,model_keys=tuple(a.models),image_size=a.image_size,batch_size=a.batch_size,head_epochs=a.head_epochs,fine_tune_epochs=a.fine_tune_epochs,head_lr=a.head_lr,backbone_lr=a.backbone_lr,fine_tune_head_lr=a.fine_tune_head_lr,weight_decay=a.weight_decay,dropout=a.dropout,classification_weight=a.classification_weight,regression_weight=a.regression_weight,patience=a.patience,warmup_epochs=a.warmup_epochs,unfreeze_blocks=a.unfreeze_blocks,num_workers=a.num_workers,prefetch_factor=a.prefetch_factor,amp_dtype=a.amp_dtype,use_compile=a.compile,use_cache=not a.no_cache,balance_training=not a.no_balance,pretrained=not a.no_pretrained,seed=a.seed,xpu_index=a.xpu_index)
    return config,a.predict_image


if __name__=="__main__":
    cfg,prediction=parse_args(); main(cfg,prediction)
