#!/usr/bin/env python3
"""
Train a model to predict controller inputs from sequences of gameplay frames.
Phase 1: Extract frames to disk (one-time)
Phase 2: Train from extracted frames (fast)
"""

import os
import sys
import json
import glob
import math
import random
import argparse
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import av
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.io import read_image
from tqdm import tqdm
from PIL import Image


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
@dataclass
class Config:
    # Data
    dataset_root: str = "./nitrogen_dataset"
    extracted_frames_dir: str = "./extracted_frames"
    img_size: int = 224          # 224 instead of 256 (matches pretrained models exactly)
    seq_len: int = 32
    frame_skip: int = 2          # skip frames to cover more time with less compute

    # Architecture
    feature_dim: int = 256*2       # smaller = faster
    n_heads: int = 4
    n_transformer_layers: int = 4  # fewer layers
    dropout: float = 0.1

    # Training
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 75
    val_split: float = 0.1
    num_workers: int = 4         # increase for faster data loading
    grad_accum_steps: int = 2
    mixed_precision: bool = True

    # Loss weights
    button_loss_weight: float = 1.0
    joystick_loss_weight: float = 2.0

    # Misc
    save_dir: str = "./checkpoints"
    seed: int = 42
    log_every: int = 100
    save_every_epoch: int = 5

    # Outputs
    n_buttons: int = 17
    n_joystick: int = 4

    @property
    def n_actions(self):
        return self.n_buttons + self.n_joystick


BUTTON_NAMES = [
    "dpad_down", "dpad_left", "dpad_right", "dpad_up",
    "left_shoulder", "left_thumb", "left_trigger",
    "right_shoulder", "right_thumb", "right_trigger",
    "south", "west", "east", "north",
    "back", "start", "guide"
]


# ─────────────────────────────────────────────
# Phase 1: Extract frames to disk
# ─────────────────────────────────────────────
def extract_all_frames(dataset_root: str, output_dir: str, img_size: int = 224):
    """Extract all video frames to JPEGs on disk. Only needs to run once."""
    videos_root = os.path.join(dataset_root, "videos")
    
    video_files = []
    for ext in ("*.avi", "*.mp4", "*.mkv", "*.mov"):
        video_files.extend(glob.glob(
            os.path.join(videos_root, "**", ext), recursive=True
        ))
    video_files = sorted(video_files)
    
    print(f"Found {len(video_files)} videos to extract")
    
    for vf in tqdm(video_files, desc="Extracting videos"):
        rel = os.path.relpath(vf, videos_root)
        chunk_name = os.path.splitext(rel)[0]
        frame_dir = os.path.join(output_dir, chunk_name)
        
        # Skip if already extracted
        done_marker = os.path.join(frame_dir, "_done.txt")
        if os.path.isfile(done_marker):
            continue
        
        os.makedirs(frame_dir, exist_ok=True)
        
        try:
            container = av.open(vf)
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            
            frame_count = 0
            for frame in container.decode(stream):
                img = frame.to_ndarray(format="rgb24")
                img_pil = Image.fromarray(img)
                img_pil = img_pil.resize((img_size, img_size), Image.BILINEAR)
                img_pil.save(
                    os.path.join(frame_dir, f"{frame_count:06d}.jpg"),
                    quality=90
                )
                frame_count += 1
            
            container.close()
            
            # Mark as done
            with open(done_marker, "w") as f:
                f.write(str(frame_count))
            
        except Exception as e:
            print(f"Error extracting {vf}: {e}")
    
    print("Extraction complete!")


# ─────────────────────────────────────────────
# Phase 2: Fast Dataset from extracted frames
# ─────────────────────────────────────────────
class ChunkInfo:
    """Represents a single video chunk + its actions."""
    def __init__(self, frames_dir: str, actions_dir: str):
        self.frames_dir = frames_dir
        self.actions_dir = actions_dir

        meta_path = os.path.join(actions_dir, "metadata.json")
        with open(meta_path, "r") as f:
            self.metadata = json.load(f)

        # Count extracted frames
        self.frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
        self.n_extracted = len(self.frame_files)

        # Load actions
        parquet_path = os.path.join(actions_dir, "actions_raw.parquet")
        self.actions_df = pd.read_parquet(parquet_path)

        # Align frame count
        self.n_frames = min(self.n_extracted, len(self.actions_df))
        self.actions_df = self.actions_df.iloc[:self.n_frames]
        self.frame_files = self.frame_files[:self.n_frames]

        # Pre-parse all actions into a numpy array for speed
        self._precompute_actions()

    def _precompute_actions(self):
        """Parse all actions upfront into a single numpy array."""
        all_actions = np.zeros((self.n_frames, 21), dtype=np.float32)
        
        for i in range(self.n_frames):
            row = self.actions_df.iloc[i]
            
            # Buttons
            for j, name in enumerate(BUTTON_NAMES):
                all_actions[i, j] = float(row[name])
            
            # Joysticks
            j_left = row["j_left"]
            j_right = row["j_right"]
            
            if isinstance(j_left, str):
                j_left = json.loads(j_left)
            if isinstance(j_right, str):
                j_right = json.loads(j_right)
            if isinstance(j_left, (list, tuple, np.ndarray)):
                all_actions[i, 17] = float(j_left[0])
                all_actions[i, 18] = float(j_left[1])
            if isinstance(j_right, (list, tuple, np.ndarray)):
                all_actions[i, 19] = float(j_right[0])
                all_actions[i, 20] = float(j_right[1])
        
        self.all_actions = all_actions

    def get_action_vector(self, frame_idx: int) -> np.ndarray:
        return self.all_actions[frame_idx]


class FastGameplayDataset(Dataset):
    """
    Lightning-fast dataset that reads pre-extracted JPEGs.
    """
    def __init__(self, config: Config):
        self.config = config
        self.chunks: List[ChunkInfo] = []
        self.samples: List[Tuple[int, int]] = []

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self._discover_chunks()
        self._build_samples()

    def _discover_chunks(self):
        dataset_root = self.config.dataset_root
        extracted_root = self.config.extracted_frames_dir
        actions_root = os.path.join(dataset_root, "actions")
        videos_root = os.path.join(dataset_root, "videos")

        # Find all extracted frame directories
        video_files = []
        for ext in ("*.avi", "*.mp4", "*.mkv", "*.mov"):
            video_files.extend(glob.glob(
                os.path.join(videos_root, "**", ext), recursive=True
            ))
        video_files = sorted(video_files)

        print(f"Found {len(video_files)} video entries")

        for vf in video_files:
            rel = os.path.relpath(vf, videos_root)
            chunk_name = os.path.splitext(rel)[0]
            
            frames_dir = os.path.join(extracted_root, chunk_name)
            actions_dir = os.path.join(actions_root, chunk_name)

            if not os.path.isdir(frames_dir):
                continue
            if not os.path.isdir(actions_dir):
                continue
            
            done_marker = os.path.join(frames_dir, "_done.txt")
            if not os.path.isfile(done_marker):
                continue

            meta_path = os.path.join(actions_dir, "metadata.json")
            parquet_path = os.path.join(actions_dir, "actions_raw.parquet")
            if not os.path.isfile(meta_path) or not os.path.isfile(parquet_path):
                continue

            try:
                chunk = ChunkInfo(frames_dir, actions_dir)
                if chunk.n_frames > 0:
                    self.chunks.append(chunk)
            except Exception as e:
                print(f"Warning: Error loading {chunk_name}: {e}")

        print(f"Loaded {len(self.chunks)} valid chunks")

    def _build_samples(self):
        seq_len = self.config.seq_len
        frame_skip = self.config.frame_skip
        required_history = (seq_len - 1) * frame_skip

        for chunk_idx, chunk in enumerate(self.chunks):
            for target_frame in range(required_history, chunk.n_frames):
                self.samples.append((chunk_idx, target_frame))

        print(f"Total training samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chunk_idx, target_frame = self.samples[idx]
        chunk = self.chunks[chunk_idx]

        frame_skip = self.config.frame_skip
        seq_len = self.config.seq_len
        frame_indices = [
            target_frame - (seq_len - 1 - i) * frame_skip
            for i in range(seq_len)
        ]

        # Load frames from disk (FAST - just JPEG reads)
        frame_tensors = []
        for fi in frame_indices:
            fi = max(0, min(fi, chunk.n_frames - 1))
            img_path = chunk.frame_files[fi]
            
            # Read JPEG and transform
            img = Image.open(img_path)
            frame_tensor = self.transform(img)
            frame_tensors.append(frame_tensor)

        frames = torch.stack(frame_tensors)
        action = torch.from_numpy(chunk.get_action_vector(target_frame))

        return frames, action


# ─────────────────────────────────────────────
# Model (optimized for speed)
# ─────────────────────────────────────────────
class FrameEncoder(nn.Module):
    """Lightweight CNN encoder using MobileNetV3-Small."""
    def __init__(self, feature_dim: int = 256):
        super().__init__()
        # MobileNetV3-Small: ~2.5M params, MUCH faster than EfficientNet
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        # MobileNetV3-Small outputs 576 channels
        self.proj = nn.Sequential(
            nn.Linear(576, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = self.pool(feat)
        feat = feat.flatten(1)
        feat = self.proj(feat)
        return feat


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class GameplayModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.frame_encoder = FrameEncoder(feature_dim=config.feature_dim)
        self.pos_enc = PositionalEncoding(config.feature_dim, max_len=512)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.feature_dim,
            nhead=config.n_heads,
            dim_feedforward=config.feature_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_transformer_layers,
        )

        self.button_head = nn.Sequential(
            nn.Linear(config.feature_dim, config.feature_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feature_dim // 2, config.n_buttons),
        )

        self.joystick_head = nn.Sequential(
            nn.Linear(config.feature_dim, config.feature_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feature_dim // 2, config.n_joystick),
            nn.Tanh(),
        )

    def forward(self, frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, C, H, W = frames.shape

        # Encode all frames at once
        frames_flat = frames.view(B * S, C, H, W)
        features_flat = self.frame_encoder(frames_flat)
        features = features_flat.view(B, S, -1)

        features = self.pos_enc(features)
        temporal_out = self.temporal_transformer(features)
        last_feature = temporal_out[:, -1, :]

        button_logits = self.button_head(last_feature)
        joystick_values = self.joystick_head(last_feature)

        return button_logits, joystick_values


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
def compute_loss(button_logits, joystick_pred, targets, config):
    button_targets = targets[:, :config.n_buttons]
    joystick_targets = targets[:, config.n_buttons:]

    button_loss = F.binary_cross_entropy_with_logits(button_logits, button_targets)
    joystick_loss = F.mse_loss(joystick_pred, joystick_targets)

    total_loss = (
        config.button_loss_weight * button_loss
        + config.joystick_loss_weight * joystick_loss
    )

    with torch.no_grad():
        button_preds = (torch.sigmoid(button_logits) > 0.5).float()
        button_acc = (button_preds == button_targets).float().mean()

    metrics = {
        "button_loss": button_loss.item(),
        "joystick_loss": joystick_loss.item(),
        "total_loss": total_loss.item(),
        "button_acc": button_acc.item(),
    }
    return total_loss, metrics


def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, config, device, epoch):
    model.train()
    total_metrics = {"button_loss": 0, "joystick_loss": 0, "total_loss": 0, "button_acc": 0}
    n_batches = 0
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]")
    for batch_idx, (frames, actions) in enumerate(pbar):
        frames = frames.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=config.mixed_precision):
            button_logits, joystick_pred = model(frames)
            loss, metrics = compute_loss(button_logits, joystick_pred, actions, config)
            loss = loss / config.grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % config.grad_accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        for k, v in metrics.items():
            total_metrics[k] += v
        n_batches += 1

        if (batch_idx + 1) % config.log_every == 0:
            pbar.set_postfix({
                "loss": f"{metrics['total_loss']:.4f}",
                "btn_acc": f"{metrics['button_acc']:.3f}",
                "joy": f"{metrics['joystick_loss']:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.1e}",
            })

    return {k: v / max(n_batches, 1) for k, v in total_metrics.items()}


@torch.no_grad()
def validate(model, dataloader, config, device, epoch):
    model.eval()
    total_metrics = {"button_loss": 0, "joystick_loss": 0, "total_loss": 0, "button_acc": 0}
    n_batches = 0

    for frames, actions in tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs} [Val]"):
        frames = frames.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=config.mixed_precision):
            button_logits, joystick_pred = model(frames)
            _, metrics = compute_loss(button_logits, joystick_pred, actions, config)

        for k, v in metrics.items():
            total_metrics[k] += v
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in total_metrics.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="./nitrogen_dataset")
    parser.add_argument("--extracted_frames", type=str, default="./extracted_frames")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--frame_skip", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--n_transformer_layers", type=int, default=3)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--extract_only", action="store_true",
                        help="Only extract frames, don't train")
    parser.add_argument("--skip_extract", action="store_true",
                        help="Skip extraction (already done)")
    args = parser.parse_args()

    # Resolve paths relative to script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = args.dataset
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(script_dir, dataset_path)
    extracted_path = args.extracted_frames
    if not os.path.isabs(extracted_path):
        extracted_path = os.path.join(script_dir, extracted_path)

    config = Config(
        dataset_root=dataset_path,
        extracted_frames_dir=extracted_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        frame_skip=args.frame_skip,
        lr=args.lr,
        img_size=args.img_size,
        feature_dim=args.feature_dim,
        n_transformer_layers=args.n_transformer_layers,
        grad_accum_steps=args.grad_accum,
        num_workers=args.num_workers,
        save_dir=args.save_dir,
    )

    # Seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.cuda.manual_seed_all(config.seed)

    # ── Phase 1: Extract frames ──
    if not args.skip_extract:
        print("\n" + "=" * 60)
        print("PHASE 1: Extracting frames to disk (one-time)")
        print("=" * 60)
        extract_all_frames(dataset_path, extracted_path, config.img_size)

    if args.extract_only:
        print("Extraction done. Exiting.")
        return

    # ── Phase 2: Train ──
    print("\n" + "=" * 60)
    print("PHASE 2: Training")
    print("=" * 60)

    print(f"\nConfig:")
    print(f"  seq_len={config.seq_len}, frame_skip={config.frame_skip}")
    print(f"  context window = {config.seq_len * config.frame_skip / 30:.1f}s at 30fps")
    print(f"  batch_size={config.batch_size}, grad_accum={config.grad_accum_steps}")
    print(f"  effective_batch={config.batch_size * config.grad_accum_steps}")
    print(f"  feature_dim={config.feature_dim}, transformer_layers={config.n_transformer_layers}")
    print(f"  img_size={config.img_size}")
    print()

    dataset = FastGameplayDataset(config)
    if len(dataset) == 0:
        print("ERROR: No samples found!")
        sys.exit(1)

    val_size = max(1, int(len(dataset) * config.val_split))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )
    print(f"Train: {train_size}, Val: {val_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=config.num_workers > 0,
        prefetch_factor=4 if config.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
        prefetch_factor=4 if config.num_workers > 0 else None,
    )

    model = GameplayModel(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,} ({total_params * 4 / 1e6:.1f} MB)")

    # Compile model for speed (PyTorch 2.0+, Linux only - requires Triton)
    if hasattr(torch, "compile") and sys.platform != "win32":
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("✓ torch.compile enabled")
        except Exception as e:
            print(f"torch.compile not available: {e}")
    else:
        print("Skipping torch.compile (Windows or not available)")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    total_steps = len(train_loader) * config.epochs // config.grad_accum_steps
    warmup_steps = min(500, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda") if config.mixed_precision and device.type == "cuda" else None

    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        # Handle compiled model state dict
        state_dict = ckpt["model_state_dict"]
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            # Remove _orig_mod. prefix from compiled model
            new_sd = {}
            for k, v in state_dict.items():
                new_sd[k.replace("_orig_mod.", "")] = v
            model.load_state_dict(new_sd)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        if "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"]:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    os.makedirs(config.save_dir, exist_ok=True)
    with open(os.path.join(config.save_dir, "config.json"), "w") as f:
        json.dump(vars(config), f, indent=2)

    # Estimate time
    print(f"\nEstimated steps per epoch: {len(train_loader)}")
    print(f"Total epochs: {config.epochs - start_epoch}")
    print()

    for epoch in range(start_epoch, config.epochs):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, config, device, epoch
        )
        val_metrics = validate(model, val_loader, config, device, epoch)
        elapsed = time.time() - t0

        remaining = elapsed * (config.epochs - epoch - 1)
        remaining_str = f"{remaining/3600:.1f}h" if remaining > 3600 else f"{remaining/60:.0f}m"

        print(f"\nEpoch {epoch+1}/{config.epochs} ({elapsed:.0f}s, ~{remaining_str} remaining)")
        print(f"  Train - Loss: {train_metrics['total_loss']:.4f}, "
              f"Btn Acc: {train_metrics['button_acc']:.3f}, "
              f"Joy: {train_metrics['joystick_loss']:.4f}")
        print(f"  Val   - Loss: {val_metrics['total_loss']:.4f}, "
              f"Btn Acc: {val_metrics['button_acc']:.3f}, "
              f"Joy: {val_metrics['joystick_loss']:.4f}")

        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            save_path = os.path.join(config.save_dir, "best_model.pt")
            state_dict = model.state_dict()
            # Remove _orig_mod prefix if compiled
            clean_sd = {}
            for k, v in state_dict.items():
                clean_sd[k.replace("_orig_mod.", "")] = v
            torch.save({
                "epoch": epoch,
                "model_state_dict": clean_sd,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "config": vars(config),
            }, save_path)
            print(f"  ✓ Best model saved (val_loss={best_val_loss:.4f})")

        if (epoch + 1) % config.save_every_epoch == 0:
            state_dict = model.state_dict()
            clean_sd = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            torch.save({
                "epoch": epoch,
                "model_state_dict": clean_sd,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "config": vars(config),
            }, os.path.join(config.save_dir, f"ckpt_epoch_{epoch+1}.pt"))

        print()

    print(f"Done! Best val loss: {best_val_loss:.4f}")


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────
class InferenceEngine:
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        ckpt = torch.load(checkpoint_path, map_location=device)
        saved_config = ckpt.get("config", {})
        self.config = Config(**{k: v for k, v in saved_config.items() if hasattr(Config, k)})
        self.device = torch.device(device)

        self.model = GameplayModel(self.config).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.config.img_size, self.config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.frame_buffer = []

    def reset(self):
        self.frame_buffer = []

    @torch.no_grad()
    def predict(self, frame: np.ndarray) -> dict:
        frame_tensor = self.transform(frame)
        self.frame_buffer.append(frame_tensor)
        if len(self.frame_buffer) > self.config.seq_len:
            self.frame_buffer = self.frame_buffer[-self.config.seq_len:]
        while len(self.frame_buffer) < self.config.seq_len:
            self.frame_buffer.insert(0, self.frame_buffer[0])

        frames = torch.stack(self.frame_buffer).unsqueeze(0).to(self.device)

        with torch.amp.autocast("cuda"):
            button_logits, joystick_pred = self.model(frames)

        buttons = (torch.sigmoid(button_logits[0]) > 0.5).cpu().numpy().astype(bool)
        joystick = joystick_pred[0].cpu().numpy()

        result = {}
        for i, name in enumerate(BUTTON_NAMES):
            result[name] = bool(buttons[i])
        result["j_left"] = [float(joystick[0]), float(joystick[1])]
        result["j_right"] = [float(joystick[2]), float(joystick[3])]
        return result


if __name__ == "__main__":
    main()
