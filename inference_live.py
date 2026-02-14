#!/usr/bin/env python3
"""
NitroGen Live Inference: Camera → AI/Manual → Pico UDP Controller
==================================================================
Captures live video from a camera, runs the trained model to predict
controller inputs OR reads from an Xbox controller, and sends them to
the Raspberry Pi Pico over UDP.

Toggle between AI and Manual (Xbox controller) mode with TAB key.

Press Q or ESC in the preview window to stop.

Usage:
    python infer_live.py --checkpoint ./checkpoints/full_inference_model.pt
    python infer_live.py --checkpoint ./checkpoints/full_inference_model.pt --camera 1
    python infer_live.py --checkpoint ./checkpoints/full_inference_model.pt --no-send
"""

import os
import sys
import json
import math
import struct
import socket
import time
import argparse
import threading
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models


# ═══════════════════════════════════════════════════════════════
# SETTINGS (mirrors the recorder / trainer)
# ═══════════════════════════════════════════════════════════════

PICO_IP = "192.168.1.44"
PICO_PORT = 4210
PLAYER_ID = 1
SEND_RATE_HZ = 30  # match camera FPS for inference

DEADZONE = 4000
TRIGGER_THRESHOLD = 30

BUTTON_NAMES = [
    "dpad_down", "dpad_left", "dpad_right", "dpad_up",
    "left_shoulder", "left_thumb", "left_trigger",
    "right_shoulder", "right_thumb", "right_trigger",
    "south", "west", "east", "north",
    "back", "start", "guide",
]

# Pico packet button bits (same mapping as the recorder)
PICO_BUTTON_BITS = {
    "north":          0x0001,
    "west":           0x0002,
    "south":          0x0004,
    "east":           0x0008,
    "left_shoulder":  0x0010,
    "right_shoulder": 0x0020,
    "left_trigger":   0x0040,
    "right_trigger":  0x0080,
    "back":           0x0100,
    "start":          0x0200,
    "left_thumb":     0x0400,
    "right_thumb":    0x0800,
    "guide":          0x1000,
}

# Button bits for building Pico packets from controller state
BUTTON_BITS = {
    "BTN_SOUTH":  0x0004,
    "BTN_EAST":   0x0008,
    "BTN_NORTH":  0x0001,
    "BTN_WEST":   0x0002,
    "BTN_TL":     0x0010,
    "BTN_TR":     0x0020,
    "BTN_TL2":    0x0040,
    "BTN_TR2":    0x0080,
    "BTN_SELECT": 0x0100,
    "BTN_START":  0x0200,
    "BTN_THUMBL": 0x0400,
    "BTN_THUMBR": 0x0800,
    "BTN_MODE":   0x1000,
}

HAT_TABLE = {
    (0, -1): 0, (1, -1): 1, (1, 0): 2, (1, 1): 3,
    (0, 1): 4, (-1, 1): 5, (-1, 0): 6, (-1, -1): 7,
    (0, 0): 8,
}


# ═══════════════════════════════════════════════════════════════
# INPUT MODE ENUM
# ═══════════════════════════════════════════════════════════════

class InputMode(Enum):
    AI = "AI"
    MANUAL = "MANUAL"


# ═══════════════════════════════════════════════════════════════
# CONFIG (must match training)
# ═══════════════════════════════════════════════════════════════

@dataclass
class Config:
    img_size: int = 224
    seq_len: int = 16
    frame_skip: int = 2
    feature_dim: int = 256
    n_heads: int = 4
    n_transformer_layers: int = 3
    dropout: float = 0.1
    n_buttons: int = 17
    n_joystick: int = 4
    backbone_dim: int = 576

    @property
    def n_actions(self):
        return self.n_buttons + self.n_joystick


# ═══════════════════════════════════════════════════════════════
# CONTROLLER STATE (from recorder)
# ═══════════════════════════════════════════════════════════════

@dataclass
class ControllerState:
    """Raw Xbox controller state."""
    # For Pico packet
    buttons: int = 0
    hat_x: int = 0
    hat_y: int = 0
    lx: int = 0
    ly: int = 0
    rx: int = 0
    ry: int = 0
    lt: int = 0
    rt: int = 0

    # For tracking booleans
    btn_south: bool = False
    btn_east: bool = False
    btn_west: bool = False
    btn_north: bool = False
    btn_tl: bool = False
    btn_tr: bool = False
    btn_thumbl: bool = False
    btn_thumbr: bool = False
    btn_select: bool = False
    btn_start: bool = False
    btn_mode: bool = False

    def to_pico_packet(self, player_id: int) -> bytes:
        """Build the 7-byte UDP packet for the Pico."""
        buttons = self.buttons
        if self.lt > TRIGGER_THRESHOLD:
            buttons |= BUTTON_BITS.get("BTN_TL2", 0)
        if self.rt > TRIGGER_THRESHOLD:
            buttons |= BUTTON_BITS.get("BTN_TR2", 0)

        hat = HAT_TABLE.get((self.hat_x, self.hat_y), 0x08)

        # Invert Y axes: Xbox up is negative, Switch up is positive
        lx = self._stick_to_u8(self.lx)
        ly = self._stick_to_u8(-self.ly)
        rx = self._stick_to_u8(self.rx)
        ry = self._stick_to_u8(-self.ry)

        return struct.pack('<BHBBBBB', player_id, buttons, hat, lx, ly, rx, ry)

    def to_action_dict(self) -> dict:
        """Convert to action dict format for visualization."""
        dz = 0.08

        def stick_norm(raw: int) -> float:
            v = raw / 32768.0
            return 0.0 if abs(v) < dz else round(v, 4)

        return {
            "dpad_down":       self.hat_y == 1,
            "dpad_left":       self.hat_x == -1,
            "dpad_right":      self.hat_x == 1,
            "dpad_up":         self.hat_y == -1,
            "left_shoulder":   self.btn_tl,
            "left_thumb":      self.btn_thumbl,
            "left_trigger":    self.lt > TRIGGER_THRESHOLD,
            "right_shoulder":  self.btn_tr,
            "right_thumb":     self.btn_thumbr,
            "right_trigger":   self.rt > TRIGGER_THRESHOLD,
            "south":           self.btn_south,
            "west":            self.btn_west,
            "east":            self.btn_east,
            "north":           self.btn_north,
            "back":            self.btn_select,
            "start":           self.btn_start,
            "guide":           self.btn_mode,
            "j_left":  [stick_norm(self.lx), stick_norm(-self.ly)],
            "j_right": [stick_norm(self.rx), stick_norm(-self.ry)],
        }

    @staticmethod
    def _stick_to_u8(raw: int) -> int:
        raw = raw if abs(raw) >= DEADZONE else 0
        return max(0, min(255, int(((raw + 32768) / 65535) * 255)))


# ═══════════════════════════════════════════════════════════════
# CONTROLLER READER (from recorder)
# ═══════════════════════════════════════════════════════════════

class ControllerReader:
    """Background thread that reads Xbox controller events."""

    def __init__(self):
        self.state = ControllerState()
        self.lock = threading.Lock()
        self.running = False
        self.gamepad = None
        self.available = False

    def find(self) -> bool:
        try:
            import inputs
            pads = inputs.devices.gamepads
            if not pads:
                return False
            self.gamepad = pads[0]
            self.available = True
            print(f"  Found controller: {self.gamepad.name}")
            return True
        except ImportError:
            print("  [inputs library not found - pip install inputs]")
            return False
        except Exception as e:
            print(f"  Controller error: {e}")
            return False

    def start(self):
        if not self.available:
            return
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self.running = False

    def snapshot(self) -> ControllerState:
        """Return a copy of the current state."""
        with self.lock:
            import copy
            return copy.copy(self.state)

    def _loop(self):
        while self.running:
            try:
                events = self.gamepad.read()
                with self.lock:
                    for e in events:
                        self._process(e)
            except EOFError:
                print("\n  Controller disconnected!")
                self.running = False
                self.available = False
            except Exception:
                time.sleep(0.001)

    def _process(self, e):
        s = self.state
        if e.ev_type == "Key":
            pressed = bool(e.state)
            bit = BUTTON_BITS.get(e.code, 0)
            if pressed:
                s.buttons |= bit
            else:
                s.buttons &= ~bit

            if e.code == "BTN_SOUTH":    s.btn_south = pressed
            elif e.code == "BTN_EAST":   s.btn_east = pressed
            elif e.code == "BTN_WEST":   s.btn_west = pressed
            elif e.code == "BTN_NORTH":  s.btn_north = pressed
            elif e.code == "BTN_TL":     s.btn_tl = pressed
            elif e.code == "BTN_TR":     s.btn_tr = pressed
            elif e.code == "BTN_THUMBL": s.btn_thumbl = pressed
            elif e.code == "BTN_THUMBR": s.btn_thumbr = pressed
            elif e.code == "BTN_SELECT": s.btn_select = pressed
            elif e.code == "BTN_START":  s.btn_start = pressed
            elif e.code == "BTN_MODE":   s.btn_mode = pressed

        elif e.ev_type == "Absolute":
            c, v = e.code, e.state
            if   c == "ABS_X":     s.lx = v
            elif c == "ABS_Y":     s.ly = v
            elif c == "ABS_RX":    s.rx = v
            elif c == "ABS_RY":    s.ry = v
            elif c == "ABS_Z":     s.lt = v
            elif c == "ABS_RZ":    s.rt = v
            elif c == "ABS_HAT0X": s.hat_x = v
            elif c == "ABS_HAT0Y": s.hat_y = v


# ═══════════════════════════════════════════════════════════════
# MODEL DEFINITIONS (identical to training script)
# ═══════════════════════════════════════════════════════════════

class FrameEncoderBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        feat = self.features(x)
        feat = self.pool(feat)
        feat = feat.flatten(1)
        return feat


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TemporalActionModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.input_proj = nn.Sequential(
            nn.Linear(config.backbone_dim, config.feature_dim),
            nn.LayerNorm(config.feature_dim),
            nn.GELU(),
        )
        self.pos_enc = PositionalEncoding(config.feature_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.feature_dim,
            nhead=config.n_heads,
            dim_feedforward=config.feature_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.n_transformer_layers
        )
        self.button_head = nn.Sequential(
            nn.Linear(config.feature_dim, config.feature_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feature_dim, config.n_buttons),
        )
        self.joystick_head = nn.Sequential(
            nn.Linear(config.feature_dim, config.feature_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feature_dim, config.n_joystick),
            nn.Tanh(),
        )

    def forward(self, features):
        x = self.input_proj(features)
        x = self.pos_enc(x)
        x = self.transformer(x)
        last = x[:, -1, :]
        buttons = self.button_head(last)
        joystick = self.joystick_head(last)
        return buttons, joystick


class FullInferenceModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.backbone = FrameEncoderBackbone()
        self.temporal = TemporalActionModel(config)

    def forward(self, frames):
        B, S, C, H, W = frames.shape
        flat = frames.view(B * S, C, H, W)
        feats = self.backbone(flat)
        feats = feats.view(B, S, -1)
        return self.temporal(feats)


# ═══════════════════════════════════════════════════════════════
# PREDICTION → PICO PACKET CONVERSION
# ═══════════════════════════════════════════════════════════════

def prediction_to_pico_packet(pred: dict, player_id: int) -> bytes:
    """Convert a model prediction dict into the 7-byte UDP packet."""
    # ── Buttons ──
    buttons = 0
    for name, bit in PICO_BUTTON_BITS.items():
        if pred.get(name, False):
            buttons |= bit

    # ── D-pad → hat ──
    hat_x = 0
    hat_y = 0
    if pred.get("dpad_left", False):
        hat_x = -1
    elif pred.get("dpad_right", False):
        hat_x = 1
    if pred.get("dpad_up", False):
        hat_y = -1
    elif pred.get("dpad_down", False):
        hat_y = 1
    hat = HAT_TABLE.get((hat_x, hat_y), 8)

    # ── Joysticks ──
    jl = pred.get("j_left", [0.0, 0.0])
    jr = pred.get("j_right", [0.0, 0.0])

    def axis_to_u8(val: float, invert: bool = False) -> int:
        if invert:
            val = -val
        return max(0, min(255, int((val + 1.0) * 0.5 * 255)))

    lx = axis_to_u8(jl[0], invert=False)
    ly = axis_to_u8(jl[1], invert=True)
    rx = axis_to_u8(jr[0], invert=False)
    ry = axis_to_u8(jr[1], invert=True)

    return struct.pack("<BHBBBBB", player_id, buttons, hat, lx, ly, rx, ry)


# ═══════════════════════════════════════════════════════════════
# CAMERA HELPERS
# ═══════════════════════════════════════════════════════════════

def discover_cameras(max_idx: int = 10) -> List[dict]:
    cameras = []
    for i in range(max_idx):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cameras.append({"index": i, "width": w, "height": h, "fps": fps})
            cap.release()
    if cameras:
        print("\n  AVAILABLE CAMERAS:")
        for c in cameras:
            print(f"    [{c['index']}]  {c['width']}x{c['height']} @ {c['fps']:.0f}fps")
        print()
    else:
        print("\n  No cameras found!\n")
    return cameras


# ═══════════════════════════════════════════════════════════════
# INPUT VISUALISER
# ═══════════════════════════════════════════════════════════════

def draw_input_panel(
    action: dict, mode: InputMode, panel_w: int = 300, panel_h: int = 480
) -> np.ndarray:
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)

    white = (255, 255, 255)
    green = (0, 255, 0)
    red = (0, 0, 255)
    cyan = (255, 200, 0)
    gray = (100, 100, 100)
    yellow = (0, 255, 255)
    magenta = (255, 0, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Title with mode indicator
    mode_color = cyan if mode == InputMode.AI else magenta
    mode_text = "AI OUTPUT" if mode == InputMode.AI else "XBOX CONTROLLER"
    cv2.putText(panel, mode_text, (10, 24), font, 0.50, mode_color, 1)
    cv2.line(panel, (10, 32), (panel_w - 10, 32), gray, 1)

    # ── Sticks ──
    jl = action.get("j_left", [0.0, 0.0])
    jr = action.get("j_right", [0.0, 0.0])

    for label, jx, jy, cx in [
        ("L-STICK", jl[0], jl[1], 80),
        ("R-STICK", jr[0], jr[1], 220),
    ]:
        cy_base = 100
        radius = 45
        cv2.circle(panel, (cx, cy_base), radius, gray, 1)
        cv2.line(
            panel, (cx - radius, cy_base), (cx + radius, cy_base), (50, 50, 50), 1
        )
        cv2.line(
            panel, (cx, cy_base - radius), (cx, cy_base + radius), (50, 50, 50), 1
        )
        dx = int(jx * radius)
        dy = int(-jy * radius)
        dot_color = green if (abs(jx) > 0.01 or abs(jy) > 0.01) else gray
        cv2.circle(panel, (cx + dx, cy_base + dy), 6, dot_color, -1)
        cv2.putText(
            panel, label, (cx - 30, cy_base + radius + 18), font, 0.35, white, 1
        )
        cv2.putText(
            panel,
            f"X:{jx:+.2f}",
            (cx - 30, cy_base + radius + 34),
            font,
            0.3,
            white,
            1,
        )
        cv2.putText(
            panel,
            f"Y:{jy:+.2f}",
            (cx - 30, cy_base + radius + 48),
            font,
            0.3,
            white,
            1,
        )

    # ── Triggers ──
    y_trig = 195
    lt_on = action.get("left_trigger", False)
    rt_on = action.get("right_trigger", False)
    cv2.putText(panel, "LT", (20, y_trig), font, 0.4, green if lt_on else gray, 1)
    cv2.rectangle(
        panel,
        (50, y_trig - 12),
        (130, y_trig),
        green if lt_on else gray,
        -1 if lt_on else 1,
    )
    cv2.putText(panel, "RT", (160, y_trig), font, 0.4, green if rt_on else gray, 1)
    cv2.rectangle(
        panel,
        (190, y_trig - 12),
        (270, y_trig),
        green if rt_on else gray,
        -1 if rt_on else 1,
    )

    # ── D-Pad ──
    y_dpad = 240
    cv2.putText(panel, "D-PAD", (10, y_dpad), font, 0.4, white, 1)
    dpad_cx, dpad_cy = 55, y_dpad + 35
    ds = 16
    directions = [
        ("U", 0, -1, action.get("dpad_up", False)),
        ("D", 0, 1, action.get("dpad_down", False)),
        ("L", -1, 0, action.get("dpad_left", False)),
        ("R", 1, 0, action.get("dpad_right", False)),
    ]
    for lbl, ox, oy, on in directions:
        px, py = dpad_cx + ox * (ds * 2), dpad_cy + oy * (ds * 2)
        color = green if on else gray
        cv2.rectangle(
            panel, (px - ds, py - ds), (px + ds, py + ds), color, -1 if on else 1
        )
        cv2.putText(
            panel, lbl, (px - 5, py + 5), font, 0.35, (0, 0, 0) if on else white, 1
        )

    # ── Face buttons ──
    face_cx, face_cy = 220, y_dpad + 35
    bs = 14
    face_buttons = [
        ("N", 0, -1, action.get("north", False)),
        ("S", 0, 1, action.get("south", False)),
        ("W", -1, 0, action.get("west", False)),
        ("E", 1, 0, action.get("east", False)),
    ]
    for lbl, ox, oy, on in face_buttons:
        px = face_cx + ox * (bs * 2 + 4)
        py = face_cy + oy * (bs * 2 + 4)
        color = red if on else gray
        cv2.circle(panel, (px, py), bs, color, -1 if on else 1)
        cv2.putText(
            panel, lbl, (px - 5, py + 5), font, 0.35, (0, 0, 0) if on else white, 1
        )
    cv2.putText(panel, "FACE", (195, y_dpad), font, 0.4, white, 1)

    # ── Shoulder / misc buttons ──
    y_btn = 340
    cv2.putText(panel, "BUTTONS", (10, y_btn), font, 0.4, white, 1)
    btn_list = [
        ("LB", action.get("left_shoulder", False)),
        ("RB", action.get("right_shoulder", False)),
        ("LS", action.get("left_thumb", False)),
        ("RS", action.get("right_thumb", False)),
        ("BK", action.get("back", False)),
        ("ST", action.get("start", False)),
        ("GD", action.get("guide", False)),
    ]
    bx = 15
    for lbl, on in btn_list:
        color = yellow if on else gray
        cv2.rectangle(
            panel, (bx, y_btn + 10), (bx + 32, y_btn + 30), color, -1 if on else 1
        )
        cv2.putText(
            panel,
            lbl,
            (bx + 3, y_btn + 25),
            font,
            0.33,
            (0, 0, 0) if on else white,
            1,
        )
        bx += 40

    # ── Active list ──
    y_active = 395
    cv2.line(panel, (10, y_active - 8), (panel_w - 10, y_active - 8), gray, 1)
    cv2.putText(panel, "ACTIVE:", (10, y_active + 8), font, 0.4, cyan, 1)
    active = [n for n in BUTTON_NAMES if action.get(n)]
    if active:
        line = ""
        ly_off = y_active + 26
        for name in active:
            if len(line) + len(name) + 2 > 35:
                cv2.putText(panel, line, (10, ly_off), font, 0.3, green, 1)
                ly_off += 14
                line = ""
            line += name + "  "
        if line:
            cv2.putText(panel, line, (10, ly_off), font, 0.3, green, 1)
    else:
        cv2.putText(panel, "(idle)", (80, y_active + 8), font, 0.4, gray, 1)

    # ── Mode toggle hint ──
    cv2.putText(panel, "[TAB] Toggle Mode", (85, panel_h - 10), font, 0.35, gray, 1)

    return panel


def draw_hud(
    frame: np.ndarray, fps: float, latency_ms: float, sending: bool,
    mode: InputMode, controller_available: bool
) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # Status line 1
    status = "SENDING" if sending else "PREVIEW ONLY"
    color = (0, 255, 0) if sending else (0, 200, 255)
    cv2.putText(
        frame,
        f"{status}  |  {fps:.0f} fps  |  {latency_ms:.0f}ms",
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
    )

    # Status line 2 - Mode indicator
    if mode == InputMode.AI:
        mode_text = "MODE: AI (Neural Network)"
        mode_color = (255, 200, 0)  # cyan
    else:
        mode_text = "MODE: MANUAL (Xbox Controller)"
        mode_color = (255, 0, 255)  # magenta
        if not controller_available:
            mode_text += " [DISCONNECTED]"
            mode_color = (0, 0, 255)  # red

    cv2.putText(
        frame,
        mode_text,
        (8, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        mode_color,
        1,
    )

    # Toggle hint on right side
    cv2.putText(
        frame,
        "[TAB] Toggle",
        (w - 110, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (150, 150, 150),
        1,
    )

    return frame


# ═══════════════════════════════════════════════════════════════
# INFERENCE ENGINE
# ═══════════════════════════════════════════════════════════════

class LiveInferenceEngine:
    """Loads the full_inference_model.pt and runs prediction on frames."""

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = torch.device(device)

        print(f"  Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        cfg_dict = ckpt.get("config", {})
        self.config = Config(
            **{k: v for k, v in cfg_dict.items() if hasattr(Config, k)}
        )

        print(f"  Config: seq_len={self.config.seq_len}, "
              f"frame_skip={self.config.frame_skip}, "
              f"feature_dim={self.config.feature_dim}, "
              f"layers={self.config.n_transformer_layers}")

        self.model = FullInferenceModel(self.config).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Model params: {total_params:,}")

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(
                    (self.config.img_size, self.config.img_size)
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.frame_buffer: List[torch.Tensor] = []
        self.frame_counter = 0
        self.button_threshold = 0.15

    def reset(self):
        self.frame_buffer.clear()
        self.frame_counter = 0

    @torch.no_grad()
    def predict(self, frame_bgr: np.ndarray) -> dict:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(frame_rgb)
        self.frame_buffer.append(tensor)
        self.frame_counter += 1

        max_buffer = self.config.seq_len * self.config.frame_skip + 10
        if len(self.frame_buffer) > max_buffer:
            self.frame_buffer = self.frame_buffer[-max_buffer:]

        seq_len = self.config.seq_len
        frame_skip = self.config.frame_skip
        buf_len = len(self.frame_buffer)

        indices = []
        for i in range(seq_len):
            offset = (seq_len - 1 - i) * frame_skip
            idx = buf_len - 1 - offset
            idx = max(0, idx)
            indices.append(idx)

        frames = torch.stack([self.frame_buffer[i] for i in indices])
        frames = frames.unsqueeze(0).to(self.device)

        with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
            button_logits, joystick_pred = self.model(frames)

        button_probs = torch.sigmoid(button_logits[0]).cpu().numpy()
        buttons = button_probs > self.button_threshold
        joy = joystick_pred[0].cpu().numpy()

        result = {}
        for i, name in enumerate(BUTTON_NAMES):
            result[name] = bool(buttons[i])
        result["j_left"] = [float(joy[0]), float(joy[1])]
        result["j_right"] = [float(joy[2]), float(joy[3])]
        result["_button_probs"] = {
            name: float(button_probs[i]) for i, name in enumerate(BUTTON_NAMES)
        }

        return result

    def update_buffer_only(self, frame_bgr: np.ndarray):
        """Update frame buffer without running inference (for manual mode)."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(frame_rgb)
        self.frame_buffer.append(tensor)
        self.frame_counter += 1

        max_buffer = self.config.seq_len * self.config.frame_skip + 10
        if len(self.frame_buffer) > max_buffer:
            self.frame_buffer = self.frame_buffer[-max_buffer:]


# ═══════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Live AI/Manual inference: camera → model/controller → Pico UDP"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to full_inference_model.pt",
    )
    parser.add_argument(
        "--camera", type=int, default=None, help="Camera index (auto-detect if omitted)"
    )
    parser.add_argument(
        "--pico-ip", type=str, default=PICO_IP, help="Pico IP address"
    )
    parser.add_argument(
        "--pico-port", type=int, default=PICO_PORT, help="Pico UDP port"
    )
    parser.add_argument(
        "--player-id", type=int, default=PLAYER_ID, help="Player ID byte"
    )
    parser.add_argument(
        "--no-send",
        action="store_true",
        help="Preview only — don't send UDP packets",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device",
    )
    parser.add_argument(
        "--button-threshold",
        type=float,
        default=0.5,
        help="Sigmoid threshold for button activation (0-1)",
    )
    parser.add_argument(
        "--target-fps",
        type=int,
        default=30,
        help="Target camera/inference FPS",
    )
    parser.add_argument(
        "--start-mode",
        type=str,
        choices=["ai", "manual"],
        default="ai",
        help="Starting input mode (ai or manual)",
    )
    args = parser.parse_args()

    os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")
    os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

    print("=" * 58)
    print("  NitroGen Live Inference (AI + Manual Mode)")
    print("=" * 58)

    # ── Load model ──
    print("\n  Loading model...")
    engine = LiveInferenceEngine(args.checkpoint, device=args.device)
    engine.button_threshold = args.button_threshold

    # ── Initialize controller ──
    print("\n  Searching for Xbox controller...")
    controller = ControllerReader()
    controller_found = controller.find()
    if controller_found:
        controller.start()
    else:
        print("  No controller found - Manual mode will be unavailable")

    # ── Find camera ──
    if args.camera is not None:
        cam_idx = args.camera
    else:
        cameras = discover_cameras()
        if not cameras:
            print("  No cameras found!")
            sys.exit(1)
        if len(cameras) == 1:
            cam_idx = cameras[0]["index"]
            print(f"  Auto-selected camera [{cam_idx}]")
        else:
            while True:
                try:
                    cam_idx = int(input("  Select camera index: ").strip())
                    if any(c["index"] == cam_idx for c in cameras):
                        break
                except ValueError:
                    pass
                print(f"  Valid: {[c['index'] for c in cameras]}")

    cap = cv2.VideoCapture(cam_idx)
    cap.set(cv2.CAP_PROP_FPS, args.target_fps)
    if not cap.isOpened():
        print(f"  Failed to open camera [{cam_idx}]!")
        sys.exit(1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Camera: [{cam_idx}] {actual_w}x{actual_h}")

    # ── UDP socket ──
    sending = not args.no_send
    sock = None
    if sending:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"  Pico:   {args.pico_ip}:{args.pico_port}  (SENDING)")
    else:
        print(f"  Pico:   (disabled — preview only)")

    print(f"  Device: {args.device}")
    print(f"  Threshold: {args.button_threshold}")

    # ── Initialize mode ──
    current_mode = InputMode.AI if args.start_mode == "ai" else InputMode.MANUAL
    if current_mode == InputMode.MANUAL and not controller_found:
        print("  Warning: Manual mode requested but no controller - starting in AI mode")
        current_mode = InputMode.AI

    print(f"  Starting mode: {current_mode.value}")
    print(f"\n  [TAB] Toggle AI/Manual  |  [Q/ESC] Quit\n")

    # ── Main loop ──
    frame_interval = 1.0 / args.target_fps
    fps_counter = 0
    fps_timer = time.perf_counter()
    display_fps = 0.0
    frame_num = 0

    try:
        while True:
            t_start = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                print("  Camera read failed!")
                break

            # ── Get inputs based on mode ──
            if current_mode == InputMode.AI:
                # AI mode: run neural network
                action = engine.predict(frame)
                packet = prediction_to_pico_packet(action, args.player_id)
            else:
                # Manual mode: read from controller
                if controller.available:
                    ctrl_state = controller.snapshot()
                    action = ctrl_state.to_action_dict()
                    packet = ctrl_state.to_pico_packet(args.player_id)
                else:
                    # Controller disconnected - send neutral
                    action = {
                        "j_left": [0.0, 0.0],
                        "j_right": [0.0, 0.0],
                        **{name: False for name in BUTTON_NAMES}
                    }
                    packet = prediction_to_pico_packet(action, args.player_id)

                # Still update frame buffer even in manual mode
                # (so switching to AI mode has context)
                engine.update_buffer_only(frame)

            # ── Send to Pico ──
            if sending and sock is not None:
                try:
                    sock.sendto(packet, (args.pico_ip, args.pico_port))
                except OSError:
                    pass

            # ── Measure timing ──
            inference_time = time.perf_counter() - t_start
            latency_ms = inference_time * 1000.0

            fps_counter += 1
            elapsed_fps = time.perf_counter() - fps_timer
            if elapsed_fps >= 1.0:
                display_fps = fps_counter / elapsed_fps
                fps_counter = 0
                fps_timer = time.perf_counter()

            # ── Visualise ──
            preview_cam = cv2.resize(frame, (640, 480))
            preview_cam = draw_hud(
                preview_cam, display_fps, latency_ms, sending,
                current_mode, controller.available
            )
            input_panel = draw_input_panel(action, current_mode, panel_w=300, panel_h=480)
            combined = np.hstack([preview_cam, input_panel])
            cv2.imshow("NitroGen AI/Manual Control", combined)

            key = cv2.waitKey(1) & 0xFF

            # Handle key presses
            if key in (ord("q"), 27):  # Q or ESC
                break
            elif key == 9:  # TAB key
                # Toggle mode
                if current_mode == InputMode.AI:
                    if controller.available:
                        current_mode = InputMode.MANUAL
                        print("  Switched to MANUAL mode (Xbox Controller)")
                    else:
                        print("  Cannot switch to Manual - no controller connected")
                else:
                    current_mode = InputMode.AI
                    print("  Switched to AI mode (Neural Network)")

            frame_num += 1

            # Frame rate limiting
            elapsed = time.perf_counter() - t_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

    except KeyboardInterrupt:
        pass

    # ── Cleanup ──
    print("\n  Stopping...")

    # Send neutral packet before closing
    if sending and sock is not None:
        neutral = {"j_left": [0, 0], "j_right": [0, 0]}
        for name in BUTTON_NAMES:
            neutral[name] = False
        try:
            sock.sendto(
                prediction_to_pico_packet(neutral, args.player_id),
                (args.pico_ip, args.pico_port),
            )
        except OSError:
            pass
        sock.close()

    controller.stop()
    cap.release()
    cv2.destroyAllWindows()

    print(f"  Processed {frame_num} frames.  Done!")


if __name__ == "__main__":
    main()
