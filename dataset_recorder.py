"""
Xbox Controller → Pico UDP Sender + NitroGen Dataset Recorder
==============================================================
1. Reads local Xbox USB controller
2. Sends inputs to Raspberry Pi Pico over UDP (for gameplay)
3. Records camera video + controller inputs in NitroGen format (for AI training)

Press Ctrl+C or Q in the preview window to stop.
"""

import struct
import socket
import time
import json
import uuid
import sys
import os
import threading
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ═══════════════════════════════════════════════════════════════
# SETTINGS
# ═══════════════════════════════════════════════════════════════

# Pico connection
PICO_IP = "192.168.1.44"
PICO_PORT = 4210
PLAYER_ID = 1
SEND_RATE_HZ = 120

# Recording
TARGET_FPS = 30
CHUNK_DURATION_SEC = 20
FRAMES_PER_CHUNK = TARGET_FPS * CHUNK_DURATION_SEC  # 600
DATASET_ROOT = "./nitrogen_dataset"
GAME_TITLE = "Unknown"

# Controller tuning
DEADZONE = 4000
TRIGGER_THRESHOLD = 30

# ═══════════════════════════════════════════════════════════════
# BUTTON / HAT MAPPINGS
# ═══════════════════════════════════════════════════════════════

# Pico packet button bits (16-bit bitmask)
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

# NitroGen parquet column names (17 booleans)
NITROGEN_BUTTON_COLUMNS = [
    "dpad_down", "dpad_left", "dpad_right", "dpad_up",
    "left_shoulder", "left_thumb", "left_trigger",
    "right_shoulder", "right_thumb", "right_trigger",
    "south", "west", "east", "north",
    "back", "start", "guide",
]

NITROGEN_JOYSTICK_COLUMNS = ["j_left", "j_right"]

# inputs library event code → NitroGen boolean column name
EVENT_TO_NITROGEN = {
    "BTN_SOUTH":  "south",
    "BTN_EAST":   "east",
    "BTN_WEST":   "west",
    "BTN_NORTH":  "north",
    "BTN_TL":     "left_shoulder",
    "BTN_TR":     "right_shoulder",
    "BTN_THUMBL": "left_thumb",
    "BTN_THUMBR": "right_thumb",
    "BTN_SELECT": "back",
    "BTN_START":  "start",
    "BTN_MODE":   "guide",
}

HAT_TABLE = {
    ( 0, -1): 0, ( 1, -1): 1, ( 1,  0): 2, ( 1,  1): 3,
    ( 0,  1): 4, (-1,  1): 5, (-1,  0): 6, (-1, -1): 7,
    ( 0,  0): 8,
}

# ═══════════════════════════════════════════════════════════════
# CONTROLLER STATE
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

    # For NitroGen parquet (booleans tracked individually)
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
        """Build the 8-byte UDP packet for the Pico.
        
        Xbox→Switch axis mapping:
          - Xbox Y axes are inverted vs Switch (up=-32768 on Xbox, up=+32767 on Switch)
          - So we negate ly and ry before converting to u8
        """
        buttons = self.buttons
        if self.lt > TRIGGER_THRESHOLD:
            buttons |= BUTTON_BITS.get("BTN_TL2", 0)
        if self.rt > TRIGGER_THRESHOLD:
            buttons |= BUTTON_BITS.get("BTN_TR2", 0)

        hat = HAT_TABLE.get((self.hat_x, self.hat_y), 0x08)

        # Invert Y axes: Xbox up is negative, Switch up is positive
        lx = self._stick_to_u8(self.lx)
        ly = self._stick_to_u8(-self.ly)   # INVERTED
        rx = self._stick_to_u8(self.rx)
        ry = self._stick_to_u8(-self.ry)   # INVERTED

        return struct.pack('<BHBBBBB', player_id, buttons, hat, lx, ly, rx, ry)

    def to_nitrogen_row(self) -> dict:
        """Build a NitroGen-format action row for the parquet file.
        
        Stores the corrected (Switch-convention) axes:
          Y is inverted so that up = positive, down = negative.
        """
        dz = 0.08  # normalized deadzone

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
            # Invert Y so that up=positive in the recorded data too
            "j_left":  [stick_norm(self.lx), stick_norm(-self.ly)],
            "j_right": [stick_norm(self.rx), stick_norm(-self.ry)],
        }

    @staticmethod
    def _stick_to_u8(raw: int) -> int:
        raw = raw if abs(raw) >= DEADZONE else 0
        return max(0, min(255, int(((raw + 32768) / 65535) * 255)))


# ═══════════════════════════════════════════════════════════════
# CHUNK WRITER (NitroGen format)
# ═══════════════════════════════════════════════════════════════

def _find_working_video_writer(path: str, fps: int, size: Tuple[int, int]) -> cv2.VideoWriter:
    """Try multiple codecs/backends in order and return the first working VideoWriter."""
    w, h = size

    # List of (fourcc_string, extension, backend) combos to try
    attempts = [
        # MJPG is almost universally available and produces non-zero files
        ("MJPG", ".avi",  None),
        # Microsoft's MPEG-4 v2  – works on most Windows OpenCV builds
        ("MP42", ".avi",  None),
        # Raw / uncompressed fallback
        ("DIVX", ".avi",  None),
        # x264 if available
        ("X264", ".mp4",  None),
        ("avc1", ".mp4",  None),
        ("mp4v", ".mp4",  None),
    ]

    base = str(Path(path).with_suffix(""))  # strip original extension

    for codec, ext, backend in attempts:
        out_path = base + ext
        fourcc = cv2.VideoWriter_fourcc(*codec)
        if backend:
            writer = cv2.VideoWriter(out_path, backend, fourcc, fps, (w, h))
        else:
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if writer.isOpened():
            return writer
        writer.release()

    return None


class ChunkWriter:
    """Writes 20-second chunks of video + actions in NitroGen format."""

    def __init__(self, dataset_root: str, video_id: str,
                 resolution: Tuple[int, int], fps: int, game_title: str):
        self.dataset_root = Path(dataset_root)
        self.video_id = video_id
        self.resolution = resolution  # (H, W)
        self.fps = fps
        self.game_title = game_title
        self.chunk_index = 0
        self.global_frame = 0
        self._frames: List[np.ndarray] = []
        self._actions: List[dict] = []

    def add(self, frame: np.ndarray, action: dict):
        self._frames.append(frame)
        self._actions.append(action)
        if len(self._frames) >= FRAMES_PER_CHUNK:
            self.flush()

    def flush(self):
        if not self._frames:
            return

        n = len(self._frames)
        chunk_name = f"{self.video_id}_chunk_{self.chunk_index:04d}"
        shard = "SHARD_0000"

        actions_dir = self.dataset_root / "actions" / shard / self.video_id / chunk_name
        videos_dir = self.dataset_root / "videos" / shard / self.video_id
        actions_dir.mkdir(parents=True, exist_ok=True)
        videos_dir.mkdir(parents=True, exist_ok=True)

        # ── Parquet ──
        data = {}
        for col in NITROGEN_BUTTON_COLUMNS:
            data[col] = pa.array([r[col] for r in self._actions], type=pa.bool_())
        for col in NITROGEN_JOYSTICK_COLUMNS:
            data[col] = pa.array(
                [r[col] for r in self._actions],
                type=pa.list_(pa.float32(), 2),
            )
        pq.write_table(
            pa.table(data),
            str(actions_dir / "actions_raw.parquet"),
            compression="snappy",
        )

        # ── Metadata ──
        meta = {
            "uuid": f"{chunk_name}_actions",
            "game": self.game_title,
            "original_video": {
                "resolution": list(self.resolution),
                "start_frame": self.global_frame,
                "end_frame": self.global_frame + n,
                "fps": self.fps,
            },
            "bbox_game_area": {"xtl": 0.0, "ytl": 0.0, "xbr": 1.0, "ybr": 1.0},
        }
        with open(actions_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        # ── Video ──
        h, w = self._frames[0].shape[:2]
        video_path = videos_dir / f"{chunk_name}.mp4"

        writer = _find_working_video_writer(str(video_path), self.fps, (w, h))

        if writer and writer.isOpened():
            for frame in self._frames:
                writer.write(frame)
            writer.release()
            # The actual file might have .avi extension if mp4 codecs failed
            # Find whichever file was actually created
            stem = video_path.with_suffix("")
            actual = None
            for ext in [".mp4", ".avi"]:
                candidate = stem.with_suffix(ext)
                if candidate.exists() and candidate.stat().st_size > 0:
                    actual = candidate
                    break
            if actual:
                size_mb = actual.stat().st_size / (1024 * 1024)
                print(f"  💾 Chunk {self.chunk_index:04d}: {n} frames, {size_mb:.1f}MB → {actual.name}")
            else:
                print(f"  ⚠ Chunk {self.chunk_index:04d}: file written but 0 bytes (codec issue)")
        else:
            if writer:
                writer.release()
            print(f"  ⚠ Chunk {self.chunk_index:04d}: no working video codec found!")

        self.global_frame += n
        self.chunk_index += 1
        self._frames.clear()
        self._actions.clear()


# ═══════════════════════════════════════════════════════════════
# CAMERA DISCOVERY
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

    # Clean summary
    if cameras:
        print("\n  AVAILABLE CAMERAS:")
        for c in cameras:
            print(f"    [{c['index']}]  {c['width']}x{c['height']} @ {c['fps']:.0f}fps")
        print()
    else:
        print("\n  AVAILABLE CAMERAS: (none found)\n")

    return cameras


# ═══════════════════════════════════════════════════════════════
# CONTROLLER READER THREAD
# ═══════════════════════════════════════════════════════════════

class ControllerReader:
    """Background thread that reads Xbox controller events."""

    def __init__(self):
        self.state = ControllerState()
        self.lock = threading.Lock()
        self.running = False
        self.gamepad = None

    def find(self) -> bool:
        try:
            import inputs
            pads = inputs.devices.gamepads
            if not pads:
                return False
            self.gamepad = pads[0]
            print(f"  Found: {self.gamepad.name}")
            return True
        except ImportError:
            print("  pip install inputs")
            return False

    def start(self):
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

            # Also track booleans for NitroGen
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
# UDP SENDER THREAD
# ═══════════════════════════════════════════════════════════════

class PicoSender:
    """Background thread that sends controller state to the Pico at a fixed rate."""

    def __init__(self, controller: ControllerReader,
                 ip: str, port: int, player_id: int, rate_hz: int):
        self.controller = controller
        self.ip = ip
        self.port = port
        self.player_id = player_id
        self.interval = 1.0 / rate_hz
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.running = False

    def start(self):
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self.running = False
        self.sock.close()

    def _loop(self):
        while self.running:
            t = time.perf_counter()
            state = self.controller.snapshot()
            packet = state.to_pico_packet(self.player_id)
            try:
                self.sock.sendto(packet, (self.ip, self.port))
            except OSError:
                pass
            sleep = self.interval - (time.perf_counter() - t)
            if sleep > 0:
                time.sleep(sleep)


# ═══════════════════════════════════════════════════════════════
# INPUT VISUALISER  (drawn beside the camera preview)
# ═══════════════════════════════════════════════════════════════

def draw_input_panel(action: dict, panel_w: int = 300, panel_h: int = 480) -> np.ndarray:
    """Render a controller-input visualisation panel."""
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)

    white  = (255, 255, 255)
    green  = (0, 255, 0)
    red    = (0, 0, 255)
    cyan   = (255, 200, 0)
    gray   = (100, 100, 100)
    yellow = (0, 255, 255)
    font   = cv2.FONT_HERSHEY_SIMPLEX

    # ── Title ──
    cv2.putText(panel, "CONTROLLER INPUT", (10, 24), font, 0.55, cyan, 1)
    cv2.line(panel, (10, 32), (panel_w - 10, 32), gray, 1)

    # ── Sticks (visual circles) ──
    jl = action.get("j_left", [0.0, 0.0])
    jr = action.get("j_right", [0.0, 0.0])

    for label, jx, jy, cx in [("L-STICK", jl[0], jl[1], 80),
                                ("R-STICK", jr[0], jr[1], 220)]:
        cy_base = 100
        radius = 45
        # Outer ring
        cv2.circle(panel, (cx, cy_base), radius, gray, 1)
        cv2.line(panel, (cx - radius, cy_base), (cx + radius, cy_base), (50, 50, 50), 1)
        cv2.line(panel, (cx, cy_base - radius), (cx, cy_base + radius), (50, 50, 50), 1)
        # Dot (note: jy is already corrected, positive=up, so negate for screen coords)
        dx = int(jx * radius)
        dy = int(jy * radius)  # screen Y is inverted
        dot_color = green if (abs(jx) > 0.01 or abs(jy) > 0.01) else gray
        cv2.circle(panel, (cx + dx, cy_base + dy), 6, dot_color, -1)
        # Label
        cv2.putText(panel, label, (cx - 30, cy_base + radius + 18), font, 0.35, white, 1)
        # Values
        cv2.putText(panel, f"X:{jx:+.2f}", (cx - 30, cy_base + radius + 34), font, 0.3, white, 1)
        cv2.putText(panel, f"Y:{jy:+.2f}", (cx - 30, cy_base + radius + 48), font, 0.3, white, 1)

    # ── Triggers ──
    y_trig = 195
    lt_on = action.get("left_trigger", False)
    rt_on = action.get("right_trigger", False)
    cv2.putText(panel, "LT", (20, y_trig), font, 0.4, green if lt_on else gray, 1)
    cv2.rectangle(panel, (50, y_trig - 12), (130, y_trig), green if lt_on else gray,
                  -1 if lt_on else 1)
    cv2.putText(panel, "RT", (160, y_trig), font, 0.4, green if rt_on else gray, 1)
    cv2.rectangle(panel, (190, y_trig - 12), (270, y_trig), green if rt_on else gray,
                  -1 if rt_on else 1)

    # ── D-Pad ──
    y_dpad = 240
    cv2.putText(panel, "D-PAD", (10, y_dpad), font, 0.4, white, 1)
    dpad_cx, dpad_cy = 55, y_dpad + 35
    ds = 16  # half-size of each d-pad square
    directions = [
        ("U", 0, -1, action.get("dpad_up", False)),
        ("D", 0,  1, action.get("dpad_down", False)),
        ("L", -1, 0, action.get("dpad_left", False)),
        ("R",  1, 0, action.get("dpad_right", False)),
    ]
    for lbl, ox, oy, on in directions:
        px, py = dpad_cx + ox * (ds * 2), dpad_cy + oy * (ds * 2)
        color = green if on else gray
        cv2.rectangle(panel, (px - ds, py - ds), (px + ds, py + ds), color, -1 if on else 1)
        cv2.putText(panel, lbl, (px - 5, py + 5), font, 0.35, (0,0,0) if on else white, 1)

    # ── Face buttons ──
    face_cx, face_cy = 220, y_dpad + 35
    bs = 14
    face_buttons = [
        ("N", 0, -1, action.get("north", False)),
        ("S", 0,  1, action.get("south", False)),
        ("W", -1, 0, action.get("west", False)),
        ("E",  1, 0, action.get("east", False)),
    ]
    for lbl, ox, oy, on in face_buttons:
        px, py = face_cx + ox * (bs * 2 + 4), face_cy + oy * (bs * 2 + 4)
        color = red if on else gray
        cv2.circle(panel, (px, py), bs, color, -1 if on else 1)
        cv2.putText(panel, lbl, (px - 5, py + 5), font, 0.35, (0,0,0) if on else white, 1)

    cv2.putText(panel, "FACE", (195, y_dpad), font, 0.4, white, 1)

    # ── Shoulder / other buttons ──
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
    for i, (lbl, on) in enumerate(btn_list):
        color = yellow if on else gray
        cv2.rectangle(panel, (bx, y_btn + 10), (bx + 32, y_btn + 30), color, -1 if on else 1)
        cv2.putText(panel, lbl, (bx + 3, y_btn + 25), font, 0.33,
                    (0, 0, 0) if on else white, 1)
        bx += 40

    # ── Active inputs text list ──
    y_active = 395
    cv2.line(panel, (10, y_active - 8), (panel_w - 10, y_active - 8), gray, 1)
    cv2.putText(panel, "ACTIVE:", (10, y_active + 8), font, 0.4, cyan, 1)
    active = [n for n in NITROGEN_BUTTON_COLUMNS if action.get(n)]
    if active:
        # Wrap into lines
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

    return panel


# ═══════════════════════════════════════════════════════════════
# PREVIEW HUD  (overlaid on the camera frame)
# ═══════════════════════════════════════════════════════════════

def draw_hud(frame: np.ndarray, action: dict, frame_num: int, chunk: int) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 30), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    cv2.putText(frame, f"REC  Chunk:{chunk:04d}  Frame:{frame_num}",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    return frame


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    global GAME_TITLE, DATASET_ROOT

    # Suppress OpenH264 / FFmpeg warnings that clutter the console
    os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
    os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"  # suppress ffmpeg logs

    print("=" * 58)
    print("  Xbox → Pico + NitroGen Dataset Recorder")
    print("=" * 58)

    # ── Controller ──
    print("\n  Searching for Xbox controller...")
    ctrl = ControllerReader()
    if not ctrl.find():
        print("  No controller found! Connect one and retry.")
        sys.exit(1)

    # ── Camera ──
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
                cam_idx = int(input(f"  Select camera index: ").strip())
                if any(c["index"] == cam_idx for c in cameras):
                    break
            except ValueError:
                pass
            print(f"  Valid: {[c['index'] for c in cameras]}")

    # ── Config ──
    GAME_TITLE = input("  Game title [Unknown]: ").strip() or "Unknown"
    DATASET_ROOT = input("  Output dir [./nitrogen_dataset]: ").strip() or "./nitrogen_dataset"

    video_id = f"rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    # ── Open camera ──
    cap = cv2.VideoCapture(cam_idx)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    if not cap.isOpened():
        print("  Failed to open camera!")
        sys.exit(1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ── Init systems ──
    writer = ChunkWriter(DATASET_ROOT, video_id,
                         (actual_h, actual_w), TARGET_FPS, GAME_TITLE)

    ctrl.start()

    sender = PicoSender(ctrl, PICO_IP, PICO_PORT, PLAYER_ID, SEND_RATE_HZ)
    sender.start()

    print(f"\n  Video ID: {video_id}")
    print(f"  Camera: [{cam_idx}] {actual_w}x{actual_h}")
    print(f"  Pico:   {PICO_IP}:{PICO_PORT} @ {SEND_RATE_HZ}Hz")
    print(f"  Record: {TARGET_FPS}fps, {CHUNK_DURATION_SEC}s chunks")
    print(f"\n  Press Q or ESC in preview window to stop.\n")

    frame_count = 0
    frame_interval = 1.0 / TARGET_FPS

    try:
        while True:
            t_start = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                print("  Camera read failed!")
                break

            # Get controller state and build NitroGen row
            state = ctrl.snapshot()
            action_row = state.to_nitrogen_row()

            # Record to dataset
            writer.add(frame, action_row)
            frame_count += 1

            # ── Build combined preview: camera + input panel ──
            preview_cam = cv2.resize(frame, (640, 480))
            preview_cam = draw_hud(preview_cam, action_row, frame_count, writer.chunk_index)

            input_panel = draw_input_panel(action_row, panel_w=300, panel_h=480)

            combined = np.hstack([preview_cam, input_panel])
            cv2.imshow("NitroGen Recorder", combined)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break

            # Frame rate limiting
            elapsed = time.perf_counter() - t_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

    except KeyboardInterrupt:
        pass

    # ── Cleanup ──
    print("\n  Stopping...")
    writer.flush()
    sender.stop()
    ctrl.stop()
    cap.release()
    cv2.destroyAllWindows()

    print(f"\n  {'=' * 50}")
    print(f"  Done! {frame_count} frames, {writer.chunk_index} chunks")
    print(f"  Saved to: {Path(DATASET_ROOT).resolve()}")
    print(f"  {'=' * 50}")


if __name__ == "__main__":
    main()
