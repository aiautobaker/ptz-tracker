import cv2
import json
import logging
import requests
import sys
import tkinter as tk
from tkinter import ttk
from pathlib import Path
from PIL import Image, ImageTk
from ultralytics import YOLO

# Resolve paths relative to this script so the app works from any directory
_SCRIPT_DIR    = Path(__file__).parent.resolve()
_MODEL_PATH    = _SCRIPT_DIR / "yolov8n.pt"
_LOG_PATH      = _SCRIPT_DIR / "tracker.log"
_SETTINGS_PATH = _SCRIPT_DIR / "settings.json"

# ───────────────────────────────────────── #
# LOGGING                                   #
# ───────────────────────────────────────── #
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(_LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("ptz")


# ───────────────────────────────────────── #
# UTILS                                     #
# ───────────────────────────────────────── #
def compute_speed(offset, deadzone, max_speed):
    """Proportionally map |offset| → speed in [1, max_speed]."""
    beyond = max(0.0, abs(offset) - deadzone)
    frac   = beyond / max(1e-6, 1.0 - deadzone)
    return max(1, min(max_speed, round(1 + frac * (max_speed - 1))))

def get_direction(ox, oy, dz_x, dz_y):
    in_x = abs(ox) < dz_x
    in_y = abs(oy) < dz_y
    if in_x and in_y:
        return "stop"
    h = "" if in_x else ("right" if ox > 0 else "left")
    v = "" if in_y else ("down"  if oy > 0 else "up")
    return (v + h) if (h and v) else (h or v)

def get_track_boxes(results):
    """Return list of (x1, y1, x2, y2, track_id) for all tracked persons."""
    boxes = []
    for r in results:
        if r.boxes.id is None:
            continue
        ids = r.boxes.id.int().tolist()
        for box, tid in zip(r.boxes, ids):
            if int(box.cls[0]) != 0:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append((x1, y1, x2, y2, tid))
    return boxes

def find_tracked_person(track_boxes, ref_id):
    """Find box by track ID. Falls back to largest person if no ref_id set."""
    if not track_boxes:
        return None
    if ref_id is None:
        return max(track_boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))[:4]
    for b in track_boxes:
        if b[4] == ref_id:
            return b[:4]
    return None


# ───────────────────────────────────────── #
# CAMERA CONTROLLER CLASS                   #
# ───────────────────────────────────────── #
class CameraController:
    """Manages the RTSP stream, YOLO inference, PTZ, and UI for a single camera."""
    
    def __init__(self, camera_ip, camera_name, parent_frame, config_key, root):
        self.ip = camera_ip
        self.name = camera_name
        self.rtsp_url = f"rtsp://{self.ip}:554/2"
        self.http_base = f"http://{self.ip}/cgi-bin/ptzctrl.cgi"
        self.config_key = config_key
        self.root = root
        
        # Load YOLO model (unique instance per camera)
        log.info(f"[{self.name}] Loading YOLO model...")
        self.model = YOLO(str(_MODEL_PATH))
        
        # Open RTSP stream
        log.info(f"[{self.name}] Connecting to {self.rtsp_url}")
        self.cap = cv2.VideoCapture(self.rtsp_url)
        if not self.cap.isOpened():
            log.error(f"[{self.name}] Could not open RTSP stream.")
            self.fw, self.fh = 1280, 720 # Fallback sizes
        else:
            self.fw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.fh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
        log.info(f"[{self.name}] Stream: {self.fw}x{self.fh}")
        
        # Internal State
        self.frame_num = 0
        self.reference_track_id = None
        self.smooth_box = None
        self.last_boxes = []
        self.capture_mode = False
        self.lost_subject_frames = 0
        self.manual_override = False
        self.last_dir = "stop"
        self.last_pan_speed = None
        self.last_tilt_speed = None
        self.zooming_out = False
        self.frame_hold = None
        self.photo_hold = None
        self.thumb_photo = None
        
        # UI interaction state
        self.drag_handle = None # 'tl', 'tr', 'bl', 'br', None
        
        # Settings
        self.settings = {
            "deadzone_x": 0.08,
            "deadzone_y": 0.30,
            "pan_speed":  3,
            "tilt_speed": 1,
            "variable_speed": True,
            "smooth_alpha": 0.5,
            "process_every_n": 3,
            "tracking_on": True,
            "target_x": 0.0,
            "target_y": -0.4,
            "auto_zoom_out": False,
            "zoom_out_speed": 3,
            "zoom_out_delay": 15
        }
        self.load_settings()
        
        self.build_ui(parent_frame)
        
    def load_settings(self):
        """Load settings specific to this camera."""
        if not _SETTINGS_PATH.exists(): return
        try:
            saved = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
            if self.config_key in saved:
                for k in self.settings:
                    if k in saved[self.config_key]:
                        self.settings[k] = saved[self.config_key][k]
        except Exception:
            pass

    def save_settings(self):
        """Save settings specifically under this camera's key."""
        saved = {}
        if _SETTINGS_PATH.exists():
            try:
                saved = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
            except Exception:
                pass
        saved[self.config_key] = self.settings
        try:
            _SETTINGS_PATH.write_text(json.dumps(saved, indent=2), encoding="utf-8")
        except Exception as e:
            log.error(f"[{self.name}] Failed to save settings: {e}")

    # --- Hardware Control ---
    def send_ptz(self, cmd):
        try:
            requests.get(f"{self.http_base}?ptzcmd&{cmd}", timeout=0.5)
        except Exception as e:
            log.warning(f"[{self.name}] PTZ command failed ({cmd}): {e}")

    def zoom(self, direction):
        speed = self.settings["zoom_out_speed"]
        cmds = {
            "in":   f"zoomin&{speed}&0",
            "out":  f"zoomout&{speed}&0",
            "stop": "zoomstop&0&0",
        }
        if direction in cmds:
            self.send_ptz(cmds[direction])

    def move(self, direction, pan=None, tilt=None):
        pan  = pan  if pan  is not None else self.settings["pan_speed"]
        tilt = tilt if tilt is not None else self.settings["tilt_speed"]
        cmds = {
            "left":      f"left&{pan}&{tilt}",
            "right":     f"right&{pan}&{tilt}",
            "up":        f"up&{pan}&{tilt}",
            "down":      f"down&{pan}&{tilt}",
            "upleft":    f"leftup&{pan}&{tilt}",
            "upright":   f"rightup&{pan}&{tilt}",
            "downleft":  f"leftdown&{pan}&{tilt}",
            "downright": f"rightdown&{pan}&{tilt}",
            "stop":      "ptzstop&0&0",
        }
        if direction in cmds:
            self.send_ptz(cmds[direction])

    # --- UI Building ---
    def build_ui(self, parent_frame):
        # ── Title Bar ──
        header = tk.Frame(parent_frame, bg="#2a2a2a", height=32)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(header, text=self.name, bg="#2a2a2a", fg="#00e5ff", 
                 font=("Helvetica", 12, "bold")).pack(side=tk.LEFT, padx=10)
        
        self.status_var = tk.StringVar(value="Waiting for stream…")
        tk.Label(header, textvariable=self.status_var, bg="#2a2a2a", fg="#aaa", 
                 font=("Helvetica", 10)).pack(side=tk.RIGHT, padx=10)

        # ── Main Area ──
        body = tk.Frame(parent_frame, bg="black")
        body.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(body, bg="black", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ── Settings panel (right side) ───────
        ctrl = tk.Frame(body, bg="#1e1e1e", width=360)
        ctrl.pack(side=tk.LEFT, fill=tk.Y)
        ctrl.pack_propagate(False)

        notebook = ttk.Notebook(ctrl)
        notebook.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        tab_tracking = tk.Frame(notebook, bg="#1e1e1e")
        tab_target   = tk.Frame(notebook, bg="#1e1e1e")
        tab_controls = tk.Frame(notebook, bg="#1e1e1e")

        notebook.add(tab_tracking, text="Tracking")
        notebook.add(tab_target,   text="Target")
        notebook.add(tab_controls, text="Controls")

        # Helpers
        def make_scrollable(parent):
            c  = tk.Canvas(parent, bg="#1e1e1e", highlightthickness=0)
            sb = ttk.Scrollbar(parent, orient="vertical", command=c.yview)
            inner = tk.Frame(c, bg="#1e1e1e", padx=12)
            win   = c.create_window((0, 0), window=inner, anchor="nw")
            inner.bind("<Configure>", lambda e: c.configure(scrollregion=c.bbox("all")))
            c.bind("<Configure>",     lambda e: c.itemconfig(win, width=e.width))
            c.configure(yscrollcommand=sb.set)
            c.bind("<MouseWheel>", lambda e: c.yview_scroll(int(-1 * (e.delta / 120)), "units"))
            sb.pack(side=tk.RIGHT, fill=tk.Y)
            c.pack(side=tk.LEFT,  fill=tk.BOTH, expand=True)
            return inner

        def section(parent, t):
            tk.Label(parent, text=t, bg="#1e1e1e", fg="#888", font=("Helvetica", 9)).pack(pady=(10, 0))

        _ent_style = dict(bg="#2a2a2a", fg="#ffffff", insertbackground="#fff",
                          relief=tk.FLAT, font=("Helvetica", 11), width=6)

        def make_entry(parent, label, key, lo, hi, is_int=False):
            row = tk.Frame(parent, bg="#1e1e1e")
            row.pack(fill=tk.X, pady=(4, 0), padx=8)
            tk.Label(row, text=label, bg="#1e1e1e", fg="#cccccc", font=("Helvetica", 10), anchor="w").pack(side=tk.LEFT)
            fmt = (lambda v: str(int(v))) if is_int else (lambda v: f"{v:.3f}")
            var = tk.StringVar(value=fmt(self.settings[key]))
            ent = tk.Entry(row, textvariable=var, **_ent_style)
            ent.pack(side=tk.RIGHT)
            def apply(e=None):
                try:
                    v = int(float(var.get())) if is_int else round(float(var.get()), 3)
                    v = max(lo, min(hi, v))
                    self.settings[key] = v
                    var.set(fmt(v))
                except ValueError:
                    var.set(fmt(self.settings[key]))
            ent.bind("<Return>", apply)
            ent.bind("<FocusOut>", apply)
            return var

        def make_entry_pair(parent, label1, key1, label2, key2, lo, hi, is_int=False):
            row = tk.Frame(parent, bg="#1e1e1e")
            row.pack(fill=tk.X, pady=(4, 0), padx=6)
            fmt = (lambda v: str(int(v))) if is_int else (lambda v: f"{v:.3f}")
            vars_out = []
            for label, key in [(label1, key1), (label2, key2)]:
                tk.Label(row, text=label+":", bg="#1e1e1e", fg="#aaa", font=("Helvetica", 9)).pack(side=tk.LEFT, padx=(0, 2))
                var = tk.StringVar(value=fmt(self.settings[key]))
                ent = tk.Entry(row, textvariable=var, **{**_ent_style, "width": 5, "font": ("Helvetica", 10)})
                ent.pack(side=tk.LEFT, padx=(0, 8))
                def apply(e=None, k=key, v=var):
                    try:
                        val = int(float(v.get())) if is_int else round(float(v.get()), 3)
                        val = max(lo, min(hi, val))
                        self.settings[k] = val
                        v.set(fmt(val))
                    except ValueError:
                        v.set(fmt(self.settings[k]))
                ent.bind("<Return>", apply)
                ent.bind("<FocusOut>", apply)
                vars_out.append(var)
            return vars_out

        # ── Tab 1: Tracking ───────────────────
        trk = make_scrollable(tab_tracking)

        section(trk, "── DEADZONE ──")
        self.dz_x_var, self.dz_y_var = make_entry_pair(trk, "Horiz", "deadzone_x", "Vert", "deadzone_y", 0.02, 0.40)

        section(trk, "── SPEED ──")
        vs_var = tk.BooleanVar(value=self.settings["variable_speed"])
        tk.Checkbutton(trk, text="Variable Speed (auto)", variable=vs_var,
                       command=lambda: self.settings.update(variable_speed=vs_var.get()),
                       bg="#1e1e1e", fg="#fff", selectcolor="#333").pack(pady=(6, 0))
        make_entry_pair(trk, "Pan", "pan_speed", "Tilt", "tilt_speed", 1, 12, is_int=True)

        section(trk, "── AUTO ZOOM-OUT ──")
        az_var = tk.BooleanVar(value=self.settings["auto_zoom_out"])
        tk.Checkbutton(trk, text="Auto Zoom-out Enabled", variable=az_var,
                       command=lambda: self.settings.update(auto_zoom_out=az_var.get()),
                       bg="#1e1e1e", fg="#fff", selectcolor="#333").pack(pady=(6, 0))
        make_entry(trk, "Zoom Speed", "zoom_out_speed", 1, 12, is_int=True)
        make_entry(trk, "Delay Frames", "zoom_out_delay", 1, 60, is_int=True)

        section(trk, "── SYSTEM ──")
        make_entry(trk, "Process Every N", "process_every_n", 1, 10, is_int=True)
        make_entry(trk, "Smoothing Alpha", "smooth_alpha", 0.1, 1.0)
        
        save_btn = tk.Button(trk, text="Save Settings", bg="#1a3c5c", fg="#000",
                             font=("Helvetica", 10), command=self.save_settings, padx=8)
        save_btn.pack(pady=20)


        # ── Tab 2: Target ─────────────────────
        tgt = make_scrollable(tab_target)

        section(tgt, "── TARGET POSITION ──")
        tk.Label(tgt, text="Drag handles on the video to move/resize the zone.",
                 bg="#1e1e1e", fg="#aaa", font=("Helvetica", 9)).pack(pady=4)

        self.tx_var, self.ty_var = make_entry_pair(tgt, "X", "target_x", "Y", "target_y", -0.9, 0.9)

        def reset_target():
            self.settings["target_x"] = 0.0
            self.settings["target_y"] = -0.4
            self.tx_var.set("0.000")
            self.ty_var.set("-0.400")

        tk.Button(tgt, text="Reset Zone", command=reset_target).pack(pady=8)

        section(tgt, "── TARGET PERSON ──")
        self.thumb_label = tk.Label(tgt, bg="#2a2a2a", relief=tk.FLAT)
        self.thumb_label.pack(pady=6)
        
        self.tgt_status = tk.StringVar(value="Tracking largest person")
        tk.Label(tgt, textvariable=self.tgt_status, bg="#1e1e1e", fg="#ccc").pack()

        btn_row = tk.Frame(tgt, bg="#1e1e1e")
        btn_row.pack(pady=6)

        def do_capture():
            self.capture_mode = True
            self.tgt_status.set("Click a person in the video…")

        def do_clear():
            self.reference_track_id = None
            self.smooth_box = None
            self.thumb_label.config(image="", width=0, height=0)
            self.capture_mode = False
            self.tgt_status.set("Tracking largest person")

        tk.Button(btn_row, text="Lock Target", command=do_capture, bg="#1a5c1a", fg="#000").pack(side=tk.LEFT, padx=4)
        tk.Button(btn_row, text="Clear", command=do_clear, bg="#5c1a1a", fg="#000").pack(side=tk.LEFT, padx=4)


        # ── Tab 3: Controls ───────────────────
        tk.Label(tab_controls, text="MANUAL CONTROLS", bg="#1e1e1e", fg="#fff").pack(pady=10)

        _db = dict(font=("Helvetica", 14), relief=tk.FLAT, activebackground="#555", cursor="hand2", width=3, fg="#000")

        def _manual_press(cmd):
            self.manual_override = True
            self.move(cmd)
        def _manual_release():
            self.move("stop")
            self.manual_override = False

        dpad = tk.Frame(tab_controls, bg="#1e1e1e")
        dpad.pack()
        cmds = [("↖", "upleft"), ("↑", "up"), ("↗", "upright"),
                ("←", "left"),   ("■", "stop"),("→", "right"),
                ("↙", "downleft"),("↓", "down"),("↘", "downright")]
        
        for i, (txt, cmd) in enumerate(cmds):
            b = tk.Button(dpad, text=txt, bg="#333" if cmd!="stop" else "#4a1a1a", **_db)
            b.grid(row=i//3, column=i%3, padx=2, pady=2)
            b.bind("<ButtonPress-1>", lambda e, c=cmd: _manual_press(c))
            b.bind("<ButtonRelease-1>", lambda e: _manual_release() if cmd!="stop" else None)

        zoom_row = tk.Frame(tab_controls, bg="#1e1e1e")
        zoom_row.pack(pady=20)
        _zb = dict(font=("Helvetica", 10), relief=tk.FLAT, activebackground="#555", cursor="hand2", padx=10, pady=6)
        zin_b  = tk.Button(zoom_row, text="＋ In",  bg="#1a3c1a", fg="#000", **_zb)
        zout_b = tk.Button(zoom_row, text="－ Out", bg="#3c1a1a", fg="#000", **_zb)
        zin_b.pack(side=tk.LEFT, padx=4)
        zout_b.pack(side=tk.LEFT, padx=4)
        zin_b.bind("<ButtonPress-1>", lambda e: self.zoom("in"))
        zin_b.bind("<ButtonRelease-1>", lambda e: self.zoom("stop"))
        zout_b.bind("<ButtonPress-1>", lambda e: self.zoom("out"))
        zout_b.bind("<ButtonRelease-1>", lambda e: self.zoom("stop"))


        # ── Canvas Mouse Events ───────────────────
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

    def _sync_zone_vars(self):
        self.tx_var.set(f"{self.settings['target_x']:.3f}")
        self.ty_var.set(f"{self.settings['target_y']:.3f}")
        self.dz_x_var.set(f"{self.settings['deadzone_x']:.3f}")
        self.dz_y_var.set(f"{self.settings['deadzone_y']:.3f}")

    def on_mouse_down(self, event):
        cw = self.canvas.winfo_width() or self.fw
        ch = self.canvas.winfo_height() or self.fh
        mx, my = event.x * self.fw / cw, event.y * self.fh / ch

        # Tracking Indicator toggle (top-left)
        if mx < 200 and my < 45:
            self.settings["tracking_on"] = not self.settings["tracking_on"]
            return

        # Target Capture Mode
        if self.capture_mode:
            for (x1, y1, x2, y2, tid) in self.last_boxes:
                if x1 <= mx <= x2 and y1 <= my <= y2:
                    self.reference_track_id = tid
                    self.smooth_box = None
                    if self.frame_hold is not None:
                        crop = self.frame_hold[int(y1):int(y2), int(x1):int(x2)]
                        if crop.size > 0:
                            img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(cv2.resize(crop, (60, 90)), cv2.COLOR_BGR2RGB)))
                            self.thumb_photo = img
                            self.thumb_label.config(image=img, width=60, height=90)
                        self.tgt_status.set(f"Target locked ✓ (ID {tid})")
                    self.capture_mode = False
                    return
            self.capture_mode = False
            self.tgt_status.set(f"Target locked ✓ (ID {self.reference_track_id})" if self.reference_track_id else "Tracking largest person")
            return

        # Handle Interaction with Deadzone Box
        tx = self.fw / 2 + self.settings["target_x"] * (self.fw / 2)
        ty = self.fh / 2 + self.settings["target_y"] * (self.fh / 2)
        dzx = self.fw / 2 * self.settings["deadzone_x"]
        dzy = self.fh / 2 * self.settings["deadzone_y"]
        
        # Check if clicked on corners (20px tolerance)
        tol = 30
        corners = {
            'tl': (tx - dzx, ty - dzy),
            'tr': (tx + dzx, ty - dzy),
            'bl': (tx - dzx, ty + dzy),
            'br': (tx + dzx, ty + dzy),
            'center': (tx, ty) # For moving the whole box
        }
        
        for name, (cx, cy) in corners.items():
            if abs(mx - cx) < tol and abs(my - cy) < tol:
                self.drag_handle = name
                return

        # If clicked inside the box but not on a corner, we move the box via center
        if abs(mx - tx) < dzx and abs(my - ty) < dzy:
            self.drag_handle = 'center'
            return

        # Clicked completely outside handles, just jump center
        self.drag_handle = 'center'
        self.on_mouse_drag(event)


    def on_mouse_drag(self, event):
        if self.drag_handle is None or self.capture_mode: return

        cw = self.canvas.winfo_width() or self.fw
        ch = self.canvas.winfo_height() or self.fh
        mx = max(0, min(self.fw, event.x * self.fw / cw))
        my = max(0, min(self.fh, event.y * self.fh / ch))

        nx = (mx - self.fw / 2) / (self.fw / 2)
        ny = (my - self.fh / 2) / (self.fh / 2)

        if self.drag_handle == 'center':
            self.settings["target_x"] = round(nx, 3)
            self.settings["target_y"] = round(ny, 3)
        else:
            # Resizing logic (expand/contract dz relative to opposite anchor)
            tx, ty = self.settings["target_x"], self.settings["target_y"]
            ndzx = abs(nx - tx)
            ndzy = abs(ny - ty)
            self.settings["deadzone_x"] = round(max(0.01, min(0.9, ndzx)), 3)
            self.settings["deadzone_y"] = round(max(0.01, min(0.9, ndzy)), 3)
            
        self._sync_zone_vars()

    def on_mouse_up(self, event):
        self.drag_handle = None

    # --- Main Loop Integration ---
    def update(self):
        if not self.cap.isOpened():
            self.status_var.set("Camera disconnected.")
            return

        # Read specific frames
        ret, frame = self.cap.read()
        if not ret:
            # Retry next tick
            return

        self.frame_hold = frame.copy()
        
        # --- Downscale before inference for better performance ---
        target_inference_w = 640
        scale = target_inference_w / self.fw
        inf_w, inf_h = int(self.fw * scale), int(self.fh * scale)
        small_frame = cv2.resize(frame, (inf_w, inf_h))
        
        tx = self.fw / 2 + self.settings["target_x"] * (self.fw / 2)
        ty = self.fh / 2 + self.settings["target_y"] * (self.fh / 2)

        self.frame_num += 1
        if self.manual_override:
            self.status_var.set("Manual control")
        elif self.frame_num % self.settings["process_every_n"] == 0:
            try:
                results = self.model.track(small_frame, verbose=False, classes=[0], persist=True)
                boxes = get_track_boxes(results)
                
                # Scale boxes back to native resolution!
                native_boxes = []
                for b in boxes:
                    nx1, ny1, nx2, ny2 = b[0]/scale, b[1]/scale, b[2]/scale, b[3]/scale
                    native_boxes.append((nx1, ny1, nx2, ny2, b[4]))
                
                self.last_boxes = native_boxes
                raw_box = find_tracked_person(native_boxes, self.reference_track_id)
            except Exception as e:
                log.error(f"[{self.name}] YOLO error: {e}")
                raw_box = None

            if raw_box:
                if self.smooth_box is None: self.smooth_box = list(raw_box)
                else:
                    a = self.settings["smooth_alpha"]
                    self.smooth_box = [a * n + (1 - a) * s for n, s in zip(raw_box, self.smooth_box)]
                person_box = tuple(self.smooth_box)
            else:
                self.smooth_box = person_box = None

            if person_box and self.settings["tracking_on"]:
                if self.zooming_out:
                    self.zoom("stop")
                    self.zooming_out = False
                self.lost_subject_frames = 0
                x1, y1, x2, y2 = person_box
                sx, sy = (x1 + x2) / 2, y1 + (y2 - y1) * 0.08
                ox = (sx - tx) / (self.fw / 2)
                oy = (sy - ty) / (self.fh / 2)
                
                dz_x, dz_y = self.settings["deadzone_x"], self.settings["deadzone_y"]
                d = get_direction(ox, oy, dz_x, dz_y)
                
                if self.settings["variable_speed"]:
                    cp = compute_speed(ox, dz_x, self.settings["pan_speed"])
                    ct = compute_speed(oy, dz_y, self.settings["tilt_speed"])
                else: cp, ct = self.settings["pan_speed"], self.settings["tilt_speed"]

                if (d != self.last_dir) or (cp != self.last_pan_speed) or (ct != self.last_tilt_speed):
                    self.move(d, pan=cp, tilt=ct)
                    self.last_dir, self.last_pan_speed, self.last_tilt_speed = d, cp, ct

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(frame, (int(sx), int(sy)), 6, (0, 255, 255), -1)
                self.status_var.set(f"Tracking | {d}")
            else:
                if self.last_dir != "stop":
                    self.move("stop")
                    self.last_dir = "stop"
                
                if self.settings["tracking_on"] and self.settings["auto_zoom_out"]:
                    self.lost_subject_frames += 1
                    delay = self.settings["zoom_out_delay"]
                    if self.lost_subject_frames >= delay and not self.zooming_out:
                        self.zoom("out")
                        self.zooming_out = True
                    self.status_var.set("Zooming out…" if self.zooming_out else f"Subject lost - {delay - self.lost_subject_frames}…")
                else:
                    if self.zooming_out:
                        self.zoom("stop")
                        self.zooming_out = False
                    self.lost_subject_frames = 0
                    self.status_var.set("Tracking OFF" if not self.settings["tracking_on"] else "Subject lost")

        # ── Annotations ──
        if self.capture_mode:
            for (x1, y1, x2, y2, _tid) in self.last_boxes:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 140, 255), 3)
            cv2.putText(frame, "Click a person to track", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 140, 255), 2)

        dzx = int(self.fw / 2 * self.settings["deadzone_x"])
        dzy = int(self.fh / 2 * self.settings["deadzone_y"])
        
        base_color = (150, 150, 150)
        handle_color = (0, 200, 255)
        
        cv2.rectangle(frame, (int(tx) - dzx, int(ty) - dzy), (int(tx) + dzx, int(ty) + dzy), base_color, 1)
        
        # Deadzone drag handles
        h_sz = 12
        # corners
        cv2.rectangle(frame, (int(tx)-dzx-h_sz, int(ty)-dzy-h_sz), (int(tx)-dzx+h_sz, int(ty)-dzy+h_sz), handle_color, -1)
        cv2.rectangle(frame, (int(tx)+dzx-h_sz, int(ty)-dzy-h_sz), (int(tx)+dzx+h_sz, int(ty)-dzy+h_sz), handle_color, -1)
        cv2.rectangle(frame, (int(tx)-dzx-h_sz, int(ty)+dzy-h_sz), (int(tx)-dzx+h_sz, int(ty)+dzy+h_sz), handle_color, -1)
        cv2.rectangle(frame, (int(tx)+dzx-h_sz, int(ty)+dzy-h_sz), (int(tx)+dzx+h_sz, int(ty)+dzy+h_sz), handle_color, -1)
        
        # Target crosshair
        arm = 14
        cv2.line(frame, (int(tx) - arm, int(ty)), (int(tx) + arm, int(ty)), handle_color, 3)
        cv2.line(frame, (int(tx), int(ty) - arm), (int(tx), int(ty) + arm), handle_color, 3)

        trk_label = "● TRACKING ON" if self.settings["tracking_on"] else "○ TRACKING OFF"
        trk_color  = (0, 220, 80) if self.settings["tracking_on"] else (80, 80, 220)
        cv2.putText(frame, trk_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, trk_color, 2, cv2.LINE_AA)

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        display = cv2.resize(frame, (cw, ch)) if cw > 1 and ch > 1 else frame
        self.photo_hold = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB)))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_hold)

    def close(self):
        log.info(f"[{self.name}] Shutting down...")
        self.save_settings()
        self.move("stop")
        if self.cap.isOpened():
            self.cap.release()

# ───────────────────────────────────────── #
# ROOT APPLICATION                          #
# ───────────────────────────────────────── #
def main():
    log.info("─── PTZ Tracker Unified Dashboard Starting ───")
    
    root = tk.Tk()
    root.title("PTZ Tracker - Dual Camera Dashboard")
    root.configure(bg="#000")
    
    # 2-camera grid layout setup
    frame_cam1 = tk.Frame(root, bg="black")
    frame_cam2 = tk.Frame(root, bg="black")
    
    frame_cam1.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    ttk.Separator(root, orient="horizontal").pack(fill=tk.X, pady=2)
    frame_cam2.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    cameras = []
    
    # First Camera
    cam1 = CameraController(
        camera_ip="192.168.200.214", 
        camera_name="Camera 1 - Front",
        parent_frame=frame_cam1,
        config_key="cam1",
        root=root
    )
    cameras.append(cam1)

    # Second Camera
    cam2 = CameraController(
        camera_ip="192.168.200.127", 
        camera_name="Camera 2 - Rear",
        parent_frame=frame_cam2,
        config_key="cam2",
        root=root
    )
    cameras.append(cam2)

    def update_all():
        for cam in cameras:
            cam.update()
        root.after(1, update_all)

    def on_close():
        for cam in cameras:
            cam.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.after(100, update_all)
    
    try:
        root.mainloop()
    except Exception as e:
        log.error(f"Mainloop failed: {e}")

if __name__ == "__main__":
    main()
