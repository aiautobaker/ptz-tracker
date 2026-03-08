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

# ─────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(_LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("ptz")

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
CAMERA_IP = "192.168.200.214"
RTSP_URL  = f"rtsp://{CAMERA_IP}:554/2"
HTTP_BASE = f"http://{CAMERA_IP}/cgi-bin/ptzctrl.cgi"

# ─────────────────────────────────────────
# LIVE SETTINGS  (sliders / mouse write here)
# ─────────────────────────────────────────
settings = {
    "deadzone_x_wide":  0.08,
    "deadzone_y_wide":  0.30,
    "deadzone_x_close": 0.05,
    "deadzone_y_close": 0.15,
    "pan_speed":       3,
    "tilt_speed":      1,
    "variable_speed":  True,
    "smooth_alpha":    0.5,    # EMA weight: 1.0 = raw, 0.1 = very smooth
    "process_every_n": 3,
    "tracking_on":     True,
    # Target offset — 0.0 = frame centre, ±1.0 = frame edge
    "target_x":          0.0,
    "target_y_wide":    -0.4,    # vertical target in wide/normal shot
    "target_y_close":   -0.2,    # vertical target in close-up shot
    # close-up is detected when head (box top) is within 12% of the top of the frame
    # Auto zoom-out on subject loss
    "auto_zoom_out":    False,
    "zoom_out_speed":   3,
    "zoom_out_delay":   15,    # consecutive missed frames before zooming out
}

def load_settings():
    """Merge saved settings.json into the defaults dict."""
    if not _SETTINGS_PATH.exists():
        return
    try:
        saved = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
        for k in settings:
            if k in saved:
                settings[k] = saved[k]
        log.info("Loaded saved settings from %s", _SETTINGS_PATH)
    except Exception:
        log.exception("Could not load settings file — using defaults")

def save_settings():
    """Write the current settings dict to settings.json."""
    try:
        _SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        log.info("Settings saved to %s", _SETTINGS_PATH)
    except Exception:
        log.exception("Could not save settings file")

# ─────────────────────────────────────────
# PTZ CONTROL
# ─────────────────────────────────────────
def send_ptz(cmd):
    try:
        requests.get(f"{HTTP_BASE}?ptzcmd&{cmd}", timeout=0.5)
    except Exception as e:
        log.warning(f"PTZ command failed ({cmd}): {e}")

def zoom(direction):
    speed = settings["zoom_out_speed"]
    cmds = {
        "in":   f"zoomin&{speed}&0",
        "out":  f"zoomout&{speed}&0",
        "stop": "zoomstop&0&0",
    }
    if direction in cmds:
        send_ptz(cmds[direction])

def compute_speed(offset, deadzone, max_speed):
    """Proportionally map |offset| → speed in [1, max_speed].
    Returns 1 at the deadzone edge, max_speed at ±1.0."""
    beyond = max(0.0, abs(offset) - deadzone)
    frac   = beyond / max(1e-6, 1.0 - deadzone)
    return max(1, min(max_speed, round(1 + frac * (max_speed - 1))))

def move(direction, pan=None, tilt=None):
    pan  = pan  if pan  is not None else settings["pan_speed"]
    tilt = tilt if tilt is not None else settings["tilt_speed"]
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
        send_ptz(cmds[direction])

# ─────────────────────────────────────────
# TRACKING LOGIC
# ─────────────────────────────────────────
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

def compute_hist(bgr_crop):
    """HSV histogram — used only for thumbnail generation."""
    if bgr_crop is None or bgr_crop.size == 0:
        return None
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    log.info("─── PTZ Tracker starting ───")
    log.info(f"Log file: {_LOG_PATH}")
    load_settings()

    log.info("Loading YOLO model...")
    if not _MODEL_PATH.exists():
        log.error(f"Model not found: {_MODEL_PATH}")
        return
    try:
        model = YOLO(str(_MODEL_PATH))
    except Exception:
        log.exception("Failed to load YOLO model")
        return

    log.info(f"Connecting to {RTSP_URL}")
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        log.error("Could not open RTSP stream. Check camera IP and network.")
        return

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info(f"Stream: {fw}x{fh}  |  Close the video window to quit.")

    # ── Single combined window ────────────
    root = tk.Tk()
    root.title("PTZ Tracker")
    root.configure(bg="black")
    root.resizable(True, True)
    root.minsize(fw + 420, fh)

    main_frame = tk.Frame(root, bg="black")
    main_frame.pack(fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(main_frame, bg="black",
                       highlightthickness=0, cursor="crosshair")
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # ── Settings panel (right side) ───────
    ctrl = tk.Frame(main_frame, bg="#1e1e1e", width=420)
    ctrl.pack(side=tk.LEFT, fill=tk.Y)
    ctrl.pack_propagate(False)

    style = ttk.Style(root)
    style.theme_use("default")
    style.configure("TNotebook", background="#1e1e1e", borderwidth=0)
    style.configure("TNotebook.Tab", background="#333", foreground="#ccc",
                    padding=[12, 5], font=("Helvetica", 10))
    style.map("TNotebook.Tab",
              background=[("selected", "#1e1e1e")],
              foreground=[("selected", "#ffffff")])
    style.configure("Vert.TScrollbar", background="#444", troughcolor="#1e1e1e",
                    arrowcolor="#888", borderwidth=0)

    status_var = tk.StringVar(value="Waiting for stream…")

    def _toggle_tracking():
        settings["tracking_on"] = not settings["tracking_on"]

    notebook = ttk.Notebook(ctrl)
    notebook.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    tab_tracking = tk.Frame(notebook, bg="#1e1e1e")
    tab_target   = tk.Frame(notebook, bg="#1e1e1e")
    tab_framing  = tk.Frame(notebook, bg="#1e1e1e")
    tab_settings = tk.Frame(notebook, bg="#1e1e1e")
    tab_controls = tk.Frame(notebook, bg="#1e1e1e")

    notebook.add(tab_tracking, text="Tracking")
    notebook.add(tab_target,   text="Target")
    notebook.add(tab_framing,  text="Framing")
    notebook.add(tab_settings, text="Settings")
    notebook.add(tab_controls, text="Controls")

    # ── Widget helpers ────────────────────

    def make_scrollable(parent):
        """Wrap a tab in a scrollable canvas; return the inner content frame."""
        c  = tk.Canvas(parent, bg="#1e1e1e", highlightthickness=0)
        sb = ttk.Scrollbar(parent, orient="vertical", command=c.yview)
        inner = tk.Frame(c, bg="#1e1e1e", padx=12)
        win   = c.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: c.configure(scrollregion=c.bbox("all")))
        c.bind("<Configure>",     lambda e: c.itemconfig(win, width=e.width))
        c.configure(yscrollcommand=sb.set)
        # macOS / Windows mouse-wheel
        c.bind("<MouseWheel>",
               lambda e: c.yview_scroll(int(-1 * (e.delta / 120)), "units"))
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        c.pack(side=tk.LEFT,  fill=tk.BOTH, expand=True)
        return inner

    def section(parent, t):
        tk.Label(parent, text=t, bg="#1e1e1e", fg="#888",
                 font=("Helvetica", 9)).pack(pady=(10, 0))

    _ent_style = dict(bg="#2a2a2a", fg="#ffffff", insertbackground="#fff",
                      relief=tk.FLAT, font=("Helvetica", 11), width=8)

    def make_entry(parent, label, key, lo, hi, res=None, is_int=False):
        row = tk.Frame(parent, bg="#1e1e1e")
        row.pack(fill=tk.X, pady=(4, 0), padx=8)
        tk.Label(row, text=label, bg="#1e1e1e", fg="#cccccc",
                 font=("Helvetica", 10), anchor="w").pack(side=tk.LEFT)
        fmt = (lambda v: str(int(v))) if is_int else (lambda v: f"{v:.3f}")
        var = tk.StringVar(value=fmt(settings[key]))
        ent = tk.Entry(row, textvariable=var, **_ent_style)
        ent.pack(side=tk.RIGHT)
        def apply(e=None):
            try:
                v = int(float(var.get())) if is_int else round(float(var.get()), 3)
                v = max(lo, min(hi, v))
                settings[key] = v
                var.set(fmt(v))
            except ValueError:
                var.set(fmt(settings[key]))
        ent.bind("<Return>", apply)
        ent.bind("<FocusOut>", apply)
        return var

    def make_entry_pair(parent, label1, key1, label2, key2, lo, hi, res=None, is_int=False):
        row = tk.Frame(parent, bg="#1e1e1e")
        row.pack(fill=tk.X, pady=(4, 0), padx=8)
        fmt = (lambda v: str(int(v))) if is_int else (lambda v: f"{v:.3f}")
        vars_out = []
        for label, key in [(label1, key1), (label2, key2)]:
            tk.Label(row, text=label + ":", bg="#1e1e1e", fg="#aaaaaa",
                     font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(0, 2))
            var = tk.StringVar(value=fmt(settings[key]))
            ent = tk.Entry(row, textvariable=var, **{**_ent_style, "width": 7})
            ent.pack(side=tk.LEFT, padx=(0, 12))
            def apply(e=None, k=key, v=var):
                try:
                    val = int(float(v.get())) if is_int else round(float(v.get()), 3)
                    val = max(lo, min(hi, val))
                    settings[k] = val
                    v.set(fmt(val))
                except ValueError:
                    v.set(fmt(settings[k]))
            ent.bind("<Return>", apply)
            ent.bind("<FocusOut>", apply)
            vars_out.append(var)
        return vars_out

    # Keep old names as aliases so call sites need no changes
    make_slider      = make_entry
    make_slider_pair = make_entry_pair

    # ── Tab 1: Tracking ───────────────────
    trk = make_scrollable(tab_tracking)

    tk.Label(trk, text="PTZ TRACKER", bg="#1e1e1e", fg="#fff",
             font=("Helvetica", 13, "bold")).pack(pady=(12, 2))

    section(trk, "── DEADZONE (Wide Shot) ──")
    make_slider_pair(trk, "Horizontal", "deadzone_x_wide", "Vertical", "deadzone_y_wide", 0.02, 0.40, 0.01)

    section(trk, "── DEADZONE (Close-up) ──")
    make_slider_pair(trk, "Horizontal", "deadzone_x_close", "Vertical", "deadzone_y_close", 0.02, 0.40, 0.01)

    section(trk, "── SPEED ──")
    vs_var = tk.BooleanVar(value=settings["variable_speed"])
    tk.Checkbutton(trk, text="Variable Speed (auto)", variable=vs_var,
                   command=lambda: settings.update(variable_speed=vs_var.get()),
                   bg="#1e1e1e", fg="#fff", selectcolor="#333",
                   font=("Helvetica", 11), activebackground="#1e1e1e").pack(pady=(6, 0))
    tk.Label(trk, text="When on, sliders set the maximum speed.",
             bg="#1e1e1e", fg="#666", font=("Helvetica", 9)).pack()
    make_slider_pair(trk, "Pan Speed", "pan_speed", "Tilt Speed", "tilt_speed", 1, 12, 1, is_int=True)

    section(trk, "── PERFORMANCE ──")
    make_slider(trk, "Process Every N Frames", "process_every_n", 1, 10, 1, is_int=True)
    make_slider(trk, "Position Smoothing  (1.0 = raw, 0.1 = max smooth)",
                "smooth_alpha", 0.1, 1.0, 0.05)


    # ── Tab 2: Target ─────────────────────
    tgt = make_scrollable(tab_target)

    section(tgt, "── TARGET POSITION ──")
    tk.Label(tgt, text="Click or drag on the video to move the target.\n"
                       "Vertical target per mode is set in the Framing tab.",
             bg="#1e1e1e", fg="#666", font=("Helvetica", 9),
             wraplength=380, justify=tk.LEFT).pack(pady=(4, 0))

    target_x_var = make_slider(tgt, "Horizontal", "target_x", -0.9, 0.9, 0.01)

    _mode_ty_vars = {}  # {"wide": DoubleVar, "close": DoubleVar} — filled in Framing tab

    def reset_target():
        settings["target_x"]    = 0.0
        settings["target_y_wide"]  = -0.4
        settings["target_y_close"] = -0.2
        target_x_var.set("0.000")
        for k, v in _mode_ty_vars.items():
            v.set(f"{settings[f'target_y_{k}']:.3f}")

    tk.Button(tgt, text="Reset to Centre", command=reset_target,
              bg="#333", fg="#000", font=("Helvetica", 10),
              relief=tk.FLAT, padx=8, pady=4).pack(pady=(8, 0))

    section(tgt, "── TARGET PERSON ──")
    tk.Label(tgt,
             text="Click 'Capture' then click a person in the video to lock on.\n"
                  "Requires a live camera with a detected person.",
             bg="#1e1e1e", fg="#666", font=("Helvetica", 9),
             wraplength=380, justify=tk.LEFT).pack(pady=(4, 0))

    thumb_label = tk.Label(tgt, bg="#2a2a2a", width=0, height=0,
                           relief=tk.FLAT)
    thumb_label.pack(pady=(6, 2))

    target_status_var = tk.StringVar(value="No target set — tracking largest person")
    tk.Label(tgt, textvariable=target_status_var, bg="#1e1e1e", fg="#aaaaaa",
             font=("Helvetica", 9), wraplength=380, justify=tk.LEFT).pack()

    btn_row = tk.Frame(tgt, bg="#1e1e1e")
    btn_row.pack(pady=(6, 8))

    def do_capture_target():
        capture_mode[0] = True
        target_status_var.set("Click a person in the video…")

    def do_clear_target():
        reference_track_id[0] = None
        smooth_box[0]         = None
        thumb_photo[0]        = None
        thumb_label.config(image="", width=0, height=0)
        capture_mode[0]       = False
        target_status_var.set("No target set — tracking largest person")
        log.info("Target person cleared")

    tk.Button(btn_row, text="Capture Target", command=do_capture_target,
              bg="#1a5c1a", fg="#000", font=("Helvetica", 10),
              relief=tk.FLAT, padx=8, pady=4).pack(side=tk.LEFT, padx=4)
    tk.Button(btn_row, text="Clear", command=do_clear_target,
              bg="#5c1a1a", fg="#000", font=("Helvetica", 10),
              relief=tk.FLAT, padx=8, pady=4).pack(side=tk.LEFT, padx=4)

    # ── Tab 3: Framing ────────────────────
    frm_ = make_scrollable(tab_framing)

    section(frm_, "── VERTICAL TARGET ──")
    tk.Label(frm_,
             text="Set where in the frame the target point sits.\n"
                  "Close-up is detected when the head (top of\n"
                  "bounding box) is near the top of the frame.",
             bg="#1e1e1e", fg="#666", font=("Helvetica", 9),
             wraplength=380, justify=tk.LEFT).pack(pady=(4, 0))

    ty_w_var, ty_c_var = make_slider_pair(
        frm_, "Wide Y", "target_y_wide", "Close-up Y", "target_y_close", -0.9, 0.9, 0.01)
    _mode_ty_vars["wide"]  = ty_w_var
    _mode_ty_vars["close"] = ty_c_var

    zoom_mode_var = tk.StringVar(value="Wide")
    framing_row = tk.Frame(frm_, bg="#1e1e1e")
    framing_row.pack(pady=(4, 0))
    tk.Label(framing_row, text="Current mode:", bg="#1e1e1e", fg="#888",
             font=("Helvetica", 9)).pack(side=tk.LEFT)
    tk.Label(framing_row, textvariable=zoom_mode_var, bg="#1e1e1e", fg="#ffcc44",
             font=("Helvetica", 9, "bold")).pack(side=tk.LEFT, padx=(4, 0))

    section(frm_, "── AUTO ZOOM-OUT ──")
    tk.Label(frm_, text="Zoom out when subject is lost.\nStops automatically when subject is reacquired.",
             bg="#1e1e1e", fg="#666", font=("Helvetica", 9),
             wraplength=380, justify=tk.LEFT).pack(pady=(4, 0))
    az_var = tk.BooleanVar(value=settings["auto_zoom_out"])
    tk.Checkbutton(frm_, text="Auto Zoom-out Enabled", variable=az_var,
                   command=lambda: settings.update(auto_zoom_out=az_var.get()),
                   bg="#1e1e1e", fg="#fff", selectcolor="#333",
                   font=("Helvetica", 11), activebackground="#1e1e1e").pack(pady=(6, 0))
    make_slider(frm_, "Zoom-out Speed", "zoom_out_speed", 1, 12, 1, is_int=True)
    make_slider(frm_, "Delay  (missed frames)", "zoom_out_delay", 1, 60, 1, is_int=True)

    # ── Tab 4: Settings ───────────────────
    stg = make_scrollable(tab_settings)

    save_btn = tk.Button(stg, text="Save as Default",
                         bg="#1a3c5c", fg="#000", font=("Helvetica", 10),
                         relief=tk.FLAT, padx=8, pady=4)
    save_btn.pack(pady=(24, 4))
    save_status_var = tk.StringVar(value="")
    tk.Label(stg, textvariable=save_status_var, bg="#1e1e1e", fg="#5599cc",
             font=("Helvetica", 9)).pack(pady=(0, 16))

    def do_save_settings():
        save_settings()
        save_status_var.set("Saved ✓")
        save_btn.config(text="Saved ✓")
        stg.after(1500, lambda: (save_status_var.set(""),
                                 save_btn.config(text="Save as Default")))

    save_btn.config(command=do_save_settings)

    # ── Tab 5: Controls ───────────────────
    tk.Label(tab_controls, text="MANUAL CONTROLS", bg="#1e1e1e", fg="#fff",
             font=("Helvetica", 12, "bold")).pack(pady=(16, 4))
    tk.Label(tab_controls, text="Hold to move  •  Release to stop",
             bg="#1e1e1e", fg="#666", font=("Helvetica", 9)).pack(pady=(0, 12))

    _db = dict(font=("Helvetica", 16), relief=tk.FLAT,
               activebackground="#555", cursor="hand2", width=3, height=1,
               fg="#000000")

    def _manual_press(cmd):
        manual_override[0] = True
        move(cmd)

    def _manual_release():
        move("stop")
        manual_override[0] = False

    def _dpad_btn(parent, text, cmd, row, col):
        b = tk.Button(parent, text=text, bg="#333", **_db)
        b.grid(row=row, column=col, padx=3, pady=3)
        b.bind("<ButtonPress-1>",   lambda e, c=cmd: _manual_press(c))
        b.bind("<ButtonRelease-1>", lambda e: _manual_release())

    dpad = tk.Frame(tab_controls, bg="#1e1e1e")
    dpad.pack()

    _dpad_btn(dpad, "↖", "upleft",    0, 0)
    _dpad_btn(dpad, "↑", "up",        0, 1)
    _dpad_btn(dpad, "↗", "upright",   0, 2)
    _dpad_btn(dpad, "←", "left",      1, 0)
    stop_b = tk.Button(dpad, text="■", bg="#4a1a1a", **_db)
    stop_b.grid(row=1, column=1, padx=3, pady=3)
    stop_b.bind("<ButtonPress-1>", lambda e: _manual_press("stop"))
    _dpad_btn(dpad, "→", "right",     1, 2)
    _dpad_btn(dpad, "↙", "downleft",  2, 0)
    _dpad_btn(dpad, "↓", "down",      2, 1)
    _dpad_btn(dpad, "↘", "downright", 2, 2)

    zoom_row = tk.Frame(tab_controls, bg="#1e1e1e")
    zoom_row.pack(pady=(16, 0))

    _zb = dict(font=("Helvetica", 11), relief=tk.FLAT,
               activebackground="#555", cursor="hand2", padx=14, pady=8)

    zin_b  = tk.Button(zoom_row, text="＋  Zoom In",  bg="#1a3c1a", fg="#000000", **_zb)
    zout_b = tk.Button(zoom_row, text="－  Zoom Out", bg="#3c1a1a", fg="#000000", **_zb)
    zin_b .pack(side=tk.LEFT, padx=6)
    zout_b.pack(side=tk.LEFT, padx=6)
    zin_b .bind("<ButtonPress-1>",   lambda e: zoom("in"))
    zin_b .bind("<ButtonRelease-1>", lambda e: zoom("stop"))
    zout_b.bind("<ButtonPress-1>",   lambda e: zoom("out"))
    zout_b.bind("<ButtonRelease-1>", lambda e: zoom("stop"))

    # ── Canvas mouse: capture mode → select person; normal → move target ──
    def set_target_from_mouse(event):
        # Scale canvas click coords → native frame coords
        cw = canvas.winfo_width()  or fw
        ch = canvas.winfo_height() or fh
        mx = event.x * fw / cw
        my = event.y * fh / ch

        # Click on tracking indicator (top-left region) → toggle
        if mx < 200 and my < 45:
            _toggle_tracking()
            return

        if capture_mode[0]:
            # Find the bounding box the user clicked inside
            for (x1, y1, x2, y2, tid) in last_boxes[0]:
                if x1 <= mx <= x2 and y1 <= my <= y2:
                    reference_track_id[0] = tid
                    smooth_box[0] = None   # reset smoothing for new target
                    if frame_hold[0] is not None:
                        crop = frame_hold[0][int(y1):int(y2), int(x1):int(x2)]
                        # Build portrait thumbnail (60×90 px)
                        if crop.size > 0:
                            th_bgr = cv2.resize(crop, (60, 90))
                            th_rgb = cv2.cvtColor(th_bgr, cv2.COLOR_BGR2RGB)
                            img = ImageTk.PhotoImage(Image.fromarray(th_rgb))
                            thumb_photo[0] = img
                            thumb_label.config(image=img, width=60, height=90)
                        target_status_var.set(f"Target locked ✓  (ID {tid})")
                        log.info("Target locked: track ID %d at (%d,%d)", tid,
                                 mx, my)
                    capture_mode[0] = False
                    return
            # Click missed every box — cancel without changing reference
            capture_mode[0] = False
            target_status_var.set(
                f"Target locked ✓  (ID {reference_track_id[0]})"
                if reference_track_id[0] is not None
                else "No target set — tracking largest person")
            return

        # Normal behaviour: move tracking target position
        nx = (mx - fw / 2) / (fw / 2)
        ny = (my - fh / 2) / (fh / 2)
        nx = max(-0.9, min(0.9, nx))
        ny = max(-0.9, min(0.9, ny))
        settings["target_x"] = round(nx, 3)
        settings[f"target_y_{zoom_mode[0]}"] = round(ny, 3)
        target_x_var.set(f"{nx:.3f}")
        v = _mode_ty_vars.get(zoom_mode[0])
        if v is not None:
            v.set(f"{ny:.3f}")

    canvas.bind("<Button-1>",  set_target_from_mouse)
    canvas.bind("<B1-Motion>", lambda e: set_target_from_mouse(e)
                               if not capture_mode[0] else None)

    # ── Frame loop ────────────────────────
    last_dir      = [None]
    frame_num     = [0]
    photo_hold    = [None]
    # ── Person-lock state ─────────────────
    reference_track_id = [None]  # ByteTrack ID of the locked target
    smooth_box         = [None]  # EMA-smoothed (x1,y1,x2,y2) of tracked person
    frame_hold         = [None]  # Latest BGR frame (for click-to-capture)
    last_boxes         = [[]]    # Latest track boxes: (x1,y1,x2,y2,tid)
    capture_mode   = [False]  # True while waiting for user to click a person
    thumb_photo    = [None]   # Keep thumbnail PhotoImage alive
    zoom_mode           = ["wide"]  # current framing mode: "wide" or "close"
    lost_subject_frames = [0]
    manual_override     = [False]  # True while a manual control button is held
    last_pan_speed      = [None]
    last_tilt_speed     = [None]
    zooming_out         = [False]

    def update():
        try:
            ret, frame = cap.read()
            if not ret:
                log.warning("cap.read() returned False — retrying in 100 ms")
                root.after(100, update)
                return

            frame_hold[0] = frame.copy()   # keep for click-to-capture

            tx = fw / 2 + settings["target_x"] * (fw / 2)
            ty = fh / 2 + settings[f"target_y_{zoom_mode[0]}"] * (fh / 2)

            frame_num[0] += 1
            if manual_override[0]:
                status_var.set("Manual control")
            elif frame_num[0] % settings["process_every_n"] == 0:
                try:
                    results       = model.track(frame, verbose=False, classes=[0],
                                                persist=True)
                    track_boxes   = get_track_boxes(results)
                    last_boxes[0] = track_boxes
                    raw_box       = find_tracked_person(track_boxes, reference_track_id[0])
                except Exception:
                    log.exception("YOLO/tracker error on frame %d", frame_num[0])
                    raw_box = None

                # EMA position smoothing
                if raw_box:
                    if smooth_box[0] is None:
                        smooth_box[0] = list(raw_box)
                    else:
                        a = settings["smooth_alpha"]
                        smooth_box[0] = [a * n + (1 - a) * s
                                         for n, s in zip(raw_box, smooth_box[0])]
                    person_box = tuple(smooth_box[0])
                else:
                    smooth_box[0] = None
                    person_box = None

                # ── Detect wide vs close-up using head (box top) position ──
                # If the head is within 12% of frame top → close-up; above 18% → wide
                if person_box:
                    bx1, by1, bx2, by2 = person_box
                    head_frac = by1 / fh               # 0 = top of frame, 1 = bottom
                    if zoom_mode[0] == "wide"  and head_frac < 0.12:
                        zoom_mode[0] = "close"
                        log.debug("Mode → close-up  (head at %.0f%%)", head_frac * 100)
                    elif zoom_mode[0] == "close" and head_frac > 0.18:
                        zoom_mode[0] = "wide"
                        log.debug("Mode → wide  (head at %.0f%%)", head_frac * 100)
                    zoom_mode_var.set(
                        f"Close-up  (head {head_frac*100:.0f}%)" if zoom_mode[0] == "close"
                        else f"Wide  (head {head_frac*100:.0f}%)"
                    )
                else:
                    zoom_mode_var.set(zoom_mode[0].capitalize())

                if person_box and settings["tracking_on"]:
                    # Subject found — reset zoom-out state
                    if zooming_out[0]:
                        zoom("stop")
                        zooming_out[0] = False
                        log.info("Auto zoom-out stopped — subject reacquired")
                    lost_subject_frames[0] = 0
                    x1, y1, x2, y2 = person_box
                    sx = (x1 + x2) / 2
                    # Track the head: ~8% down from the top of the bounding box
                    sy = y1 + (y2 - y1) * 0.08
                    ox = (sx - tx) / (fw / 2)
                    oy = (sy - ty) / (fh / 2)
                    suffix = "close" if zoom_mode[0] == "close" else "wide"
                    dz_x = settings[f"deadzone_x_{suffix}"]
                    dz_y = settings[f"deadzone_y_{suffix}"]
                    d = get_direction(ox, oy, dz_x, dz_y)
                    if settings["variable_speed"]:
                        c_pan  = compute_speed(ox, dz_x, settings["pan_speed"])
                        c_tilt = compute_speed(oy, dz_y, settings["tilt_speed"])
                    else:
                        c_pan, c_tilt = settings["pan_speed"], settings["tilt_speed"]
                    if (d != last_dir[0] or c_pan != last_pan_speed[0]
                            or c_tilt != last_tilt_speed[0]):
                        if d != last_dir[0]:
                            log.debug("Direction change: %s → %s", last_dir[0], d)
                        move(d, pan=c_pan, tilt=c_tilt)
                        last_dir[0]       = d
                        last_pan_speed[0]  = c_pan
                        last_tilt_speed[0] = c_tilt
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    # Draw dot on the tracked head point
                    cv2.circle(frame, (int(sx), int(sy)), 6, (0, 255, 255), -1)
                    status_var.set(f"Tracking  |  {d}")
                else:
                    if last_dir[0] != "stop":
                        move("stop")
                        last_dir[0] = "stop"
                    if settings["tracking_on"] and settings["auto_zoom_out"]:
                        lost_subject_frames[0] += 1
                        delay = settings["zoom_out_delay"]
                        if lost_subject_frames[0] >= delay and not zooming_out[0]:
                            zoom("out")
                            zooming_out[0] = True
                            log.info("Auto zoom-out triggered after %d missed frames",
                                     lost_subject_frames[0])
                        if zooming_out[0]:
                            status_var.set("Zooming out…")
                        else:
                            remaining = delay - lost_subject_frames[0]
                            status_var.set(f"No subject — zoom-out in {remaining}…")
                    else:
                        if zooming_out[0]:
                            zoom("stop")
                            zooming_out[0] = False
                        lost_subject_frames[0] = 0
                        status_var.set("Tracking OFF" if not settings["tracking_on"]
                                       else "No subject detected")

            # ── Capture-mode overlay: highlight all detected persons ──
            if capture_mode[0]:
                for (x1, y1, x2, y2, _tid) in last_boxes[0]:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                  (0, 140, 255), 3)
                cv2.putText(frame, "Click a person to track",
                            (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                            (0, 140, 255), 2, cv2.LINE_AA)

            # Deadzone box centred on the moveable target
            _sfx = "close" if zoom_mode[0] == "close" else "wide"
            dzx = int(fw / 2 * settings[f"deadzone_x_{_sfx}"])
            dzy = int(fh / 2 * settings[f"deadzone_y_{_sfx}"])
            cv2.rectangle(frame,
                          (int(tx) - dzx, int(ty) - dzy),
                          (int(tx) + dzx, int(ty) + dzy),
                          (255, 255, 0), 1)

            # Target crosshair (cyan)
            arm = 14
            cv2.line(frame, (int(tx) - arm, int(ty)), (int(tx) + arm, int(ty)), (0, 220, 220), 2)
            cv2.line(frame, (int(tx), int(ty) - arm), (int(tx), int(ty) + arm), (0, 220, 220), 2)
            cv2.circle(frame, (int(tx), int(ty)), 4, (0, 220, 220), -1)

            # Tracking indicator (top-left) — click to toggle
            trk_label = "● TRACKING ON" if settings["tracking_on"] else "○ TRACKING OFF"
            trk_color  = (0, 220, 80)   if settings["tracking_on"] else (80, 80, 220)
            cv2.putText(frame, trk_label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, trk_color, 2, cv2.LINE_AA)

            # Wide/close-up badge (top-right) — always visible
            badge_label = "CLOSE-UP" if zoom_mode[0] == "close" else "WIDE"
            badge_color = (0, 140, 255) if zoom_mode[0] == "close" else (160, 80, 0)
            (tw, th), _ = cv2.getTextSize(badge_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            pad = 6
            bx1, by1 = fw - tw - pad * 2 - 10, 10
            bx2, by2 = fw - 10, by1 + th + pad * 2
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), badge_color, -1)
            cv2.putText(frame, badge_label, (bx1 + pad, by2 - pad),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

            # HUD
            cv2.putText(frame,
                        f"DZ {settings[f'deadzone_x_{_sfx}']:.2f}/{settings[f'deadzone_y_{_sfx}']:.2f}  "
                        f"SPD {settings['pan_speed']}/{settings['tilt_speed']}  "
                        f"TGT {settings['target_x']:+.2f},{settings[f'target_y_{_sfx}']:+.2f}",
                        (10, fh - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cw = canvas.winfo_width()
            ch = canvas.winfo_height()
            display = cv2.resize(frame, (cw, ch)) if cw > 1 and ch > 1 else frame
            rgb   = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            photo_hold[0] = photo

            root.after(1, update)

        except Exception:
            log.exception("Unhandled exception in update() — frame %d", frame_num[0])
            status_var.set("ERROR — check tracker.log")
            # Attempt to keep the loop alive rather than silently die
            root.after(500, update)

    def on_close():
        log.info("Shutting down — stop command sent to camera")
        save_settings()
        move("stop")
        cap.release()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    root.after(0, update)
    try:
        root.mainloop()
    except Exception:
        log.exception("Unhandled exception in mainloop")
    finally:
        log.info("─── PTZ Tracker stopped ───")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Fatal error in main()")
        sys.exit(1)
