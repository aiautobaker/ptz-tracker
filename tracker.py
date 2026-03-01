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
    "deadzone_x":      0.15,
    "deadzone_y":      0.18,
    "pan_speed":       2,
    "tilt_speed":      2,
    "process_every_n": 3,
    "tracking_on":     True,
    # Target offset — 0.0 = frame centre, ±1.0 = frame edge
    "target_x":        0.0,
    "target_y":        0.0,
    # Zoom-aware framing
    "framing_on":        False,
    "framing_wide_y":    0.0,    # vertical target when person is small in frame
    "framing_close_y":  -0.15,   # vertical target when person fills the frame
    "framing_threshold": 35,     # person-height % of frame that triggers "close-up"
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

def move(direction):
    pan, tilt = settings["pan_speed"], settings["tilt_speed"]
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
def get_direction(ox, oy):
    in_x = abs(ox) < settings["deadzone_x"]
    in_y = abs(oy) < settings["deadzone_y"]
    if in_x and in_y:
        return "stop"
    h = "" if in_x else ("right" if ox > 0 else "left")
    v = "" if in_y else ("down"  if oy > 0 else "up")
    return (v + h) if (h and v) else (h or v)

def find_best_person(results):
    best, best_area = None, 0
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) != 0:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area, best = area, (x1, y1, x2, y2)
    return best

def get_all_persons(results):
    """Return a list of (x1,y1,x2,y2) for every detected person."""
    boxes = []
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) != 0:
                continue
            boxes.append(tuple(box.xyxy[0].tolist()))
    return boxes

def compute_hist(bgr_crop):
    """HSV hue-saturation histogram, normalised to [0,1]."""
    if bgr_crop is None or bgr_crop.size == 0:
        return None
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist

def find_target_person(results, frame, ref_hist):
    """Return the box of the person who best matches ref_hist.
    Falls back to the largest person when no reference is set."""
    if ref_hist is None:
        return find_best_person(results)
    best_box, best_score = None, 0.40   # minimum similarity floor
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) != 0:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            crop = frame[int(y1):int(y2), int(x1):int(x2)]
            hist = compute_hist(crop)
            if hist is None:
                continue
            score = cv2.compareHist(ref_hist, hist, cv2.HISTCMP_CORREL)
            if score > best_score:
                best_score, best_box = score, (x1, y1, x2, y2)
    return best_box

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

    # ── Main video window ─────────────────
    root = tk.Tk()
    root.title("PTZ Tracker — Video")
    root.configure(bg="black")
    root.resizable(False, False)

    canvas = tk.Canvas(root, width=fw, height=fh, bg="black",
                       highlightthickness=0, cursor="crosshair")
    canvas.pack()

    # ── Separate controls window ──────────
    ctrl = tk.Toplevel(root)
    ctrl.title("PTZ Tracker — Controls")
    ctrl.configure(bg="#1e1e1e")
    ctrl.resizable(True, True)
    ctrl.geometry("300x420")

    style = ttk.Style(ctrl)
    style.theme_use("default")
    style.configure("TNotebook", background="#1e1e1e", borderwidth=0)
    style.configure("TNotebook.Tab", background="#333", foreground="#ccc",
                    padding=[12, 5], font=("Helvetica", 10))
    style.map("TNotebook.Tab",
              background=[("selected", "#1e1e1e")],
              foreground=[("selected", "#ffffff")])
    style.configure("Vert.TScrollbar", background="#444", troughcolor="#1e1e1e",
                    arrowcolor="#888", borderwidth=0)

    notebook = ttk.Notebook(ctrl)
    notebook.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    tab_tracking = tk.Frame(notebook, bg="#1e1e1e")
    tab_target   = tk.Frame(notebook, bg="#1e1e1e")
    tab_framing  = tk.Frame(notebook, bg="#1e1e1e")
    tab_settings = tk.Frame(notebook, bg="#1e1e1e")

    notebook.add(tab_tracking, text="Tracking")
    notebook.add(tab_target,   text="Target")
    notebook.add(tab_framing,  text="Framing")
    notebook.add(tab_settings, text="Settings")

    # ── Widget helpers ────────────────────
    lbl  = {"bg": "#1e1e1e", "fg": "#cccccc", "font": ("Helvetica", 11)}
    sldr = {"bg": "#1e1e1e", "fg": "#ffffff", "troughcolor": "#444",
            "highlightthickness": 0, "length": 240}

    def make_scrollable(parent):
        """Wrap a tab in a scrollable canvas; return the inner content frame."""
        c  = tk.Canvas(parent, bg="#1e1e1e", highlightthickness=0)
        sb = ttk.Scrollbar(parent, orient="vertical", command=c.yview,
                           style="Vert.TScrollbar")
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

    def make_slider(parent, label, key, lo, hi, res, is_int=False):
        tk.Label(parent, text=label, **lbl).pack(pady=(4, 0))
        var = tk.DoubleVar(value=settings[key])
        def cb(v):
            settings[key] = int(float(v)) if is_int else round(float(v), 2)
        tk.Scale(parent, from_=lo, to=hi, resolution=res,
                 orient="horizontal", variable=var, command=cb, **sldr).pack()
        return var

    def make_slider_pair(parent, label1, key1, label2, key2, lo, hi, res, is_int=False):
        row = tk.Frame(parent, bg="#1e1e1e")
        row.pack(fill=tk.X, pady=(4, 0))
        vars_out = []
        for label, key in [(label1, key1), (label2, key2)]:
            col = tk.Frame(row, bg="#1e1e1e")
            col.pack(side=tk.LEFT, fill=tk.X, expand=True)
            tk.Label(col, text=label, bg="#1e1e1e", fg="#cccccc",
                     font=("Helvetica", 10)).pack(pady=(0, 2))
            var = tk.DoubleVar(value=settings[key])
            def cb(v, k=key):
                settings[k] = int(float(v)) if is_int else round(float(v), 2)
            tk.Scale(col, from_=lo, to=hi, resolution=res, orient="horizontal",
                     variable=var, command=cb, bg="#1e1e1e", fg="#fff",
                     troughcolor="#444", highlightthickness=0, length=110).pack()
            vars_out.append(var)
        return vars_out

    # ── Tab 1: Tracking ───────────────────
    trk = make_scrollable(tab_tracking)

    tk.Label(trk, text="PTZ TRACKER", bg="#1e1e1e", fg="#fff",
             font=("Helvetica", 13, "bold")).pack(pady=(12, 2))

    section(trk, "── DEADZONE ──")
    make_slider_pair(trk, "Horizontal", "deadzone_x", "Vertical", "deadzone_y", 0.02, 0.40, 0.01)

    section(trk, "── SPEED ──")
    make_slider_pair(trk, "Pan Speed", "pan_speed", "Tilt Speed", "tilt_speed", 1, 12, 1, is_int=True)

    section(trk, "── PERFORMANCE ──")
    make_slider(trk, "Process Every N Frames", "process_every_n", 1, 10, 1, is_int=True)

    section(trk, "── TRACKING ──")
    tvar = tk.BooleanVar(value=settings["tracking_on"])
    tk.Checkbutton(trk, text="Tracking Enabled", variable=tvar,
                   command=lambda: settings.update(tracking_on=tvar.get()),
                   bg="#1e1e1e", fg="#fff", selectcolor="#333",
                   font=("Helvetica", 11), activebackground="#1e1e1e").pack(pady=(6, 0))

    status_var = tk.StringVar(value="Waiting for stream…")
    tk.Label(trk, textvariable=status_var, bg="#1e1e1e", fg="#00cc44",
             font=("Helvetica", 10), wraplength=240, justify=tk.LEFT).pack(pady=(4, 8))

    # ── Tab 2: Target ─────────────────────
    tgt = make_scrollable(tab_target)

    section(tgt, "── TARGET POSITION ──")
    tk.Label(tgt, text="Click or drag on the video to move the target.",
             bg="#1e1e1e", fg="#666", font=("Helvetica", 9),
             wraplength=240, justify=tk.LEFT).pack(pady=(4, 0))

    target_x_var, target_y_var = make_slider_pair(
        tgt, "Horizontal", "target_x", "Vertical", "target_y", -0.9, 0.9, 0.01)

    def reset_target():
        settings["target_x"] = 0.0
        settings["target_y"] = 0.0
        target_x_var.set(0.0)
        target_y_var.set(0.0)

    tk.Button(tgt, text="Reset to Centre", command=reset_target,
              bg="#333", fg="#fff", font=("Helvetica", 10),
              relief=tk.FLAT, padx=8, pady=4).pack(pady=(8, 0))

    section(tgt, "── TARGET PERSON ──")
    tk.Label(tgt,
             text="Click 'Capture' then click a person in the video to lock on.\n"
                  "Requires a live camera with a detected person.",
             bg="#1e1e1e", fg="#666", font=("Helvetica", 9),
             wraplength=240, justify=tk.LEFT).pack(pady=(4, 0))

    thumb_label = tk.Label(tgt, bg="#2a2a2a", width=64, height=64,
                           relief=tk.FLAT)
    thumb_label.pack(pady=(6, 2))

    target_status_var = tk.StringVar(value="No target set — tracking largest person")
    tk.Label(tgt, textvariable=target_status_var, bg="#1e1e1e", fg="#aaaaaa",
             font=("Helvetica", 9), wraplength=240, justify=tk.LEFT).pack()

    btn_row = tk.Frame(tgt, bg="#1e1e1e")
    btn_row.pack(pady=(6, 8))

    def do_capture_target():
        capture_mode[0] = True
        target_status_var.set("Click a person in the video…")

    def do_clear_target():
        reference_hist[0] = None
        thumb_photo[0]    = None
        thumb_label.config(image="", width=64, height=64)
        capture_mode[0]   = False
        target_status_var.set("No target set — tracking largest person")
        log.info("Target person cleared")

    tk.Button(btn_row, text="Capture Target", command=do_capture_target,
              bg="#1a5c1a", fg="#fff", font=("Helvetica", 10),
              relief=tk.FLAT, padx=8, pady=4).pack(side=tk.LEFT, padx=4)
    tk.Button(btn_row, text="Clear", command=do_clear_target,
              bg="#5c1a1a", fg="#fff", font=("Helvetica", 10),
              relief=tk.FLAT, padx=8, pady=4).pack(side=tk.LEFT, padx=4)

    # ── Tab 3: Framing ────────────────────
    frm_ = make_scrollable(tab_framing)

    section(frm_, "── ZOOM FRAMING ──")
    tk.Label(frm_,
             text="Auto-adjusts vertical framing based on how large\n"
                  "the person appears (proxy for zoom level).",
             bg="#1e1e1e", fg="#666", font=("Helvetica", 9),
             wraplength=240, justify=tk.LEFT).pack(pady=(4, 0))

    framing_var = tk.BooleanVar(value=settings["framing_on"])
    tk.Checkbutton(frm_, text="Zoom Framing Enabled", variable=framing_var,
                   command=lambda: settings.update(framing_on=framing_var.get()),
                   bg="#1e1e1e", fg="#fff", selectcolor="#333",
                   font=("Helvetica", 11), activebackground="#1e1e1e").pack(pady=(6, 0))

    make_slider_pair(frm_, "Wide Shot  Y", "framing_wide_y", "Close-up  Y", "framing_close_y", -0.9, 0.9, 0.01)
    make_slider(frm_, "Switch Threshold  (% frame height)", "framing_threshold", 5, 80, 1, is_int=True)

    zoom_mode_var = tk.StringVar(value="—")
    framing_row = tk.Frame(frm_, bg="#1e1e1e")
    framing_row.pack(pady=(4, 0))
    tk.Label(framing_row, text="Current:", bg="#1e1e1e", fg="#888",
             font=("Helvetica", 9)).pack(side=tk.LEFT)
    tk.Label(framing_row, textvariable=zoom_mode_var, bg="#1e1e1e", fg="#ffcc44",
             font=("Helvetica", 9, "bold")).pack(side=tk.LEFT, padx=(4, 0))

    # ── Tab 4: Settings ───────────────────
    stg = make_scrollable(tab_settings)

    save_btn = tk.Button(stg, text="Save as Default",
                         bg="#1a3c5c", fg="#fff", font=("Helvetica", 10),
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

    # ── Canvas mouse: capture mode → select person; normal → move target ──
    def set_target_from_mouse(event):
        if capture_mode[0]:
            # Find the bounding box the user clicked inside
            for (x1, y1, x2, y2) in last_boxes[0]:
                if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                    if frame_hold[0] is not None:
                        crop = frame_hold[0][int(y1):int(y2), int(x1):int(x2)]
                        reference_hist[0] = compute_hist(crop)
                        # Build portrait thumbnail (60×90 px)
                        if crop.size > 0:
                            th_bgr = cv2.resize(crop, (60, 90))
                            th_rgb = cv2.cvtColor(th_bgr, cv2.COLOR_BGR2RGB)
                            img = ImageTk.PhotoImage(Image.fromarray(th_rgb))
                            thumb_photo[0] = img
                            thumb_label.config(image=img, width=60, height=90)
                        target_status_var.set("Target locked ✓")
                        log.info("Target person captured from click at (%d,%d)",
                                 event.x, event.y)
                    capture_mode[0] = False
                    return
            # Click missed every box — cancel without changing reference
            capture_mode[0] = False
            target_status_var.set("Target locked ✓" if reference_hist[0] is not None
                                  else "No target set — tracking largest person")
            return

        # Normal behaviour: move tracking target position
        nx = (event.x - fw / 2) / (fw / 2)
        ny = (event.y - fh / 2) / (fh / 2)
        nx = max(-0.9, min(0.9, nx))
        ny = max(-0.9, min(0.9, ny))
        settings["target_x"] = round(nx, 3)
        settings["target_y"] = round(ny, 3)
        target_x_var.set(nx)
        target_y_var.set(ny)

    canvas.bind("<Button-1>",  set_target_from_mouse)
    canvas.bind("<B1-Motion>", lambda e: set_target_from_mouse(e)
                               if not capture_mode[0] else None)

    # ── Frame loop ────────────────────────
    last_dir      = [None]
    frame_num     = [0]
    photo_hold    = [None]
    # ── Person-lock state ─────────────────
    reference_hist = [None]   # HSV histogram of the locked target
    frame_hold     = [None]   # Latest BGR frame (for click-to-capture)
    last_boxes     = [[]]     # Latest detected bounding boxes
    capture_mode   = [False]  # True while waiting for user to click a person
    thumb_photo    = [None]   # Keep thumbnail PhotoImage alive
    zoom_mode      = ["wide"] # current framing mode: "wide" or "close"

    def update():
        try:
            ret, frame = cap.read()
            if not ret:
                log.warning("cap.read() returned False — retrying in 100 ms")
                root.after(100, update)
                return

            frame_hold[0] = frame.copy()   # keep for click-to-capture

            tx = fw / 2 + settings["target_x"] * (fw / 2)
            ty = fh / 2 + settings["target_y"] * (fh / 2)

            frame_num[0] += 1
            if frame_num[0] % settings["process_every_n"] == 0:
                try:
                    results    = model(frame, verbose=False, classes=[0])
                    last_boxes[0] = get_all_persons(results)
                    person_box = find_target_person(results, frame, reference_hist[0])
                except Exception:
                    log.exception("YOLO inference error on frame %d", frame_num[0])
                    person_box = None

                # ── Zoom-aware framing: override ty based on box size ──
                if settings["framing_on"] and person_box:
                    bx1, by1, bx2, by2 = person_box
                    box_frac = (by2 - by1) / fh          # 0–1 height fraction
                    thr      = settings["framing_threshold"] / 100.0
                    hyster   = 0.04                       # ±4 % hysteresis band
                    if zoom_mode[0] == "wide"  and box_frac > thr + hyster:
                        zoom_mode[0] = "close"
                        log.debug("Framing → close-up  (box %.0f%%)", box_frac * 100)
                    elif zoom_mode[0] == "close" and box_frac < thr - hyster:
                        zoom_mode[0] = "wide"
                        log.debug("Framing → wide  (box %.0f%%)", box_frac * 100)
                    eff_y = (settings["framing_close_y"] if zoom_mode[0] == "close"
                             else settings["framing_wide_y"])
                    ty = fh / 2 + eff_y * (fh / 2)
                    zoom_mode_var.set(
                        f"Close-up  ({box_frac*100:.0f}%)" if zoom_mode[0] == "close"
                        else f"Wide  ({box_frac*100:.0f}%)"
                    )
                elif not settings["framing_on"]:
                    zoom_mode_var.set("—")

                if person_box and settings["tracking_on"]:
                    x1, y1, x2, y2 = person_box
                    sx = (x1 + x2) / 2
                    # Track the head: ~8% down from the top of the bounding box
                    sy = y1 + (y2 - y1) * 0.08
                    ox = (sx - tx) / (fw / 2)
                    oy = (sy - ty) / (fh / 2)
                    d  = get_direction(ox, oy)
                    if d != last_dir[0]:
                        log.debug("Direction change: %s → %s", last_dir[0], d)
                        move(d)
                        last_dir[0] = d
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    # Draw dot on the tracked head point
                    cv2.circle(frame, (int(sx), int(sy)), 6, (0, 255, 255), -1)
                    status_var.set(f"Tracking  |  {d}")
                else:
                    if last_dir[0] != "stop":
                        move("stop")
                        last_dir[0] = "stop"
                    status_var.set("Tracking OFF" if not settings["tracking_on"]
                                   else "No subject detected")

            # ── Capture-mode overlay: highlight all detected persons ──
            if capture_mode[0]:
                for (x1, y1, x2, y2) in last_boxes[0]:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                  (0, 140, 255), 3)
                cv2.putText(frame, "Click a person to track",
                            (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                            (0, 140, 255), 2, cv2.LINE_AA)

            # Deadzone box centred on the moveable target
            dzx = int(fw / 2 * settings["deadzone_x"])
            dzy = int(fh / 2 * settings["deadzone_y"])
            cv2.rectangle(frame,
                          (int(tx) - dzx, int(ty) - dzy),
                          (int(tx) + dzx, int(ty) + dzy),
                          (255, 255, 0), 1)

            # Target crosshair (cyan)
            arm = 14
            cv2.line(frame, (int(tx) - arm, int(ty)), (int(tx) + arm, int(ty)), (0, 220, 220), 2)
            cv2.line(frame, (int(tx), int(ty) - arm), (int(tx), int(ty) + arm), (0, 220, 220), 2)
            cv2.circle(frame, (int(tx), int(ty)), 4, (0, 220, 220), -1)

            # HUD
            framing_hud = (f"  FRAMING:{zoom_mode[0].upper()}"
                           if settings["framing_on"] else "")
            cv2.putText(frame,
                        f"DZ {settings['deadzone_x']:.2f}/{settings['deadzone_y']:.2f}  "
                        f"SPD {settings['pan_speed']}/{settings['tilt_speed']}  "
                        f"TGT {settings['target_x']:+.2f},{settings['target_y']:+.2f}"
                        f"{framing_hud}",
                        (10, fh - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
        move("stop")
        cap.release()
        root.destroy()

    # Closing either window shuts everything down
    root.protocol("WM_DELETE_WINDOW", on_close)
    ctrl.protocol("WM_DELETE_WINDOW", on_close)

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
