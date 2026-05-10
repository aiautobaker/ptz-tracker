#!/usr/bin/env python3
"""
PTZ Web Controller – browser-based gamepad/touch PTZ camera control.

Run:  python app.py
Open: http://localhost:8080  (or your Mac's LAN IP on any device)
"""

import os
import socket
import logging

import requests
from flask import Flask, jsonify, request, send_from_directory

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/static/<path:path>")
def static_files(path: str):
    return send_from_directory("static", path)


@app.route("/api/ptz", methods=["POST"])
def ptz():
    """Proxy a PTZ command to the camera CGI endpoint."""
    data = request.get_json(silent=True) or {}
    ip   = data.get("ip",      "").strip()
    cmd  = data.get("command", "").strip()
    pan  = max(1, min(32, int(data.get("pan_speed",  8))))
    tilt = max(1, min(32, int(data.get("tilt_speed", 8))))

    if not ip or not cmd:
        return jsonify(ok=False, error="Missing ip or command"), 400

    # Build the CGI URL
    if cmd in ("ptzstop", "zoomstop"):
        url = f"http://{ip}/cgi-bin/ptzctrl.cgi?ptzcmd&{cmd}&0&0"
    elif cmd in ("zoomin", "zoomout"):
        url = f"http://{ip}/cgi-bin/ptzctrl.cgi?ptzcmd&{cmd}&{pan}&0"
    else:
        url = f"http://{ip}/cgi-bin/ptzctrl.cgi?ptzcmd&{cmd}&{pan}&{tilt}"

    log.info("PTZ  %s  →  %s", ip, url.split("?")[-1])

    try:
        r = requests.get(url, timeout=2)
        return jsonify(ok=True, status=r.status_code)
    except requests.exceptions.Timeout:
        return jsonify(ok=False, error="Camera timeout"), 504
    except Exception as exc:
        return jsonify(ok=False, error=str(exc)), 500


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    try:
        lan_ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        lan_ip = "?.?.?.?"

    print()
    print("  PTZ Web Controller")
    print("  ══════════════════════════════════════")
    print("  Local  :  http://localhost:8080")
    print(f"  Network:  http://{lan_ip}:8080")
    print("  ══════════════════════════════════════")
    print()
    app.run(host="0.0.0.0", port=8080, threaded=True, debug=False)
