#!/bin/bash
set -euo pipefail

show_alert() {
    local msg="$1"
    /usr/bin/osascript -e "display alert \"PTZ Tracker\" message \"$msg\" as critical buttons {\"OK\"} default button \"OK\""
}

# Resolve project directory from .../PTZ Tracker.app/Contents/MacOS/launch.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_DIR"

pick_python() {
    if [ -x "$PROJECT_DIR/venv-tk/bin/python3" ]; then
        echo "$PROJECT_DIR/venv-tk/bin/python3"
        return
    fi
    if [ -x "$PROJECT_DIR/venv/bin/python3" ]; then
        echo "$PROJECT_DIR/venv/bin/python3"
        return
    fi
    echo ""
}

PYTHON_BIN="$(pick_python)"
if [ -z "$PYTHON_BIN" ]; then
    show_alert "No virtual environment found. Create one with Tk support, then install requirements.txt."
    exit 1
fi

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import _tkinter
import cv2
import requests
import tkinter
from PIL import Image, ImageTk
from ultralytics import YOLO
PY
then
    show_alert "Python dependencies are incomplete for GUI launch. Use a Tk-enabled Python and install requirements.txt in venv-tk."
    exit 1
fi

exec "$PYTHON_BIN" "$PROJECT_DIR/tracker.py"
