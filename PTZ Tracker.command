#!/bin/bash
# PTZ Tracker Launcher
# Double-click this file in Finder to start the tracker.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "╔══════════════════════════════════════╗"
echo "║           PTZ TRACKER               ║"
echo "╚══════════════════════════════════════╝"
echo ""

# Pick Python environment (prefer Tk-capable venv-tk, fallback to venv)
if [ -d "venv-tk" ]; then
    VENV_DIR="venv-tk"
elif [ -d "venv" ]; then
    VENV_DIR="venv"
else
    echo "[ERROR] Virtual environment not found."
    echo "        Run this first:"
    echo "        cd $SCRIPT_DIR"
    echo "        python3 -m venv venv-tk"
    echo "        source venv-tk/bin/activate"
    echo "        pip install -r requirements.txt"
    echo ""
    read -p "Press Enter to close..."
    exit 1
fi

if [ ! -f "yolov8n.pt" ]; then
    echo "[ERROR] YOLO model file 'yolov8n.pt' not found in:"
    echo "        $SCRIPT_DIR"
    echo ""
    read -p "Press Enter to close..."
    exit 1
fi

# Activate venv and run
source "$VENV_DIR/bin/activate"

if ! python3 - <<'PY' >/dev/null 2>&1
import _tkinter
PY
then
    echo "[ERROR] This Python environment does not include Tk (_tkinter)."
    echo "        Create venv-tk with a Tk-enabled Python and reinstall requirements:"
    echo "        /usr/local/bin/python3.13 -m venv venv-tk"
    echo "        source venv-tk/bin/activate"
    echo "        pip install -r requirements.txt"
    echo ""
    read -p "Press Enter to close..."
    exit 1
fi

echo "[INFO] Starting PTZ Tracker..."
echo "[INFO] Camera: 192.168.200.214"
echo "[INFO] Press Q in the video window to quit."
echo ""

python3 tracker.py

echo ""
echo "[INFO] PTZ Tracker stopped."
read -p "Press Enter to close this window..."
