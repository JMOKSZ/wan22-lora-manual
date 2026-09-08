#!/bin/bash
# LoRA 工作台: 后端 server.py (aiohttp) + 静态向导页, 绑 0.0.0.0:8331 (tailnet 可用)
cd "$(dirname "$0")"
PY="$HOME/Projects/AI-Tools/ComfyUI/.venv/bin/python"
[ -x "$PY" ] || PY="python3"
exec "$PY" server.py
