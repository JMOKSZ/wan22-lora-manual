#!/usr/bin/env python3
"""LoRA 训练工作台 —— 向导式后端 (aiohttp, 单文件)
绑定 0.0.0.0:8331, tailnet 内可用。静态页 + JSON API + ComfyUI 代理。
"""
import asyncio
import hashlib
import json
import os
import shutil
import time
import zipfile
from pathlib import Path

from aiohttp import web, ClientSession, FormData, ClientTimeout
from PIL import Image

ROOT = Path(__file__).resolve().parent
COMFY = "http://127.0.0.1:8188"
COMFY_DIR = Path.home() / "Projects/AI-Tools/ComfyUI"
MODELS = COMFY_DIR / "models"
LORAS_DIR = MODELS / "loras"
STATE_FILE = ROOT / "state.json"
RUNS_FILE = ROOT / "runs.json"
CACHE_DIR = ROOT / ".cache"
PROXY = "http://127.0.0.1:7897"  # 本机代理, 用于下载境外模型文件
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

WAN_GGUF_LOW = "Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf"
WAN_GGUF_HIGH = "Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf"
WAN_LORA_LOW = "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"
WAN_LORA_HIGH = "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"
WAN_CLIP = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
WAN_VAE = "wan_2.1_vae.safetensors"
WAN_NEGATIVE = (
    "色调艳丽, 过曝, 静态, 细节模糊不清, 字幕, 风格, 作品, 画作, 画面, 静止, 整体发灰, "
    "最差质量, 低质量, JPEG压缩残留, 丑陋的, 残缺的, 多余的手指, 画得不好的手部, "
    "画得不好的脸部, 畸形的, 毁容的, 形态畸形的肢体, 手指融合, 静止不动的画面, "
    "杂乱的背景, 三条腿, 背景人很多, 倒着走"
)

# ---------------------------------------------------------------- state ----
def load_json(path: Path, default):
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception:
        return default

def save_json(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")

def get_state():
    return load_json(STATE_FILE, {
        "step": 1, "trigger": "", "dataset_dir": "", "zip_path": "",
        "current_lora": "", "expert": "low",
        "done": {"env": False, "scan": False, "captions": False,
                 "zip": False, "train": False, "install": False, "test": False},
    })

# ---------------------------------------------------------------- comfy ----
async def comfy_up() -> bool:
    try:
        async with ClientSession(timeout=ClientTimeout(total=3)) as s:
            async with s.get(COMFY + "/system_stats") as r:
                return r.status == 200
    except Exception:
        return False

async def queue_prompt(workflow: dict) -> str:
    async with ClientSession() as s:
        async with s.post(COMFY + "/prompt", json={"prompt": workflow}) as r:
            data = await r.json()
            if "prompt_id" not in data:
                raise web.HTTPBadRequest(text=json.dumps(data, ensure_ascii=False))
            return data["prompt_id"]

def build_identity_test(prompt: str, negative: str, image_name: str,
                        width: int, height: int, frames: int, fps: int,
                        seed: int, lora_name: str, strength: float,
                        expert: str) -> dict:
    """在已验证的双专家 I2V 工作流上串联身份 LoRA。
    expert=low: 只挂 LowNoise 链; expert=both: 两个专家都挂。"""
    wf = {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": WAN_GGUF_HIGH}},
        "2": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": WAN_GGUF_LOW}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": WAN_CLIP, "type": "wan"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": WAN_VAE}},
        "5": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "lora_name": WAN_LORA_HIGH, "strength_model": 1.0, "model": ["1", 0]}},
        "6": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "lora_name": WAN_LORA_LOW, "strength_model": 1.0, "model": ["2", 0]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["3", 0]}},
        "8": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["3", 0]}},
        "9": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "10": {"class_type": "WanImageToVideo", "inputs": {
            "width": width, "height": height, "length": frames, "batch_size": 1,
            "positive": ["7", 0], "negative": ["8", 0],
            "vae": ["4", 0], "start_image": ["9", 0]}},
        "11": {"class_type": "KSamplerAdvanced", "inputs": {
            "add_noise": "enable", "noise_seed": seed, "steps": 4, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple",
            "start_at_step": 0, "end_at_step": 2, "return_with_leftover_noise": "enable",
            "model": ["5", 0],
            "positive": ["10", 0], "negative": ["10", 1], "latent_image": ["10", 2]}},
        "12": {"class_type": "KSamplerAdvanced", "inputs": {
            "add_noise": "disable", "noise_seed": seed, "steps": 4, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple",
            "start_at_step": 2, "end_at_step": 4, "return_with_leftover_noise": "disable",
            "model": ["20", 0],
            "positive": ["10", 0], "negative": ["10", 1], "latent_image": ["11", 0]}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["4", 0]}},
        "14": {"class_type": "VHS_VideoCombine", "inputs": {
            "frame_rate": fps, "loop_count": 0, "filename_prefix": "lora_test",
            "format": "video/h264-mp4", "pingpong": False, "save_output": True,
            "images": ["13", 0]}},
        # 身份 LoRA 串在 LowNoise 链末尾 (lightx2v 之后)
        "20": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "lora_name": lora_name, "strength_model": strength, "model": ["6", 0]}},
    }
    if expert == "both":
        wf["21"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
            "lora_name": lora_name, "strength_model": strength, "model": ["5", 0]}}
        wf["11"]["inputs"]["model"] = ["21", 0]
    return wf

# ---------------------------------------------------------------- routes ---
routes = web.RouteTableDef()

@routes.get("/")
async def index(_):
    return web.FileResponse(ROOT / "workbench.html")

@routes.get("/manual")
async def manual(_):
    return web.FileResponse(ROOT / "index.html")

@routes.get("/api/status")
async def api_status(_):
    up = await comfy_up()
    df = shutil.disk_usage(str(Path.home()))
    def exists(p): return p.exists()
    return web.json_response({
        "comfy": up,
        "models": {
            "low_noise_gguf": exists(MODELS / "diffusion_models" / WAN_GGUF_LOW) or exists(MODELS / "unet" / WAN_GGUF_LOW),
            "high_noise_gguf": exists(MODELS / "unet" / WAN_GGUF_HIGH) or exists(MODELS / "diffusion_models" / WAN_GGUF_HIGH),
            "lightx2v_low": exists(LORAS_DIR / WAN_LORA_LOW),
            "lightx2v_high": exists(LORAS_DIR / WAN_LORA_HIGH),
            "clip": exists(MODELS / "text_encoders" / WAN_CLIP),
            "vae": exists(MODELS / "vae" / WAN_VAE),
        },
        "loras_writable": os.access(LORAS_DIR, os.W_OK),
        "disk_free_gb": round(df.free / 1e9, 1),
        "state": get_state(),
    })

@routes.post("/api/scan")
async def api_scan(req):
    body = await req.json()
    d = Path(os.path.expanduser(body.get("path", ""))).resolve()
    if not d.is_dir():
        return web.json_response({"error": f"目录不存在: {d}"}, status=400)
    items = []
    for f in sorted(d.iterdir()):
        if f.suffix.lower() not in IMG_EXTS or f.name.startswith("."):
            continue
        try:
            with Image.open(f) as im:
                w, h = im.size
        except Exception:
            continue
        short = min(w, h)
        ratio = w / h
        bucket = ("1:1" if abs(ratio - 1) < 0.1 else
                  "3:4" if abs(ratio - 0.75) < 0.1 else
                  "4:3" if abs(ratio - 4 / 3) < 0.1 else
                  "9:16" if abs(ratio - 9 / 16) < 0.12 else "其他")
        items.append({
            "name": f.name, "w": w, "h": h, "short": short, "bucket": bucket,
            "small": short < 1024,
            "captioned": (d / (f.stem + ".txt")).exists(),
        })
    return web.json_response({"dir": str(d), "count": len(items), "items": items})

@routes.get("/api/thumb")
async def api_thumb(req):
    d = Path(os.path.expanduser(req.query.get("dir", "")))
    name = req.query.get("name", "")
    f = (d / name).resolve()
    if d.resolve() not in f.parents or not f.is_file():
        return web.Response(status=404)
    CACHE_DIR.mkdir(exist_ok=True)
    key = hashlib.md5(f"{f}{f.stat().st_mtime}".encode()).hexdigest()
    cache = CACHE_DIR / f"{key}.jpg"
    if not cache.exists():
        try:
            with Image.open(f) as im:
                im = im.convert("RGB")
                im.thumbnail((320, 320))
                im.save(cache, "JPEG", quality=82)
        except Exception:
            return web.Response(status=415)
    return web.FileResponse(cache, headers={"Cache-Control": "max-age=3600"})

@routes.get("/api/photo")
async def api_photo(req):
    """打标界面用的中等尺寸图 (最长边 900)。"""
    d = Path(os.path.expanduser(req.query.get("dir", "")))
    f = (d / req.query.get("name", "")).resolve()
    if d.resolve() not in f.parents or not f.is_file():
        return web.Response(status=404)
    try:
        with Image.open(f) as im:
            im = im.convert("RGB")
            im.thumbnail((900, 900))
            import io
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=88)
            return web.Response(body=buf.getvalue(), content_type="image/jpeg")
    except Exception:
        return web.Response(status=415)

@routes.get("/api/captions")
async def api_captions_get(req):
    d = Path(os.path.expanduser(req.query.get("dir", "")))
    out = {}
    if d.is_dir():
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in IMG_EXTS and not f.name.startswith("."):
                t = d / (f.stem + ".txt")
                out[f.name] = t.read_text("utf-8").strip() if t.exists() else ""
    return web.json_response(out)

@routes.post("/api/caption")
async def api_caption_post(req):
    body = await req.json()
    d = Path(os.path.expanduser(body["dir"]))
    f = (d / body["name"]).resolve()
    if d.resolve() not in f.parents:
        return web.Response(status=403)
    (d / (f.stem + ".txt")).write_text(body.get("caption", "").strip(), "utf-8")
    return web.json_response({"ok": True})

@routes.post("/api/zip")
async def api_zip(req):
    body = await req.json()
    d = Path(os.path.expanduser(body.get("path", ""))).resolve()
    if not d.is_dir():
        return web.json_response({"error": "目录不存在"}, status=400)
    zip_path = d.parent / (d.name + ".zip")
    n_img = n_txt = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(d.iterdir()):
            if f.name.startswith("."):
                continue
            if f.suffix.lower() in IMG_EXTS:
                z.write(f, f.name); n_img += 1
                t = d / (f.stem + ".txt")
                if t.exists():
                    z.write(t, t.name); n_txt += 1
    st = get_state()
    st["zip_path"] = str(zip_path)
    save_json(STATE_FILE, st)
    return web.json_response({
        "zip": str(zip_path), "images": n_img, "captions": n_txt,
        "size_mb": round(zip_path.stat().st_size / 1e6, 1),
    })

@routes.get("/api/loras")
async def api_loras(_):
    out = []
    for f in sorted(LORAS_DIR.glob("*.safetensors")):
        out.append({"name": f.name, "size_mb": round(f.stat().st_size / 1e6, 1),
                    "mtime": f.stat().st_mtime})
    return web.json_response(out)

@routes.post("/api/lora/install")
async def api_lora_install(req):
    body = await req.json()
    source = os.path.expanduser(body.get("source", "").strip())
    name = body.get("name", "").strip()
    if not name.endswith(".safetensors"):
        name += ".safetensors"
    dest = LORAS_DIR / name
    if source.startswith("http://") or source.startswith("https://"):
        timeout = ClientTimeout(total=1800)
        async with ClientSession(timeout=timeout) as s:
            async with s.get(source, proxy=PROXY) as r:
                if r.status != 200:
                    return web.json_response({"error": f"下载失败 HTTP {r.status}"}, status=502)
                with open(dest, "wb") as fp:
                    async for chunk in r.content.iter_chunked(1 << 20):
                        fp.write(chunk)
    else:
        src = Path(source)
        if not src.is_file():
            return web.json_response({"error": f"文件不存在: {src}"}, status=400)
        shutil.copy2(src, dest)
    mb = dest.stat().st_size / 1e6
    if mb < 5:
        dest.unlink()
        return web.json_response({"error": "文件过小, 疑似下载损坏, 已删除"}, status=502)
    return web.json_response({"ok": True, "name": name, "size_mb": round(mb, 1)})

@routes.put("/api/upload")
async def api_upload(req):
    """浏览器 PUT 原图字节 → 转发到 ComfyUI /upload/image。"""
    name = req.query.get("name", "ref.png")
    data = await req.read()
    form = FormData()
    form.add_field("image", data, filename=name, content_type="application/octet-stream")
    async with ClientSession() as s:
        async with s.post(COMFY + "/upload/image", data=form) as r:
            resp = await r.json()
    return web.json_response({"name": resp.get("name", name)})

@routes.post("/api/test")
async def api_test(req):
    b = await req.json()
    strengths = [float(x) for x in b.get("strengths", [0.8])]
    seed = int(b.get("seed", 42))
    runs = load_json(RUNS_FILE, [])
    submitted = []
    for st_val in strengths:
        wf = build_identity_test(
            prompt=b["prompt"], negative=b.get("negative") or WAN_NEGATIVE,
            image_name=b["image_name"],
            width=int(b.get("width", 640)), height=int(b.get("height", 640)),
            frames=int(b.get("frames", 49)), fps=int(b.get("fps", 16)),
            seed=seed, lora_name=b["lora"], strength=st_val,
            expert=b.get("expert", "low"),
        )
        pid = await queue_prompt(wf)
        run = {
            "id": pid, "ts": time.strftime("%Y-%m-%d %H:%M"),
            "lora": b["lora"], "strength": st_val, "expert": b.get("expert", "low"),
            "seed": seed, "prompt": b["prompt"][:120],
            "width": int(b.get("width", 640)), "height": int(b.get("height", 640)),
            "frames": int(b.get("frames", 49)),
            "score": 0, "note": "", "video": None,
        }
        runs.insert(0, run)
        submitted.append(run)
    save_json(RUNS_FILE, runs)
    return web.json_response({"submitted": submitted})

@routes.get("/api/test/status")
async def api_test_status(req):
    ids = req.query.get("ids", "").split(",")
    out = {}
    async with ClientSession(timeout=ClientTimeout(total=5)) as s:
        for pid in ids:
            if not pid:
                continue
            try:
                async with s.get(f"{COMFY}/history/{pid}") as r:
                    h = await r.json()
            except Exception:
                out[pid] = {"status": "unknown"}
                continue
            if pid not in h:
                out[pid] = {"status": "running"}
                continue
            entry = h[pid]
            status = entry.get("status", {})
            done = status.get("completed", False)
            video = None
            for node_out in entry.get("outputs", {}).values():
                for key in ("gifs", "videos", "images"):
                    for item in node_out.get(key, []):
                        video = item
                        break
            out[pid] = {"status": "done" if done else "running", "video": video}
    # 回填 runs.json 里的 video 字段
    runs = load_json(RUNS_FILE, [])
    changed = False
    for run in runs:
        v = out.get(run["id"], {}).get("video")
        if v and not run.get("video"):
            run["video"] = v
            changed = True
    if changed:
        save_json(RUNS_FILE, runs)
    return web.json_response(out)

@routes.get("/api/view")
async def api_view(req):
    """代理 ComfyUI /view, 让 tailnet 设备能播放输出视频。"""
    qs = req.query_string
    async with ClientSession() as s:
        async with s.get(f"{COMFY}/view?{qs}") as r:
            data = await r.read()
            ct = r.headers.get("Content-Type", "application/octet-stream")
    return web.Response(body=data, content_type=ct)

@routes.route("*", "/api/state")
async def api_state(req):
    if req.method == "PUT":
        body = await req.json()
        st = get_state()
        st.update(body)
        save_json(STATE_FILE, st)
    return web.json_response(get_state())

@routes.get("/api/runs")
async def api_runs(_):
    return web.json_response(load_json(RUNS_FILE, []))

@routes.post("/api/rate")
async def api_rate(req):
    b = await req.json()
    runs = load_json(RUNS_FILE, [])
    for run in runs:
        if run["id"] == b["id"]:
            run["score"] = int(b.get("score", 0))
            run["note"] = b.get("note", "")
            break
    save_json(RUNS_FILE, runs)
    return web.json_response({"ok": True})

# ---------------------------------------------------------------- main -----
def main():
    app = web.Application(client_max_size=512 << 20)
    app.add_routes(routes)
    print("LoRA 工作台 → http://0.0.0.0:8331 (tailnet: http://100.71.5.29:8331)")
    web.run_app(app, host="0.0.0.0", port=8331, print=None)

if __name__ == "__main__":
    main()
