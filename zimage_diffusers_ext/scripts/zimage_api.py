"""
Transparent Z-Image (diffusers) integration for Forge.

When a txt2img request selects a Z-Image checkpoint (name matches ZIMAGE_PATTERN),
this extension diverts generation to a diffusers ZImagePipeline (Tongyi-MAI/Z-Image-Turbo)
instead of forge's native SDXL backend. SDXL/other checkpoints are untouched, so forge's
pixel-exact precision is preserved. Works for both /sdapi/v1/txt2img and the sd-queue
extension (which calls the same route endpoint).

Routing key: override_settings.sd_model_checkpoint contains 'z_image' / 'z-image' / 'zimage'.
"""
import base64
import io
import json
import os
import re
import time

import torch
from fastapi import FastAPI

from modules import script_callbacks, shared

ZIMAGE_PATTERN = re.compile(r"z[_\- ]?image", re.IGNORECASE)
ZIMAGE_REPO = os.environ.get("ZIMAGE_REPO", "Tongyi-MAI/Z-Image-Turbo")

_pipe = None


def _log(msg):
    print(f"[ZImage-Diffusers] {msg}", flush=True)


def _is_zimage(req) -> bool:
    try:
        ckpt = (req.override_settings or {}).get("sd_model_checkpoint", "") or ""
    except Exception:
        ckpt = ""
    return bool(ZIMAGE_PATTERN.search(str(ckpt)))


def _get_pipe():
    global _pipe
    if _pipe is None:
        from diffusers import ZImagePipeline
        _log(f"loading {ZIMAGE_REPO} ...")
        t0 = time.time()
        _pipe = ZImagePipeline.from_pretrained(ZIMAGE_REPO, torch_dtype=torch.bfloat16)
        # 12GB VRAM 向け: 逐次オフロード（forgeのモデルとVRAMを取り合わない）
        _pipe.enable_sequential_cpu_offload()
        _log(f"loaded in {time.time()-t0:.0f}s")
    return _pipe


def _b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _generate_zimage(req):
    p = _get_pipe()
    steps = int(getattr(req, "steps", 8) or 8)
    cfg = float(getattr(req, "cfg_scale", 1.0) or 1.0)
    w = int(getattr(req, "width", 1024) or 1024)
    h = int(getattr(req, "height", 1024) or 1024)
    n = int(getattr(req, "batch_size", 1) or 1)
    seed = int(getattr(req, "seed", -1) or -1)
    prompt = getattr(req, "prompt", "") or ""
    neg = getattr(req, "negative_prompt", "") or ""

    gen = None
    if seed is not None and seed >= 0:
        gen = torch.Generator(device="cpu").manual_seed(seed)

    _log(f"generate: steps={steps} cfg={cfg} {w}x{h} n={n} seed={seed}")
    t0 = time.time()
    out = p(
        prompt=prompt,
        negative_prompt=neg or None,
        num_inference_steps=steps,
        guidance_scale=cfg,
        width=w,
        height=h,
        num_images_per_prompt=n,
        generator=gen,
    )
    _log(f"done in {time.time()-t0:.0f}s")

    images = [_b64(im) for im in out.images]
    infotext = (
        f"{prompt}\nNegative prompt: {neg}\n"
        f"Steps: {steps}, CFG scale: {cfg}, Seed: {seed}, Size: {w}x{h}, "
        f"Model: Z-Image-Turbo (diffusers), Backend: diffusers"
    )
    info = {
        "prompt": prompt, "negative_prompt": neg, "seed": seed, "width": w, "height": h,
        "steps": steps, "cfg_scale": cfg, "infotexts": [infotext] * len(images),
        "sd_model_name": "Z-Image-Turbo", "backend": "diffusers",
    }
    return images, info, infotext


def _wrap_txt2img(app: FastAPI):
    """`/sdapi/v1/txt2img` の endpoint をラップし、z_image は diffusers に振り分ける。"""
    from modules.api import models as api_models

    target = None
    for route in app.routes:
        if getattr(route, "path", None) == "/sdapi/v1/txt2img":
            target = route
            break
    if target is None:
        _log("WARN: /sdapi/v1/txt2img route not found; passthrough only")
        return

    original = target.endpoint

    def wrapped(txt2imgreq, *args, **kwargs):
        if _is_zimage(txt2imgreq):
            try:
                images, info, infotext = _generate_zimage(txt2imgreq)
                return api_models.TextToImageResponse(
                    images=images, parameters={}, info=json.dumps(info)
                )
            except Exception as e:
                _log(f"ERROR: {e!r}")
                raise
        return original(txt2imgreq, *args, **kwargs)

    # FastAPIルートと、依存解決用にendpoint属性も差し替え
    target.endpoint = wrapped
    if hasattr(target, "dependant"):
        target.dependant.call = wrapped
    _log("wrapped /sdapi/v1/txt2img (z_image -> diffusers, others -> forge)")


def on_app_started(demo, app: FastAPI):
    _wrap_txt2img(app)


script_callbacks.on_app_started(on_app_started)
