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
import random
import re
import time

import torch
from PIL import Image
from fastapi import FastAPI

from modules import script_callbacks, shared

ZIMAGE_PATTERN = re.compile(r"z[_\- ]?image", re.IGNORECASE)
ZIMAGE_REPO = os.environ.get("ZIMAGE_REPO", "Tongyi-MAI/Z-Image-Turbo")

_pipe = None
_img2img = None


def _log(msg):
    print(f"[ZImage-Diffusers] {msg}", flush=True)


def _is_zimage(req) -> bool:
    try:
        ckpt = (req.override_settings or {}).get("sd_model_checkpoint", "") or ""
    except Exception:
        ckpt = ""
    return bool(ZIMAGE_PATTERN.search(str(ckpt)))


def _apply_offload(pipe):
    """forge と同じ発想のオフロードを diffusers で行う(量子化なし=無損失)。

    group: transformer だけをブロック単位で pinned + 非同期 CUDA stream 転送し、
    text_encoder(~3.75GB) と vae(160MB)は cuda 常駐(forge の model_load 相当)。
    text_encoder を group offload すると埋め込みの index_select で cpu/cuda 不整合に
    なるため常駐させる。旧 sequential はモデル全体を毎ステップ転送するので遅い。
    env ZIMAGE_OFFLOAD で sequential(既定/安定)/group(検証中)/model/none を切替。
    """
    mode = os.environ.get("ZIMAGE_OFFLOAD", "sequential").lower()
    if mode == "model":
        pipe.enable_model_cpu_offload()
        _log("offload=model")
        return
    if mode == "none":
        pipe.to("cuda")
        _log("offload=none (full resident)")
        return
    if mode == "group":
        try:
            onload, offload = torch.device("cuda"), torch.device("cpu")
            nbpg = int(os.environ.get("ZIMAGE_BLOCKS_PER_GROUP", "1") or "1")
            pipe.text_encoder.to(onload)  # 常駐(index_select 不整合回避)
            pipe.vae.to(onload)           # 160MB なので常駐
            pipe.transformer.enable_group_offload(
                onload_device=onload, offload_device=offload, offload_type="block_level",
                num_blocks_per_group=nbpg, use_stream=True, record_stream=True,
            )
            _log(f"offload=group(transformer block_level nbpg={nbpg}, stream; TE/vae resident)")
            return
        except Exception as e:
            _log(f"group offload failed ({e!r}); fallback sequential")
    pipe.enable_sequential_cpu_offload()
    _log("offload=sequential")


def _get_pipe():
    global _pipe
    if _pipe is None:
        from diffusers import ZImagePipeline
        _log(f"loading {ZIMAGE_REPO} ...")
        t0 = time.time()
        _pipe = ZImagePipeline.from_pretrained(ZIMAGE_REPO, torch_dtype=torch.bfloat16)
        # Z-Image の VAE は float32 のまま読まれることがあり、img2img(HR)の
        # prepare_latents が画像を bf16 にして vae.encode するため dtype 不整合
        # (Input BFloat16 / bias float)で落ちる。VAE も bf16 に揃える。
        try:
            _pipe.vae.to(dtype=torch.bfloat16)
        except Exception as e:
            _log(f"vae bf16 cast skipped: {e!r}")
        # HiRes(2x で 1664x2432 等)の VAE encode/decode は活性化が巨大で、
        # 12GB GPU だと一括 3.8GB+ の確保に失敗し CUDA OOM になる。タイル/スライス
        # 分割で確保ピークを下げる(無損失。継ぎ目は内部オーバーラップで処理)。
        for meth in ("enable_tiling", "enable_slicing"):
            try:
                getattr(_pipe.vae, meth)()
            except Exception as e:
                _log(f"vae {meth} skipped: {e!r}")
        _apply_offload(_pipe)
        _log(f"loaded in {time.time()-t0:.0f}s")
    return _pipe


def _get_img2img():
    """HiRes Fix の 2nd pass 用。base と全コンポーネントを共有(再ロード/追加VRAM無し)。"""
    global _img2img
    if _img2img is None:
        from diffusers import ZImageImg2ImgPipeline
        _img2img = ZImageImg2ImgPipeline.from_pipe(_get_pipe())
    return _img2img


def _align(v, m=16):
    v = int(v)
    return max(m, v - (v % m))


def _b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _req_dict(req):
    """pydantic リクエストを dict 化(forge の parameters echo 用)。"""
    for m in ("model_dump", "dict"):
        fn = getattr(req, m, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    try:
        return dict(vars(req))
    except Exception:
        return {}


def _ckpt_name(req):
    try:
        n = (req.override_settings or {}).get("sd_model_checkpoint", "") or ""
    except Exception:
        n = ""
    return str(n) or "Z-Image-Turbo"


def _make_infotext(prompt, neg, steps, sampler, cfg, seed, w, h, model,
                   denoise=None, hr=None):
    """A1111/forge 互換の 1行 infotext を組み立てる。"""
    parts = [
        f"Steps: {steps}", f"Sampler: {sampler}", f"CFG scale: {cfg}",
        f"Seed: {seed}", f"Size: {w}x{h}", f"Model: {model}",
    ]
    if denoise is not None:
        parts.append(f"Denoising strength: {denoise}")
    if hr:  # (hr_scale, hr_steps, hr_upscaler)
        parts += [
            f"Hires upscale: {hr[0]}", f"Hires steps: {hr[1]}",
            f"Hires upscaler: {hr[2]}",
        ]
    parts.append("Version: forge-zimage-diffusers")
    line = ", ".join(parts)
    head = prompt
    if neg:
        head += f"\nNegative prompt: {neg}"
    return f"{head}\n{line}"


def _progress_cb(st, offset):
    """diffusers の step コールバックで forge の shared.state を更新(queue 進捗用)。"""
    def cb(pipe, step, timestep, cbk):
        try:
            st.sampling_step = offset + step + 1
        except Exception:
            pass
        return cbk
    return cb


def _generate_zimage(req):
    p = _get_pipe()
    steps = int(getattr(req, "steps", 8) or 8)
    cfg = float(getattr(req, "cfg_scale", 1.0) or 1.0)
    w = int(getattr(req, "width", 1024) or 1024)
    h = int(getattr(req, "height", 1024) or 1024)
    n = int(getattr(req, "batch_size", 1) or 1)
    seed = int(getattr(req, "seed", -1) or -1)
    if seed < 0:  # forge は info.seed に実際に使った seed を返す(-1 のままにしない)
        seed = random.randint(0, 2**32 - 1)
    prompt = getattr(req, "prompt", "") or ""
    neg = getattr(req, "negative_prompt", "") or ""
    gen = torch.Generator(device="cpu").manual_seed(seed)

    # --- HiRes Fix パラメータを先に解決(進捗の総ステップ数算出のため) ---
    # forge の txt2img をバイパスしているため HR は自前で行う。base 画像を目標解像度へ
    # アップスケールし、ZImageImg2ImgPipeline で denoising_strength 分だけ焼き直す。
    enable_hr = bool(getattr(req, "enable_hr", False))
    fw, fh = w, h
    denoise = None
    hr_meta = None
    tw = th = hr_steps = 0
    hr_actual = 0
    if enable_hr:
        hr_scale = float(getattr(req, "hr_scale", 2.0) or 2.0)
        hr_rx = int(getattr(req, "hr_resize_x", 0) or 0)
        hr_ry = int(getattr(req, "hr_resize_y", 0) or 0)
        if hr_rx > 0 and hr_ry > 0:
            tw, th = _align(hr_rx), _align(hr_ry)
        else:
            tw, th = _align(w * hr_scale), _align(h * hr_scale)
        hr_steps = int(getattr(req, "hr_second_pass_steps", 0) or 0) or steps
        denoise = getattr(req, "denoising_strength", None)
        denoise = float(denoise) if denoise is not None else 0.7
        hr_upscaler = getattr(req, "hr_upscaler", "") or "Lanczos"
        hr_prompt = getattr(req, "hr_prompt", "") or prompt
        hr_neg = getattr(req, "hr_negative_prompt", "") or neg
        hr_actual = max(1, int(round(denoise * hr_steps)))  # img2img の実ステップ数

    # --- forge 進捗(shared.state)を初期化 → queue の /sdapi/v1/progress が動く ---
    # begin() が time_start も設定する。これが無いと progressapi が
    # `time.time() - state.time_start` で None 減算 TypeError になり queue が落ちる。
    st = shared.state
    try:
        st.begin("zimage")
        st.job_count = 1
        st.job_no = 0
        st.sampling_steps = steps + hr_actual
        st.sampling_step = 0
    except Exception:
        pass

    _log(f"generate: steps={steps} cfg={cfg} {w}x{h} n={n} seed={seed} hr={enable_hr}")
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
        callback_on_step_end=_progress_cb(st, 0),
    )
    pil_images = list(out.images)
    _log(f"base done in {time.time()-t0:.0f}s")

    if enable_hr:
        i2i = _get_img2img()
        upscaled = [im.convert("RGB").resize((tw, th), Image.LANCZOS) for im in pil_images]
        _log(f"hires: {w}x{h} -> {tw}x{th} steps={hr_steps} denoise={denoise}")
        t1 = time.time()
        gen2 = torch.Generator(device="cpu").manual_seed(seed)
        hr_out = i2i(
            prompt=hr_prompt,
            negative_prompt=hr_neg or None,
            image=upscaled if len(upscaled) > 1 else upscaled[0],
            strength=denoise,
            num_inference_steps=hr_steps,
            guidance_scale=cfg,
            width=tw,
            height=th,
            num_images_per_prompt=1,
            generator=gen2,
            callback_on_step_end=_progress_cb(st, steps),
        )
        pil_images = list(hr_out.images)
        fw, fh = tw, th
        hr_meta = (hr_scale, hr_steps, hr_upscaler)
        _log(f"hires done in {time.time()-t1:.0f}s")

    _log(f"done in {time.time()-t0:.0f}s")

    images = [_b64(im) for im in pil_images]
    count = len(images)
    sampler = getattr(req, "sampler_name", None) or getattr(req, "sampler_index", None) \
        or p.scheduler.__class__.__name__
    model = _ckpt_name(req)
    infotext = _make_infotext(prompt, neg, steps, sampler, cfg, seed, fw, fh, model,
                              denoise=denoise, hr=hr_meta)

    extra = {"Backend": "diffusers (Z-Image)"}
    if hr_meta:
        extra.update({"Hires upscale": hr_meta[0], "Hires steps": hr_meta[1],
                      "Hires upscaler": hr_meta[2]})

    # forge の Processed.js() と同一キー構造で返す(クライアント互換)。
    info = {
        "prompt": prompt,
        "all_prompts": [prompt] * count,
        "negative_prompt": neg,
        "all_negative_prompts": [neg] * count,
        "seed": seed,
        "all_seeds": [seed] * count,
        "subseed": seed,
        "all_subseeds": [seed] * count,
        "subseed_strength": 0.0,
        "width": fw,
        "height": fh,
        "sampler_name": sampler,
        "cfg_scale": cfg,
        "steps": steps,
        "batch_size": n,
        "restore_faces": False,
        "face_restoration_model": None,
        "sd_model_name": model,
        "sd_model_hash": None,
        "sd_vae_name": None,
        "sd_vae_hash": None,
        "seed_resize_from_w": -1,
        "seed_resize_from_h": -1,
        "denoising_strength": denoise,
        "extra_generation_params": extra,
        "index_of_first_image": 0,
        "infotexts": [infotext] * count,
        "styles": [],
        "job_timestamp": time.strftime("%Y%m%d%H%M%S"),
        "clip_skip": 1,
        "is_using_inpainting_conditioning": False,
        "version": "forge-zimage-diffusers",
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
                    images=images, parameters=_req_dict(txt2imgreq), info=json.dumps(info)
                )
            except Exception as e:
                _log(f"ERROR: {e!r}")
                raise
            finally:
                # 進捗をクリア(/sdapi/v1/progress を完了状態に戻す)。
                # end() は job_count=0/job="" にし torch_gc も行う。
                try:
                    shared.state.end()
                    shared.state.sampling_step = 0
                except Exception:
                    pass
        return original(txt2imgreq, *args, **kwargs)

    # FastAPIルートと、依存解決用にendpoint属性も差し替え
    target.endpoint = wrapped
    if hasattr(target, "dependant"):
        target.dependant.call = wrapped
    _log("wrapped /sdapi/v1/txt2img (z_image -> diffusers, others -> forge)")


def on_app_started(demo, app: FastAPI):
    _wrap_txt2img(app)


script_callbacks.on_app_started(on_app_started)
