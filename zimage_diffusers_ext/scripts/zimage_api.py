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
import gc
import io
import json
import os
import random
import re
import time

import torch
from PIL import Image, ImageFilter
from fastapi import FastAPI

from modules import script_callbacks, shared

ZIMAGE_PATTERN = re.compile(r"z[_\- ]?image", re.IGNORECASE)
ZIMAGE_REPO = os.environ.get("ZIMAGE_REPO", "Tongyi-MAI/Z-Image-Turbo")
USERDATA_DIR = os.environ.get("USERDATA_DIR", "/app/data")
ADETAILER_EXT_DIR = os.path.join(USERDATA_DIR, "extensions", "adetailer")

_pipe = None
_img2img = None
_inpaint = None


def _log(msg):
    print(f"[ZImage-Diffusers] {msg}", flush=True)


def _vram(tag):
    """GPU メモリ状況をログ(OOM 切り分け用)。"""
    try:
        free, total = torch.cuda.mem_get_info()
        alloc = torch.cuda.memory_allocated()
        _log(f"vram[{tag}] alloc={alloc/2**30:.2f}G free={free/2**30:.2f}G/{total/2**30:.2f}G")
    except Exception:
        pass


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
    既定 group(forge風・高速): 収まる分は常駐 + 溢れ分のみ非同期 stream 転送。
    LoRA 適用/解除のたびに _reapply_group_offload() で diffusers 本来の再適用
    (_maybe_remove_and_reapply_group_offloading)を明示的に呼び、lora パラメータも
    stream 対象に含める(pipe.load_lora_weights 経路では自動発火しないため)。
    env ZIMAGE_OFFLOAD で sequential/model/none に上書き可。
    """
    mode = os.environ.get("ZIMAGE_OFFLOAD", "group").lower()
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
            pipe.vae.to(onload)           # 160MB なので常駐
            # transformer(diffusers モデル)は group offload が正しく動く(検証済: 常駐0.16G、
            # base 53s/hires 167s、LoRA も自動再適用される)。record_stream=False が重要:
            # True だと先読みブロックの解放が遅延し prefetch 暴走で全ブロック抱え込み OOM。
            pipe.transformer.enable_group_offload(
                onload_device=onload, offload_device=offload, offload_type="block_level",
                num_blocks_per_group=nbpg, use_stream=True, record_stream=False,
            )
            # text_encoder(transformers の Qwen3)は diffusers group offload が壊れる
            # (leaf を onload するが offload で戻さず OOM、検証済)。CPU 常駐にし、encode 時
            # だけ _encode() が GPU へ一括ロード→即 CPU 退避する(transformer が 0.16G しか
            # 使わないので 7.7G+活性化が収まる)。
            pipe.text_encoder.to(offload)
            _log(f"offload=group(transformer block_level nbpg={nbpg}, stream rec=False; vae resident, TE=CPU/GPU-swap-encode)")
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


def _get_inpaint():
    """ADetailer の再描画用。base と全コンポーネントを共有(再ロード/追加VRAM無し)。"""
    global _inpaint
    if _inpaint is None:
        from diffusers import ZImageInpaintPipeline
        _inpaint = ZImageInpaintPipeline.from_pipe(_get_pipe())
    return _inpaint


def _encode(pipe, prompt, neg, cfg):
    """プロンプトを embeds 化して返す(全パイプライン呼び出しを prompt_embeds 経由にし、
    denoising ループ中に TE を使わない)。TE(~7.7GB)を encode の間だけ GPU に一括ロード
    →即 CPU 退避する。transformer は group offload で 0.16G しか使わないので、TE 一括でも
    収まる。diffusers の encode は pos/neg 逐次(batch1)なので活性化は小さい。
    group offload 時のみ swap。sequential 等は accelerate が TE を管理するので素通し。"""
    do_cfg = bool(cfg is not None and cfg > 0)
    group_mode = os.environ.get("ZIMAGE_OFFLOAD", "group").lower() == "group"
    # max_sequence_length を実トークン長に絞る。512 固定だと padding で encode 活性化が
    # 大きく(~3.8G)、TE 一括ロード(7.7G)と合わせて OOM する。実長に詰めれば ~1/3。
    max_seq = 512
    try:
        n_p = len(pipe.tokenizer(prompt or "").input_ids)
        n_n = len(pipe.tokenizer(neg).input_ids) if neg else 0
        max_seq = min(512, max(64, max(n_p, n_n) + 16))
    except Exception:
        pass
    if group_mode:
        try:
            pipe.text_encoder.to("cuda")
        except Exception:
            pass
    try:
        pe, ne = pipe.encode_prompt(
            prompt=prompt, device=torch.device("cuda"),
            do_classifier_free_guidance=do_cfg,
            negative_prompt=(neg or None),
            max_sequence_length=max_seq,
        )
    finally:
        if group_mode:
            try:
                pipe.text_encoder.to("cpu")
                torch.cuda.empty_cache()
            except Exception:
                pass
    return pe, ne


def _resolve_lora(name):
    """name → LoRA ファイルパス。forge の LoRA レジストリ(エイリアス対応)を再利用。
    forge ネイティブ自身がこのレジストリに依存するため、無ければ forge でも未検出=skip。"""
    import networks
    e = networks.available_networks.get(name) or networks.available_network_aliases.get(name)
    return e.filename if e is not None and getattr(e, "filename", None) else None


def _extract_loras(text):
    """プロンプトから extra network タグを抽出・除去。forge の正規パーサ
    (modules.extra_networks.parse_prompt)を再利用。戻り値 (clean_text, [(name,weight)])."""
    if not text:
        return text, []
    from modules import extra_networks
    clean, data = extra_networks.parse_prompt(text)
    loras = []
    for params in data.get("lora", []):
        items = getattr(params, "items", None) or []
        if not items:
            continue
        name = str(items[0]).strip()
        try:
            w = float(items[1]) if len(items) > 1 else 1.0
        except (ValueError, TypeError):
            w = 1.0
        loras.append((name, w))
    clean = re.sub(r"\s*,\s*,\s*", ", ", clean).strip().strip(",").strip()
    return clean, loras


def _apply_loras(pipe, loras):
    """LoRA を pipe.transformer に適用。diffusers は sequential offload を検知して
    hook を外し→ロード→再適用してくれるのでデバイス不整合は起きない。"""
    names, weights = [], []
    for nm, wt in loras:
        path = _resolve_lora(nm)
        if not path:
            _log(f"lora NOT FOUND, skip: {nm}")
            continue
        adapter = re.sub(r"[^A-Za-z0-9_]", "_", nm)[:64]
        try:
            pipe.load_lora_weights(path, adapter_name=adapter)
            names.append(adapter)
            weights.append(wt)
            _log(f"lora applied: {nm} (w={wt})")
        except Exception as e:
            _log(f"lora load failed {nm}: {e!r}")
    if names:
        try:
            pipe.set_adapters(names, weights)
        except Exception as e:
            _log(f"set_adapters failed: {e!r}")
    return names


def _reset_loras(pipe):
    """前リクエストの LoRA を確実に解除(クラッシュ時の残留対策で生成前に毎回呼ぶ)。"""
    try:
        pipe.unload_lora_weights()
    except Exception:
        pass


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


def _free_forge_vram():
    """forge ネイティブが GPU に常駐させた直前モデル(SDXL KModel/JointTextEncoder 等)を
    退避させる。forge は効率化のため生成後もモデルを VRAM に残すため、直前に SDXL ジョブが
    走っていると 6GB+ 占有が残り、その上で Z-Image(diffusers)の HiRes が OOM する。
    diffusers と forge はメモリ管理を共有しないので、Z-Image 開始前に明示的に空ける。"""
    try:
        from backend import memory_management as mm
        mm.unload_all_models()
        mm.soft_empty_cache(force=True)
        _log("freed forge-resident VRAM before Z-Image")
    except Exception as e:
        _log(f"free_forge_vram skipped: {e!r}")


def _hard_reset_pipe():
    """パイプ本体を破棄して GPU 残骸を確実に開放する。OOM クラッシュ後は group offload の
    onload ブロックや TE が GPU に張り付き、コンポーネント単位の .to('cpu')/再 offload では
    戻らない(Python 参照が残るため empty_cache も効かない)。globals を None にして全参照を
    断ち切り、gc + empty_cache で実メモリを解放する。次回リクエストで from_pretrained 再ロード
    (ディスクキャッシュ済みで数十秒)。クラッシュ時のみ発生するコストなので許容。"""
    global _pipe, _img2img, _inpaint
    _pipe = None
    _img2img = None
    _inpaint = None
    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception:
        pass


def _release_zimage_gpu():
    """diffusers Z-Image パイプが握っている GPU メモリを解放。まず全コンポーネントを CPU 退避
    (パイプは再利用のため保持)。それでも GPU が枯渇したまま(=OOM クラッシュで残骸が張り付いて
    ソフト解放が効かない)なら、パイプ本体を破棄するハード解放にフォールバックする。
    これをしないと残骸が forge ネイティブ生成や次の Z-Image を道連れにし続ける。"""
    if _pipe is None:
        return
    try:
        # transformer は group offload。OOM クラッシュ後は onload 済みブロックが GPU に
        # 残るため、_maybe_remove_and_reapply_group_offloading で hook を貼り直し offloaded
        # (CPU)状態にリセットする(.to("cpu") は group hook と喧嘩して効かない)。
        tr = getattr(_pipe, "transformer", None)
        if tr is not None:
            try:
                from diffusers.hooks.group_offloading import _maybe_remove_and_reapply_group_offloading
                _maybe_remove_and_reapply_group_offloading(tr)
            except Exception:
                pass
        # TE は素の nn.Module(group offload 無し)なので .to("cpu") で確実に退避。
        te = getattr(_pipe, "text_encoder", None)
        if te is not None:
            try:
                te.to("cpu")
            except Exception:
                pass
        torch.cuda.empty_cache()
        # ソフト解放後も空きが少ない=残骸が張り付いている。パイプごと破棄して確実に開放する。
        free, total = torch.cuda.mem_get_info()
        if free < 6 * 2**30:
            _log(f"soft release left only {free/2**30:.2f}G free -> hard reset pipe")
            _hard_reset_pipe()
            free, total = torch.cuda.mem_get_info()
        _log(f"released Z-Image GPU memory (free={free/2**30:.2f}G)")
    except Exception as e:
        _log(f"release_zimage_gpu soft path failed: {e!r} -> hard reset pipe")
        _hard_reset_pipe()


def _parse_adetailer_units(req):
    """alwayson_scripts.ADetailer.args から有効ユニット(ad_model 指定あり)を抽出。
    args は [bool, {...}] か [{...}, ...] の両形式があるので dict のみ拾う。"""
    try:
        scripts = getattr(req, "alwayson_scripts", None) or {}
        if not isinstance(scripts, dict):
            scripts = _req_dict(req).get("alwayson_scripts", {}) or {}
        ad = scripts.get("ADetailer") or scripts.get("adetailer")
    except Exception:
        ad = None
    if not isinstance(ad, dict):
        return []
    args = ad.get("args")
    if not isinstance(args, list):
        return []
    units = []
    for a in args:
        if not isinstance(a, dict):
            continue  # 先頭の enable bool 等は無視
        model = a.get("ad_model", "None")
        if model and model != "None":
            units.append(a)
    return units


_ad_models_cache = None


def _ad_models():
    """ADetailer のモデル一覧(ローカル models/adetailer + HF 既定)を 1 回だけ解決。"""
    global _ad_models_cache
    if _ad_models_cache is None:
        import sys
        if ADETAILER_EXT_DIR not in sys.path:
            sys.path.insert(0, ADETAILER_EXT_DIR)
        from adetailer.common import get_models
        local = os.path.join(USERDATA_DIR, "models", "adetailer")
        _ad_models_cache = get_models(local)
    return _ad_models_cache


def _heartbeat_cb(st):
    """進捗を必ず前進させるコールバック。クライアントは『6分 progress 据え置き』で
    interrupt するため、ADetailer 等の長フェーズでも step 毎に sampling_step を進める。
    上限に達したら総数も伸ばし、progress が頭打ち(=据え置き)にならないようにする。"""
    def cb(pipe, step, timestep, cbk):
        try:
            st.sampling_step += 1
            if st.sampling_step >= st.sampling_steps:
                st.sampling_steps = st.sampling_step + 10
        except Exception:
            pass
        return cbk
    return cb


def _run_adetailer(pil_images, units, prompt, neg, base_steps, base_cfg, seed, st=None):
    """各画像に ADetailer を適用。検出/マスクは adetailer を流用、再描画は
    ZImageInpaintPipeline。ad_inpaint_only_masked 相当のクロップ inpaint+貼り戻し。"""
    import sys
    if ADETAILER_EXT_DIR not in sys.path:
        sys.path.insert(0, ADETAILER_EXT_DIR)
    from adetailer.ultralytics import ultralytics_predict
    from adetailer.mask import mask_preprocess, filter_by_ratio, filter_k_largest, sort_bboxes

    models = _ad_models()
    inpaint = _get_inpaint()
    out = []
    for img in pil_images:
        cur = img.convert("RGB")
        W, H = cur.size
        for ui, u in enumerate(units):
            name = u.get("ad_model")
            mp = models.get(name, name)
            conf = float(u.get("ad_confidence", 0.3) or 0.3)
            classes = u.get("ad_model_classes", "") or ""
            try:
                pred = ultralytics_predict(mp, cur, confidence=conf, classes=classes)
            except Exception as e:
                _log(f"adetailer[{name}] predict failed: {e!r}")
                continue
            if not pred.bboxes:
                _log(f"adetailer[{name}]: no detections")
                continue
            # 面積比フィルタ → k 件に制限 → 並べ替え(ADetailer 既定踏襲)
            pred = filter_by_ratio(
                pred,
                low=float(u.get("ad_mask_min_ratio", 0.0) or 0.0),
                high=float(u.get("ad_mask_max_ratio", 1.0) or 1.0),
            )
            k = int(u.get("ad_mask_k", u.get("ad_mask_k_largest", 0)) or 0)
            if k > 0:
                pred = filter_k_largest(pred, k)
            pred = sort_bboxes(pred)
            masks = mask_preprocess(
                pred.masks,
                kernel=int(u.get("ad_dilate_erode", 4) or 0),
                x_offset=int(u.get("ad_x_offset", 0) or 0),
                y_offset=int(u.get("ad_y_offset", 0) or 0),
                merge_invert=u.get("ad_mask_merge_invert", "None") or "None",
            )
            if not masks:
                continue

            blur = int(u.get("ad_mask_blur", 4) or 0)
            denoise = float(u.get("ad_denoising_strength", 0.4) or 0.4)
            steps = int(u.get("ad_steps", base_steps)) if u.get("ad_use_steps") else base_steps
            cfg = float(u.get("ad_cfg_scale", base_cfg)) if u.get("ad_use_cfg_scale") else base_cfg
            ad_prompt = (u.get("ad_prompt") or "").strip() or prompt
            ad_neg = (u.get("ad_negative_prompt") or "").strip() or neg
            pad = int(u.get("ad_inpaint_only_masked_padding", 32) or 0)
            use_wh = bool(u.get("ad_use_inpaint_width_height", False))
            iw = int(u.get("ad_inpaint_width", 512) or 512)
            ih = int(u.get("ad_inpaint_height", 512) or 512)
            ape, ane = _encode(inpaint, ad_prompt, ad_neg, cfg)

            n_done = 0
            for mi, mask in enumerate(masks):
                bb = mask.getbbox()
                if bb is None:
                    continue
                x1, y1, x2, y2 = bb
                cx1 = max(0, x1 - pad); cy1 = max(0, y1 - pad)
                cx2 = min(W, x2 + pad); cy2 = min(H, y2 + pad)
                cw = _align(cx2 - cx1); ch = _align(cy2 - cy1)
                cx2 = min(W, cx1 + cw); cy2 = min(H, cy1 + ch)
                if cx2 - cx1 < 16 or cy2 - cy1 < 16:
                    continue
                crop_img = cur.crop((cx1, cy1, cx2, cy2))
                crop_mask = mask.crop((cx1, cy1, cx2, cy2))
                cw, ch = crop_img.size
                # inpaint 解像度(指定 or クロップサイズを16整列)
                tw = _align(iw) if use_wh else _align(cw)
                th = _align(ih) if use_wh else _align(ch)
                in_img = crop_img.resize((tw, th), Image.LANCZOS)
                in_mask = crop_mask.resize((tw, th), Image.LANCZOS)
                if blur:
                    in_mask = in_mask.filter(ImageFilter.GaussianBlur(blur))
                g = torch.Generator(device="cpu").manual_seed(seed + 1000 * (ui + 1) + mi)
                try:
                    res = inpaint(
                        prompt_embeds=ape,
                        negative_prompt_embeds=ane,
                        image=in_img,
                        mask_image=in_mask,
                        strength=denoise,
                        width=tw,
                        height=th,
                        num_inference_steps=steps,
                        guidance_scale=cfg,
                        num_images_per_prompt=1,
                        generator=g,
                        callback_on_step_end=_heartbeat_cb(st) if st else None,
                    )
                except Exception as e:
                    _log(f"adetailer[{name}] inpaint failed: {e!r}")
                    continue
                redone = res.images[0].resize((cw, ch), Image.LANCZOS)
                # マスク領域だけ合成(継ぎ目を blur で馴染ませる)
                paste_mask = crop_mask
                if blur:
                    paste_mask = crop_mask.filter(ImageFilter.GaussianBlur(blur))
                cur.paste(redone, (cx1, cy1), mask=paste_mask)
                n_done += 1
            _log(f"adetailer[{name}]: {n_done}/{len(masks)} region(s) inpainted")
        out.append(cur)
    return out


def _generate_zimage(req):
    _free_forge_vram()
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

    # --- LoRA: <lora:name:weight> を抽出・除去し diffusers に適用 ---
    # forge ネイティブをバイパスしているため LoRA も自前で処理する。未処理だとタグが
    # テキストとしてプロンプトに混入し、かつ LoRA が一切効かない。
    prompt, loras = _extract_loras(prompt)
    neg, _ = _extract_loras(neg)
    _reset_loras(p)
    lora_names = _apply_loras(p, loras) if loras else []
    _vram("after-lora")

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
        hr_prompt = _extract_loras(getattr(req, "hr_prompt", "") or "")[0] or prompt
        hr_neg = _extract_loras(getattr(req, "hr_negative_prompt", "") or "")[0] or neg
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
    pe, ne = _encode(p, prompt, neg, cfg)
    _vram("pre-base")
    out = p(
        prompt_embeds=pe,
        negative_prompt_embeds=ne,
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
    _vram("post-base")

    if enable_hr:
        i2i = _get_img2img()
        upscaled = [im.convert("RGB").resize((tw, th), Image.LANCZOS) for im in pil_images]
        _log(f"hires: {w}x{h} -> {tw}x{th} steps={hr_steps} denoise={denoise}")
        t1 = time.time()
        gen2 = torch.Generator(device="cpu").manual_seed(seed)
        hpe, hne = _encode(i2i, hr_prompt, hr_neg, cfg)
        _vram("pre-hires")
        hr_out = i2i(
            prompt_embeds=hpe,
            negative_prompt_embeds=hne,
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

    # --- ADetailer(既定OFF: Z-Image では検出領域=ほぼ全体だとフル解像度パスをもう一回
    # 走らせる事になり極端に遅い。env ZIMAGE_ADETAILER=1 で必要時のみ有効化)。
    ad_units = _parse_adetailer_units(req) if os.environ.get("ZIMAGE_ADETAILER") == "1" else []
    ad_models_used = []
    if ad_units:
        _log(f"adetailer: {len(ad_units)} unit(s)")
        ta = time.time()
        try:
            pil_images = _run_adetailer(pil_images, ad_units, prompt, neg, steps, cfg, seed, st)
            ad_models_used = [u.get("ad_model") for u in ad_units]
        except Exception as e:
            _log(f"adetailer skipped: {e!r}")
        _log(f"adetailer done in {time.time()-ta:.0f}s")

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
    if ad_models_used:
        extra["ADetailer model"] = ", ".join(m for m in ad_models_used if m)
    if loras:
        extra["Lora"] = ", ".join(f"{n}:{w}" for n, w in loras)

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
                # 成功/失敗どちらでも diffusers の GPU 残留をここで必ず解放する。
                # finally は OOM クラッシュ時も走るので、Z-Image 終了時点で GPU を綺麗に
                # 戻せる → forge ネイティブ経路は Z-Image を一切意識しなくてよい。
                _release_zimage_gpu()
        return original(txt2imgreq, *args, **kwargs)

    # FastAPIルートと、依存解決用にendpoint属性も差し替え
    target.endpoint = wrapped
    if hasattr(target, "dependant"):
        target.dependant.call = wrapped
    _log("wrapped /sdapi/v1/txt2img (z_image -> diffusers, others -> forge)")


def on_app_started(demo, app: FastAPI):
    _wrap_txt2img(app)


script_callbacks.on_app_started(on_app_started)
