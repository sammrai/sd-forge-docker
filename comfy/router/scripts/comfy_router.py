"""ComfyUI ルーター拡張 (forge)

forge(lllyasviel backend)が判別できないアーキ — Krea 2 と Z-Image — の生成だけを
別コンテナの ComfyUI へ透過的に振り分ける。SDXL/SD1.5/Flux は一切触らず forge
ネイティブ経路をそのまま通るので、既存の生成のピクセル一致は保たれる。

振り分けキー: override_settings.sd_model_checkpoint が MODEL_SPECS の pattern に一致するか。
`/sdapi/v1/txt2img` の endpoint をラップするため、直接叩く経路と sd-queue 経由
(同じ route.endpoint を呼ぶ)の両方が自動的に対象になる。

GPU は sdui と共有しているため、Comfy を呼ぶ直前に forge のモデルを VRAM から降ろし、
生成後に Comfy 側も解放する。sd-queue のワーカーは単一スレッドなので、この順序を
守る限り両エンジンが同時に VRAM を掴むことはない。
"""
import base64
import json
import os
import re
import sys
import time
import urllib.request
import uuid

import gradio as gr
from fastapi import FastAPI

from modules import script_callbacks, shared

COMFY_URL = os.environ.get("COMFY_URL", "http://comfy:8188")
COMFY_WS = COMFY_URL.replace("http://", "ws://").replace("https://", "wss://")
# comfy の出力先は同じボリュームを共有しているので HTTP 転送せずファイルを直接読む。
# comfy: /data/comfy/output  ==  sdui: /app/data/comfy/output
COMFY_OUTPUT_DIR = os.environ.get("COMFY_OUTPUT_DIR", "/app/data/comfy/output")
# comfy の input も同じボリュームなので、HR/inpaint の入力画像はファイルを直接置く
# (multipart アップロードより速く、失敗点も減る)。
COMFY_INPUT_DIR = os.environ.get("COMFY_INPUT_DIR", "/app/data/comfy/input")
ADETAILER_EXT_DIR = os.environ.get("ADETAILER_EXT_DIR", "/app/data/extensions/adetailer")
ADETAILER_MODEL_DIR = os.environ.get("ADETAILER_MODEL_DIR", "/app/data/models/adetailer")

# 各アーキの正解値。公式ワークフローテンプレートの subgraph を展開して得たもので、
# 推測値は含まない(CLIP の type や latent ノードの種類を間違えると静かに壊れる)。
MODEL_SPECS = [
    {
        "name": "krea2",
        "pattern": re.compile(r"krea\s*-?_?2", re.IGNORECASE),
        "unet": "krea2_turbo_fp8_scaled.safetensors",
        "clip": "qwen3vl_4b_fp8_scaled.safetensors",
        "clip_type": "krea2",
        "vae": "qwen_image_vae.safetensors",
        "latent_node": "EmptyLatentImage",
        "aura_shift": None,
        "sampler": "euler",
        "scheduler": "simple",
        "steps": 8,
        "cfg": 1.0,
        # HR の 2nd pass も base と同じサンプラーを使う。公式アップスケーラの
        # テンプレートは dpmpp_2m_sde/beta だったが、それは TURBO_SAFE_SAMPLERS の
        # 外側で、8step 蒸留モデルで破綻することをこちらで実測済みの系統。
        # テンプレートの前提(step 数・モデル)が違うものをそのまま持ち込んでいた。
        "hr_sampler": "euler",
        "hr_scheduler": "simple",
        "hr_steps": 5,
        # 直接生成(346.0s)と ESRGAN 2パス(300.6s)を同一プロンプト・シードで比較し、
        # 画質に有意差が無く速度も 2パスがわずかに速かったため 2パスに統一した。
        # Z-Image と方式が揃うので spec の分岐も減る。
        "hr_mode": "esrgan_2pass",
        "hr_upscale_model": "4x-UltraSharp.pth",
        "hr_denoise": 0.33,
    },
    {
        "name": "z-image",
        "pattern": re.compile(r"z[_\- ]?image", re.IGNORECASE),
        "unet": "z_image_turbo_int8_convrot.safetensors",
        "clip": "qwen_3_4b_fp8_mixed.safetensors",
        "clip_type": "lumina2",
        "vae": "z_image_ae.safetensors",
        "latent_node": "EmptySD3LatentImage",
        "aura_shift": 3.0,
        "sampler": "res_multistep",
        "scheduler": "simple",
        "steps": 8,
        "cfg": 1.0,
        # base と同じサンプラーを使う(dpmpp_2m_sde は安全リスト外。上の krea2 参照)
        "hr_sampler": "res_multistep",
        "hr_scheduler": "simple",
        "hr_steps": 5,
        # Z-Image は直接生成すると網目状の劣化が出る(実測)。公式に 2K アップスケーラの
        # テンプレートがあるのはこのためと思われる。ESRGAN でモデル拡大してから
        # 低 denoise で焼き直す(Lanczos だとぼけた画像を渡すことになり破綻する)。
        "hr_mode": "esrgan_2pass",
        "hr_upscale_model": "4x-UltraSharp.pth",
        "hr_denoise": 0.33,
    },
]

# A1111/forge のサンプラー名 -> comfy の (sampler_name, scheduler)。
# forge 側が scheduler を別フィールドで送ってくる場合はそちらを優先する。
SAMPLER_MAP = {
    "euler": ("euler", "normal"),
    "euler a": ("euler_ancestral", "normal"),
    "heun": ("heun", "normal"),
    "lms": ("lms", "normal"),
    "ddim": ("ddim", "normal"),
    "unipc": ("uni_pc", "normal"),
    "dpm++ 2m": ("dpmpp_2m", "normal"),
    "dpm++ 2m karras": ("dpmpp_2m", "karras"),
    "dpm++ sde": ("dpmpp_sde", "normal"),
    "dpm++ sde karras": ("dpmpp_sde", "karras"),
    "dpm++ 2m sde": ("dpmpp_2m_sde", "normal"),
    "dpm++ 2m sde karras": ("dpmpp_2m_sde", "karras"),
    "dpm++ 3m sde": ("dpmpp_3m_sde", "normal"),
    "restart": ("euler", "normal"),
}
SCHEDULER_MAP = {
    "karras": "karras", "exponential": "exponential", "sgm uniform": "sgm_uniform",
    "simple": "simple", "normal": "normal", "beta": "beta", "ddim": "ddim_uniform",
    "automatic": None,
}

LORA_RE = re.compile(r"<lora:([^:>]+):([0-9.]+)>")


def _log(msg):
    print(f"[comfy-router] {msg}", flush=True)


def _req_dict(req):
    for attr in ("model_dump", "dict"):
        fn = getattr(req, attr, None)
        if fn:
            try:
                return fn()
            except Exception:
                pass
    return {k: v for k, v in vars(req).items() if not k.startswith("_")}


def _ckpt_name(req):
    try:
        return str((getattr(req, "override_settings", None) or {}).get("sd_model_checkpoint", "") or "")
    except Exception:
        return ""


def _pick_spec(req):
    """checkpoint 名から振り分け先を決める。該当なしなら None(=forge ネイティブ)。"""
    name = _ckpt_name(req)
    if not name:
        return None
    for spec in MODEL_SPECS:
        if spec["pattern"].search(name):
            return spec
    return None


def _extract_loras(text):
    """`<lora:name:weight>` を抽出し、プロンプトからは取り除く。

    forge ネイティブ経路をバイパスするので LoRA も自前で処理する。放置すると
    タグがそのままテキストとして encode され、かつ LoRA が一切効かない。
    """
    loras = [(m.group(1), float(m.group(2))) for m in LORA_RE.finditer(text or "")]
    return LORA_RE.sub("", text or "").strip(), loras


_unet_index = None


def _unet_catalog(refresh=False):
    """comfy が認識している diffusion_models を basename -> 相対パス で引けるようにする。"""
    global _unet_index
    if _unet_index is not None and not refresh:
        return _unet_index
    index = {}
    try:
        for rel in json.loads(urllib.request.urlopen(
                COMFY_URL + "/api/models/diffusion_models", timeout=60).read()):
            base = os.path.basename(rel)
            stem = base[:-len(".safetensors")] if base.endswith(".safetensors") else base
            index.setdefault(stem, rel)
            index.setdefault(base, rel)
            index.setdefault(rel, rel)
    except Exception as e:
        _log(f"WARN: cannot fetch comfy diffusion_models list: {e!r}")
    _unet_index = index
    return index


def _resolve_unet(spec, requested):
    """クライアントが指定したチェックポイント名が comfy 側に実在すればそれを使う。

    振り分けは名前のパターンマッチで行うが、`z_image_turbo_bf16.safetensors` のように
    同じアーキで別の量子化を指定したい場合がある。実在するならその指定を尊重し、
    実在しなければ spec の既定(実運用で選んだ量子化)にフォールバックする。
    """
    name = (requested or "").strip()
    if not name:
        return spec["unet"]
    cat = _unet_catalog()
    hit = cat.get(name) or cat.get(os.path.basename(name))
    if hit is None:
        hit = _unet_catalog(refresh=True).get(name)
    if hit and hit != spec["unet"]:
        _log(f"unet override: {spec['unet']} -> {hit} (クライアント指定)")
        return hit
    return spec["unet"]


_lora_index = None


def _lora_catalog(refresh=False):
    """comfy が認識している LoRA の一覧を basename -> 相対パス で引けるようにする。

    クライアントはファイル名の stem(`Foo-mid_1-vid_2`)だけを送ってくる。forge は
    models/Lora を再帰探索して解決するが、comfy は `サブディレクトリ/ファイル名` の
    相対パスを要求するため、ここで対応表を作る。
    """
    global _lora_index
    if _lora_index is not None and not refresh:
        return _lora_index
    index = {}
    try:
        names = json.loads(urllib.request.urlopen(
            COMFY_URL + "/api/models/loras", timeout=60).read())
        for rel in names:
            base = os.path.basename(rel)
            stem = base[:-len(".safetensors")] if base.endswith(".safetensors") else base
            # 同名 stem が複数階層にある場合は先勝ち(forge の探索順と揃える意図はなく、
            # 実運用では civitdl の vid 付き命名で一意になる)。
            index.setdefault(stem, rel)
            index.setdefault(base, rel)
            index.setdefault(rel, rel)
    except Exception as e:
        _log(f"WARN: cannot fetch comfy lora list: {e!r}")
    _lora_index = index
    return index


def _resolve_lora(name):
    """forge 側の LoRA 名を comfy の loras 一覧に載っている相対パスへ解決する。"""
    cat = _lora_catalog()
    hit = cat.get(name)
    if hit is None and name.endswith(".safetensors"):
        hit = cat.get(name[:-len(".safetensors")])
    if hit is None:
        cat = _lora_catalog(refresh=True)  # 直前に落としたものかもしれない
        hit = cat.get(name) or cat.get(name + ".safetensors")
    if hit is None:
        raise RuntimeError(
            f"LoRA '{name}' is not visible to comfy. "
            f"Krea2/Z-Image 用の LoRA が models/Lora 配下にあるか確認すること")
    return hit


def _align(v, m=16):
    """latent の都合で解像度を m の倍数に丸める。"""
    return max(m, int(round(float(v) / m)) * m)


# comfy 側のサンプラー/スケジューラ名。A1111 に対応する名前が無いもの
# (res_multistep など)をクライアントから指定できるよう、生の名前も受け付ける。
_comfy_samplers = None
_comfy_schedulers = None


def _comfy_sampler_names():
    global _comfy_samplers, _comfy_schedulers
    if _comfy_samplers is None:
        try:
            info = json.loads(urllib.request.urlopen(
                COMFY_URL + "/api/object_info/KSampler", timeout=60).read())
            req = info["KSampler"]["input"]["required"]
            _comfy_samplers = set(req["sampler_name"][0])
            _comfy_schedulers = set(req["scheduler"][0])
        except Exception as e:
            _log(f"WARN: cannot fetch comfy sampler list: {e!r}")
            _comfy_samplers, _comfy_schedulers = set(), set()
    return _comfy_samplers, _comfy_schedulers


def _map_sampler_pair(raw, sched_raw, def_sampler, def_scheduler):
    """A1111 のサンプラー名(+ scheduler)を comfy の組へ変換する。

    クライアントは `sampler_index: "DPM++ 2M Karras"` のように scheduler 込みの
    名前で送ってくることもあれば、`hr_sampler_name` + `hr_scheduler` のように
    分けて送ってくることもある。両方を受ける。

    加えて comfy の生のサンプラー名(`res_multistep` など)もそのまま通す。
    A1111 に対応名が無いサンプラーを指定する手段がこれしかないため。
    """
    key = str(raw or "").strip().lower()
    samplers, schedulers = _comfy_sampler_names()
    if key in samplers:
        sampler, scheduler = key, def_scheduler
    else:
        sampler, scheduler = SAMPLER_MAP.get(key, (def_sampler, def_scheduler))
    if sched_raw:
        sched_key = str(sched_raw).strip().lower()
        if sched_key in schedulers:
            scheduler = sched_key
        else:
            mapped = SCHEDULER_MAP.get(sched_key)
            if mapped:
                scheduler = mapped
    return sampler, scheduler


def _map_sampler(req, spec):
    return _map_sampler_pair(
        getattr(req, "sampler_name", None) or getattr(req, "sampler_index", None),
        getattr(req, "scheduler", None), spec["sampler"], spec["scheduler"])


# 蒸留前(RAW)バリアントの検出用。camelCase の境界に区切りを入れてから
# トークン単位で "raw" を探す。正規表現の lookaround に IGNORECASE を効かせると
# `[a-z]` が大文字にもマッチしてしまい `krea2RawInt8` を取りこぼす。
# RAW に turbo 向けの値が来たと判断する閾値。RAW の適正は目安 20〜30 step /
# cfg 3.5〜5 なので、そこから明確に外れた値だけを弾く(正常な RAW 利用を巻き込まない)。
RAW_MIN_STEPS = 10
RAW_MIN_CFG = 1.5

_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SPLIT_RE = re.compile(r"[^A-Za-z0-9]+")


def _is_raw_variant(unet_name):
    """蒸留前の RAW モデルかどうか。

    Civitai の画像が申告する Checkpoint リソースが RAW バリアントを指していることが
    あり(モデル名に "Turbo" と入っていてもバージョンが raw のことがある)、クライアントは
    それをそのまま civitdl する。RAW を turbo 前提の steps=8 / cfg=1 で回すと絵にならない。
    例外も出ず parameters と info も整合するのでサイレントに壊れる。ここで検出して
    steps/cfg の矯正を止める。
    """
    base = os.path.basename(unet_name or "")
    tokens = _SPLIT_RE.split(_CAMEL_RE.sub("_", base).lower())
    return "raw" in tokens


def _normalize_sampling(req, spec, is_raw=False):
    """turbo(蒸留)モデル向けに steps/cfg を正規化する。

    クライアントは SDXL 向けの既定値(steps=20, cfg_scale=7)を送ってくるが、
    Krea2 Turbo / Z-Image Turbo は cfg=1 で蒸留されており、cfg>1 では絵が破綻する。
    steps も 8 前後を前提に蒸留されている。ここで spec 値へ寄せ、黙って値を
    書き換えたことがログに残るようにする(info にも実際に使った値を返す)。
    """
    req_steps = int(getattr(req, "steps", 0) or 0)
    req_cfg = float(getattr(req, "cfg_scale", 0) or 0)
    if is_raw:
        # RAW は蒸留を経ていないので turbo 用の spec 値を当ててはいけない。
        # 送信値をそのまま使う(クライアントが RAW 向けの値を送る前提)。
        steps, cfg = (req_steps or spec["steps"]), (req_cfg or spec["cfg"])
        # ただし turbo 向けの値がそのまま来た場合は生成しても絵にならない。
        # クライアントは Civitai 画像のメタから steps/cfg を取るため、turbo 画像が
        # RAW バリアントを申告していると必ずこの組み合わせになる。矯正しても
        # 送信値を使っても壊れるので、**黙って壊れた画像を返さずエラーで落とす**。
        # 未送信(0)は判定対象外。送られた値だけを見る。
        if (req_steps and req_steps <= RAW_MIN_STEPS) or \
                (req_cfg and req_cfg <= RAW_MIN_CFG):
            raise RuntimeError(
                f"RAW variant '{spec['name']}' に turbo 向けのパラメータが指定されました "
                f"(steps={req_steps}, cfg_scale={req_cfg})。"
                f"RAW は蒸留前のモデルで {RAW_MIN_STEPS + 1} step / cfg {RAW_MIN_CFG} 超"
                f"(目安 20〜30 step / cfg 3.5〜5)を想定しています。"
                f"この値のままでは絵になりません。turbo バリアントの checkpoint を"
                f"指定するか、RAW 向けの steps / cfg_scale を指定してください。")
        _log(f"RAW variant detected: steps/cfg は矯正せず送信値を使う "
             f"(steps={steps} cfg={cfg})")
        return steps, cfg
    # 送信値をそのまま使う。値の決定はクライアントの責任で、ルーターは書き換えない
    # (書き換えると EXIF に送っていない値が残り、クライアントから制御もできない)。
    steps = req_steps or spec["steps"]
    cfg = req_cfg if req_cfg else spec["cfg"]
    if abs(cfg - spec["cfg"]) > 1e-6:
        _log(f"WARN: cfg={cfg} は cfg={spec['cfg']} で蒸留されたモデルには不適合。"
             f"送信値のまま使用する。")
    return steps, cfg


# 8step 蒸留モデルで実用になることを確認したサンプラー。ここから外れると
# 顔が潰れるなどの破綻が起きる(dpmpp_2m+karras で目が黒い塊になるのを実測)。
TURBO_SAFE_SAMPLERS = {"euler", "euler_ancestral", "res_multistep", "er_sde", "lcm"}


def _normalize_sampler(sampler, scheduler, spec):
    """蒸留モデルに不適合なサンプラーを spec 値へ矯正する。

    クライアントは SDXL 向けに `sampler_index: "DPM++ 2M Karras"` を送ってくるが、
    8step 蒸留モデルではこれで顔が完全に潰れる。steps / cfg と同様、互換性は
    こちら側で吸収する(クライアントに分岐を実装させない)。
    turbo で実用になることを確認済みのサンプラーが明示された場合はそれを尊重する。
    """
    if sampler not in TURBO_SAFE_SAMPLERS:
        _log(f"WARN: sampler {sampler}/{scheduler} は 8step 蒸留モデルで破綻する "
             f"(安全リスト: {', '.join(sorted(TURBO_SAFE_SAMPLERS))})。"
             f"送信値のまま使用する。")
    return sampler, scheduler


def _warn_unsupported(req, spec):
    """対応していない指定をサイレントに落とさず、ログに出す。"""
    ov = getattr(req, "override_settings", None) or {}
    if ov.get("sd_vae"):
        _log(f"NOTE: override_settings.sd_vae={ov['sd_vae']!r} は無視される "
             f"(comfy 側は {spec['vae']} 固定)")
    if bool(getattr(req, "enable_hr", False)):
        unused = [k for k in ("hr_upscaler", "hr_second_pass_steps", "hr_sampler_name",
                              "hr_scheduler", "denoising_strength")
                  if getattr(req, k, None) not in (None, "", 0)]
        if unused:
            _log(f"NOTE: {', '.join(unused)} は使われない "
                 f"(HR の 2nd パス設定は spec の内部固定値を使うため)")
    if getattr(req, "hr_cfg", None) not in (None, "") and float(getattr(req, "hr_cfg", 1) or 1) != 1.0:
        _log(f"NOTE: hr_cfg={getattr(req, 'hr_cfg')} は無視される "
             f"(蒸留モデルは cfg=1 前提)")
    for unit in _adetailer_units(req):
        if unit.get("ad_controlnet_model") not in (None, "None", ""):
            _log("NOTE: ad_controlnet_* は Krea2/Z-Image 用 ControlNet が無いため無視")


def _wait_forge_idle(timeout=600.0):
    """forge ネイティブの生成が走っていないことを確認してから進む。

    sd-queue 経由なら単一ワーカーが直列化するので本来ここは素通りする。
    ガードが要るのは `/sdapi/v1/txt2img` を直接叩く経路で、これはキューを迂回して
    並走するため、生成中に VRAM を剥がすと相手のジョブが極端に遅くなる
    (実測 0.68s/it -> 3.21s/it)。最悪 OOM で落とす可能性もある。
    """
    st = shared.state
    t0 = time.time()
    warned = False
    while getattr(st, "job", "") and time.time() - t0 < timeout:
        if not warned:
            _log(f"waiting for forge job to finish: {st.job!r} "
                 f"({st.sampling_step}/{st.sampling_steps})")
            warned = True
        time.sleep(0.5)
    if warned:
        _log(f"forge idle after {time.time() - t0:.1f}s")


def _free_forge_vram():
    """forge のモデルを VRAM から降ろす。CPU RAM には残るので次のジョブでは
    RAM->VRAM の転送だけで復帰する(ディスク再読み込みは発生しない)。"""
    try:
        from backend import memory_management
        memory_management.unload_all_models()
        memory_management.soft_empty_cache(force=True)
        _log("forge VRAM released")
    except Exception as e:
        _log(f"WARN: forge VRAM release failed: {e!r}")


def _free_comfy_vram():
    """comfy は生成後もモデルを VRAM に保持し続けるため、明示的に解放させる。
    これを怠ると次の forge ジョブがロードに失敗する(実測 8GB 保持)。"""
    try:
        req = urllib.request.Request(
            COMFY_URL + "/api/free",
            data=json.dumps({"unload_models": True, "free_memory": True}).encode(),
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=120).read()
        _log("comfy VRAM released")
    except Exception as e:
        _log(f"WARN: comfy VRAM release failed: {e!r}")


def _build_txt2img(spec, prompt, neg, w, h, steps, cfg, seed, batch,
                   sampler, scheduler, loras, hr2pass=None, unet=None):
    """指定解像度で生成する。

    hr2pass が指定された場合は公式 2K アップスケーラ相当の 2nd パスを同じグラフに
    続ける: ESRGAN でモデル拡大 -> 目標サイズへ縮小 -> VAEEncode -> 低 denoise で焼き直し。
    ピクセル拡大に Lanczos を使うと情報の無いぼけた画像を渡すことになり破綻するので、
    必ずモデル拡大を経由させる。
    """
    g, model_out = _loaders(spec, prompt, neg, cfg, loras, unet=unet)
    g["5"] = {"class_type": spec["latent_node"],
              "inputs": {"width": w, "height": h, "batch_size": batch}}
    g["3"] = {"class_type": "KSampler",
              "inputs": {"model": model_out, "seed": seed, "steps": steps, "cfg": cfg,
                         "sampler_name": sampler, "scheduler": scheduler,
                         "positive": ["6", 0], "negative": ["13", 0],
                         "latent_image": ["5", 0], "denoise": 1.0}}
    g["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["12", 0]}}
    image_out = ["8", 0]

    if hr2pass:
        g["40"] = {"class_type": "UpscaleModelLoader",
                   "inputs": {"model_name": hr2pass["model"]}}
        g["41"] = {"class_type": "ImageUpscaleWithModel",
                   "inputs": {"upscale_model": ["40", 0], "image": ["8", 0]}}
        # ESRGAN は 4x なので、目標倍率へ縮小して合わせる
        g["42"] = {"class_type": "ImageScaleBy",
                   "inputs": {"image": ["41", 0], "upscale_method": "lanczos",
                              "scale_by": hr2pass["shrink"]}}
        g["43"] = {"class_type": "VAEEncode",
                   "inputs": {"pixels": ["42", 0], "vae": ["12", 0]}}
        g["44"] = {"class_type": "KSampler",
                   "inputs": {"model": model_out, "seed": seed,
                              "steps": hr2pass["steps"], "cfg": cfg,
                              "sampler_name": spec["hr_sampler"],
                              "scheduler": spec["hr_scheduler"],
                              "positive": ["6", 0], "negative": ["13", 0],
                              "latent_image": ["43", 0],
                              "denoise": hr2pass["denoise"]}}
        g["45"] = {"class_type": "VAEDecode",
                   "inputs": {"samples": ["44", 0], "vae": ["12", 0]}}
        image_out = ["45", 0]

    g["9"] = {"class_type": "SaveImage",
              "inputs": {"images": image_out, "filename_prefix": "router"}}
    return g


def _model_hash(name):
    """forge と同じ方法で checkpoint の shorthash を出す。

    ルーター経由のモデルは forge が一度もロードしないため、forge 側では
    hash が付かず info.sd_model_hash が null のままになる(213 モデル中
    189 に hash があり、comfy 行きのものだけ全滅していた)。
    forge の hashes.sha256 をそのまま使うので、キャッシュも
    /sdapi/v1/sd-models への反映もネイティブと同じ経路に乗る。
    初回だけファイル全体を読むので HDD では 1 分以上かかる。
    """
    if not name:
        return None
    try:
        from modules import sd_models, hashes
        ci = sd_models.get_closet_checkpoint_match(name)
        if ci is None:
            return None
        sha = hashes.sha256(ci.filename, f"checkpoint/{ci.name}")
        return sha[0:10] if sha else None
    except Exception as e:
        _log(f"WARN: model hash failed for {name!r}: {e}")
        return None


class _Interrupted(Exception):
    """クライアントが /sdapi/v1/interrupt を叩いた。"""


def _cancel_comfy(pid):
    """走っている prompt を止め、キュー待ちなら取り下げる。

    /api/interrupt は「いま実行中のもの」しか止めないので、キューに積まれた
    ぶんは /api/queue の delete で消す必要がある。両方投げる。
    """
    for path, body in (("/api/interrupt", {}), ("/api/queue", {"delete": [pid]})):
        try:
            urllib.request.urlopen(urllib.request.Request(
                COMFY_URL + path, data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"}), timeout=15).read()
        except Exception as e:
            _log(f"WARN: {path} failed: {e}")


def _submit_and_wait(graph, st, step_offset=0, node_offsets=None):
    """comfy にワークフローを投げ、ws で進捗を受けながら完了まで待つ。

    ws を先に張ってから /api/prompt を投げる(逆順だと最初の progress を取りこぼす)。
    受け取った step は shared.state に流し込むので、forge の /sdapi/v1/progress と
    sd-queue の task status がそのまま動く。
    """
    from websockets.sync.client import connect

    client_id = str(uuid.uuid4())
    with connect(f"{COMFY_WS}/ws?clientId={client_id}", open_timeout=30,
                 max_size=64 * 1024 * 1024) as ws:
        req = urllib.request.Request(
            COMFY_URL + "/api/prompt",
            data=json.dumps({"prompt": graph, "client_id": client_id}).encode(),
            headers={"Content-Type": "application/json"})
        try:
            resp = json.loads(urllib.request.urlopen(req, timeout=60).read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:1500]
            raise RuntimeError(f"comfy rejected the workflow: {detail}") from None
        pid = resp["prompt_id"]

        last_value, last_max, inner_offset = 0, 0, 0
        last_msg = time.time()
        while True:
            # forge の中断フラグを見る。recv を 600s でブロックすると中断に
            # 気づけないので、短い timeout で回して毎周チェックする。
            if getattr(st, "interrupted", False) or getattr(st, "skipped", False):
                _log("interrupt requested -> cancelling comfy job")
                _cancel_comfy(pid)
                raise _Interrupted()
            try:
                raw = ws.recv(timeout=1.0)
            except TimeoutError:
                if time.time() - last_msg > 600:
                    _cancel_comfy(pid)
                    raise RuntimeError("comfy did not respond within 600s")
                continue
            last_msg = time.time()
            if isinstance(raw, (bytes, bytearray)):
                continue  # プレビュー画像のバイナリフレームは使わない
            msg = json.loads(raw)
            mtype, data = msg.get("type"), msg.get("data", {})
            if mtype == "progress" and data.get("max"):
                try:
                    # 1ジョブ内で base -> HR -> ADetailer と複数回 comfy を呼ぶため、
                    # 各フェーズの step を通し番号に積み上げてクライアントへ見せる。
                    v, mx = int(data["value"]), int(data["max"])
                    # comfy は KSampler 以外(ESRGAN のタイル処理など)も進捗を送る。
                    # node id で対象を絞らないと別ノードの 0..N が混ざり、進捗が
                    # 巻き戻って見える(実測: 1.0 -> 0.72 -> 1.0 -> 0.79)。
                    node = str(data.get("node") or "")
                    if node_offsets is not None:
                        if node not in node_offsets:
                            continue
                        inner_offset = node_offsets[node]
                    elif v < last_value:
                        inner_offset += last_max
                    last_value, last_max = v, mx
                    cur = step_offset + inner_offset + v
                    st.sampling_step = cur
                    # ADetailer の検出数は事前に読めないため見積もりを超えることがある。
                    # 超えたら上限を押し上げ、progress が 1.0 を超えないようにする。
                    if cur > getattr(st, "sampling_steps", 0):
                        st.sampling_steps = cur + mx
                except Exception:
                    pass
            elif mtype == "execution_error" and data.get("prompt_id") == pid:
                raise RuntimeError(f"comfy execution error: {json.dumps(data)[:1200]}")
            elif mtype == "executing" and data.get("prompt_id") == pid \
                    and data.get("node") is None:
                break  # node=None はこの prompt の実行完了を意味する

    hist = json.loads(urllib.request.urlopen(
        COMFY_URL + f"/api/history/{pid}", timeout=60).read())
    if pid not in hist:
        raise RuntimeError("comfy finished but produced no history entry")
    return hist[pid]


def _open_outputs(hist_entry):
    """SaveImage が書いた PNG を共有ボリュームから PIL で開く。"""
    from PIL import Image
    out = []
    for _, node_out in (hist_entry.get("outputs") or {}).items():
        for im in node_out.get("images", []):
            if im.get("type") not in (None, "output"):
                continue
            path = os.path.join(COMFY_OUTPUT_DIR, im.get("subfolder") or "", im["filename"])
            out.append(Image.open(path).convert("RGB"))
    if not out:
        raise RuntimeError("comfy produced no readable images")
    return out


def _put_comfy_input(pil, tag):
    """comfy の input ディレクトリへ直接書く(同じボリュームなのでアップロード不要)。"""
    name = f"router_{tag}_{uuid.uuid4().hex[:12]}.png"
    os.makedirs(COMFY_INPUT_DIR, exist_ok=True)
    pil.save(os.path.join(COMFY_INPUT_DIR, name))
    return name


def _loaders(spec, prompt, neg, cfg, loras, unet=None):
    """ローダ + 条件付け + LoRA + ModelSampling までの共通部分。"""
    g = {
        "10": {"class_type": "UNETLoader",
               "inputs": {"unet_name": unet or spec["unet"],
                          "weight_dtype": "default"}},
        "11": {"class_type": "CLIPLoader",
               "inputs": {"clip_name": spec["clip"], "type": spec["clip_type"]}},
        "12": {"class_type": "VAELoader", "inputs": {"vae_name": spec["vae"]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["11", 0]}},
    }
    if cfg > 1.0 and neg:
        g["13"] = {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["11", 0]}}
    else:
        g["13"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["6", 0]}}
    model_out = ["10", 0]
    for i, (name, weight) in enumerate(loras):
        nid = f"lora{i}"
        g[nid] = {"class_type": "LoraLoaderModelOnly",
                  "inputs": {"model": model_out, "lora_name": _resolve_lora(name),
                             "strength_model": weight}}
        model_out = [nid, 0]
    if spec["aura_shift"] is not None:
        g["14"] = {"class_type": "ModelSamplingAuraFlow",
                   "inputs": {"model": model_out, "shift": spec["aura_shift"]}}
        model_out = ["14", 0]
    return g, model_out


def _build_inpaint(spec, image_name, mask_name, prompt, neg, steps, cfg, seed,
                   denoise, sampler, scheduler, loras, unet=None):
    """ADetailer の inpaint。クロップとマスク生成は forge 側で済ませてある前提。"""
    g, model_out = _loaders(spec, prompt, neg, cfg, loras, unet=unet)
    g["20"] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
    g["22"] = {"class_type": "LoadImageMask", "inputs": {"image": mask_name, "channel": "red"}}
    g["21"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["20", 0], "vae": ["12", 0]}}
    g["23"] = {"class_type": "SetLatentNoiseMask",
               "inputs": {"samples": ["21", 0], "mask": ["22", 0]}}
    g["3"] = {"class_type": "KSampler",
              "inputs": {"model": model_out, "seed": seed, "steps": steps, "cfg": cfg,
                         "sampler_name": sampler, "scheduler": scheduler,
                         "positive": ["6", 0], "negative": ["13", 0],
                         "latent_image": ["23", 0], "denoise": denoise}}
    g["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["12", 0]}}
    g["9"] = {"class_type": "SaveImage",
              "inputs": {"images": ["8", 0], "filename_prefix": "router_ad"}}
    return g


# ---------------------------------------------------------------- HR (hires fix)

def _hr_params(req, spec, base_w, base_h, base_steps):
    """enable_hr を目標解像度へ翻訳する。実際の実現方式は spec の hr_mode 次第。

    A1111 の hires fix は「SDXL は 1Mpx を超えると構図が破綻する」ことへの回避策で、
    低解像度で構図を作ってから拡大・焼き直す。

    **現在の方式は MODEL_SPECS の hr_mode を見ること(Krea2 / Z-Image とも
    esrgan_2pass)。** 下の実測は ESRGAN 2パス導入**前**の比較で、当時は
    「直接生成のみ成功」という結論だった。

      直接生成                     289.6s  破綻なし
      ピクセル2パス(Lanczos+VAE往復) 245s   鱗状の反復パターン
      latent 2パス(bislerp)         244s   ぼやけ + 網目テクスチャ

    その後、拡大をモデル拡大(ESRGAN)に替えると 2パスが成立すると分かり、
    Krea2 は直接生成(346.0s)と ESRGAN 2パス(300.6s)を比較して画質に有意差が無く
    速度も 2パスが速かったため esrgan_2pass に統一した。上表の「2パスは破綻する」は
    Lanczos / bislerp 拡大に限った話で、ESRGAN 拡大には当てはまらない。

    denoising_strength / hr_upscaler / hr_second_pass_steps は、クライアントから
    合わせにいけない内部固定値なので使わない(値は parameters にエコーバックする)。
    """
    if not bool(getattr(req, "enable_hr", False)):
        return None
    rx = int(getattr(req, "hr_resize_x", 0) or 0)
    ry = int(getattr(req, "hr_resize_y", 0) or 0)
    if rx > 0 and ry > 0:
        tw, th = _align(rx), _align(ry)
    else:
        scale = float(getattr(req, "hr_scale", 2.0) or 2.0)
        tw, th = _align(base_w * scale), _align(base_h * scale)
    return {"w": tw, "h": th,
            "scale": float(getattr(req, "hr_scale", 2.0) or 2.0)}


# ---------------------------------------------------------------- ADetailer

def _adetailer_units(req):
    """alwayson_scripts.ADetailer.args から有効ユニットを取り出す。

    先頭の bool(ad_enable, skip_img2img)は forge の位置引数なので読み飛ばす。
    クライアントは normalize_adetailer_args() で必ず [True, False, unit...] の形に
    揃えて送ってくる。
    """
    try:
        scripts = getattr(req, "alwayson_scripts", None) or {}
        if not isinstance(scripts, dict):
            scripts = _req_dict(req).get("alwayson_scripts", {}) or {}
        args = ((scripts.get("ADetailer") or {}).get("args")) or []
    except Exception:
        return []
    units = []
    for a in args:
        if not isinstance(a, dict):
            continue  # 先頭の bool
        model = a.get("ad_model", "None")
        if model and model != "None":
            units.append(a)
    return units


_ad_models_cache = None


def _ad_models():
    global _ad_models_cache
    if _ad_models_cache is None:
        if ADETAILER_EXT_DIR not in sys.path:
            sys.path.insert(0, ADETAILER_EXT_DIR)
        from adetailer.common import get_models
        _ad_models_cache = get_models(ADETAILER_MODEL_DIR)
    return _ad_models_cache


def _detect(unit, pil):
    """ADetailer 本体のコードで検出とマスク整形を行う。

    YOLO は CPU で回す。comfy が VRAM を握っている状態で GPU 推論を挟むと
    12GB では詰むため(設計判断として CPU 固定)。検出モデルは数十MB なので
    全体時間から見れば誤差。
    """
    if ADETAILER_EXT_DIR not in sys.path:
        sys.path.insert(0, ADETAILER_EXT_DIR)
    from adetailer import mask as admask
    from adetailer.ultralytics import ultralytics_predict

    models = _ad_models()
    name = unit.get("ad_model")
    path = models.get(name)
    if path is None:
        _log(f"WARN: ADetailer model {name!r} not found; unit skipped")
        return []

    pred = ultralytics_predict(
        path, pil,
        confidence=float(unit.get("ad_confidence", 0.3) or 0.3),
        device="cpu",
        classes=unit.get("ad_model_classes", "") or "")
    if not pred.masks:
        return []

    pred = admask.filter_by_ratio(
        pred,
        low=float(unit.get("ad_mask_min_ratio", 0.0) or 0.0),
        high=float(unit.get("ad_mask_max_ratio", 1.0) or 1.0))
    k = int(unit.get("ad_mask_k", unit.get("ad_mask_k_largest", 0)) or 0)
    if k > 0:
        method = str(unit.get("ad_mask_filter_method", "Area") or "Area").lower()
        if method.startswith("conf"):
            pred = admask.filter_k_most_confident(pred, k)
        else:
            pred = admask.filter_k_largest(pred, k)
    if not pred.masks:
        return []
    # 本物の ADetailer は filter のあとに並べ替える(pred_preprocessing)。
    # 順序はユニットごとの seed(seed + i)に効くので、揃えないと再現しない。
    try:
        from adetailer.args import BBOX_SORTBY
        pred = admask.sort_bboxes(pred, BBOX_SORTBY.index(BBOX_SORTBY[0]))
    except Exception as e:
        _log(f"WARN: bbox sort skipped ({e})")
    return admask.mask_preprocess(
        pred.masks,
        kernel=int(unit.get("ad_dilate_erode", 4) or 0),
        x_offset=int(unit.get("ad_x_offset", 0) or 0),
        y_offset=int(unit.get("ad_y_offset", 0) or 0),
        merge_invert=unit.get("ad_mask_merge_invert", "None") or "None")


def _expand_crop_region(crop_region, pw, ph, iw, ih):
    """クロップ枠を処理解像度のアスペクトに合わせて広げる。

    forge modules/masking.py の expand_crop_region と同じ計算。これをやらずに
    クロップをそのまま pw x ph に伸ばすと、貼り戻したときに顔が縦横に歪む。
    """
    x1, y1, x2, y2 = crop_region
    if (x2 - x1) / (y2 - y1) > pw / ph:
        diff = int((x2 - x1) / (pw / ph) - (y2 - y1))
        y1 -= diff // 2
        y2 += diff - diff // 2
        if y2 >= ih:
            y1 -= y2 - ih
            y2 = ih
        if y1 < 0:
            y2 -= y1
            y1 = 0
        y2 = min(y2, ih)
    else:
        diff = int((y2 - y1) * (pw / ph) - (x2 - x1))
        x1 -= diff // 2
        x2 += diff - diff // 2
        if x2 >= iw:
            x1 -= x2 - iw
            x2 = iw
        if x1 < 0:
            x2 -= x1
            x1 = 0
        x2 = min(x2, iw)
    return max(0, x1), max(0, y1), x2, y2


def _run_adetailer_unit(spec, pil, unit, prompt, neg, cfg, seed, loras, st, offset,
                        unet=None, pw=None, ph=None):
    """1ユニット分の ADetailer。検出→クロップ→comfy で inpaint→貼り戻し。

    ad_inpaint_only_masked 相当のクロップ処理を forge 側で行うので、comfy 側は
    「小さい画像 + マスク」を受け取るだけで済み、Impact Pack 等の追加ノードが要らない。
    """
    from PIL import Image, ImageFilter

    masks = _detect(unit, pil)
    if not masks:
        _log(f"ADetailer[{unit.get('ad_model')}]: no detection")
        return pil, 0

    blur = int(unit.get("ad_mask_blur", 4) or 0)
    denoise = float(unit.get("ad_denoising_strength", 0.4) or 0.4)
    steps = int(unit.get("ad_steps", 0) or 0) if unit.get("ad_use_steps") else 0
    steps = steps or spec["steps"]
    ad_cfg = float(unit.get("ad_cfg_scale", 0) or 0) if unit.get("ad_use_cfg_scale") else 0
    ad_cfg = ad_cfg or cfg
    ad_prompt = (unit.get("ad_prompt") or "").strip() or prompt
    ad_neg = (unit.get("ad_negative_prompt") or "").strip() or neg
    only_masked = bool(unit.get("ad_inpaint_only_masked", True))
    pad = int(unit.get("ad_inpaint_only_masked_padding", 32) or 0)

    # inpaint 解像度。A1111 は ad_use_inpaint_width_height が偽なら
    # **元の生成解像度**(p.width / p.height)で描き直す
    # (adetailer scripts/!adetailer.py の get_width_height)。
    # クロップ実寸で描くと Krea2/Z-Image では学習分布外の小解像度になり顔が崩れる。
    if unit.get("ad_use_inpaint_width_height"):
        iw = _align(int(unit.get("ad_inpaint_width", 512) or 512))
        ih = _align(int(unit.get("ad_inpaint_height", 512) or 512))
    else:
        iw = _align(int(pw or pil.width))
        ih = _align(int(ph or pil.height))

    result = pil.copy()
    done = 0
    for i, m in enumerate(masks):
        m = m.convert("L")
        if only_masked:
            box = m.getbbox()
            if box is None:
                continue
            x0, y0, x1, y1 = box
            x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
            x1, y1 = min(result.width, x1 + pad), min(result.height, y1 + pad)
            x0, y0, x1, y1 = _expand_crop_region((x0, y0, x1, y1), iw, ih,
                                                 result.width, result.height)
        else:
            x0, y0, x1, y1 = 0, 0, result.width, result.height
        if x1 - x0 < 16 or y1 - y0 < 16:
            continue

        crop = result.crop((x0, y0, x1, y1))
        cmask = m.crop((x0, y0, x1, y1))
        work = crop.resize((iw, ih), Image.LANCZOS)
        wmask = cmask.resize((iw, ih), Image.LANCZOS)
        if blur > 0:
            wmask = wmask.filter(ImageFilter.GaussianBlur(blur))

        img_name = _put_comfy_input(work, "adimg")
        msk_name = _put_comfy_input(wmask.convert("RGB"), "admask")
        graph = _build_inpaint(spec, img_name, msk_name, ad_prompt, ad_neg, steps,
                               ad_cfg, seed + i, denoise, spec["sampler"],
                               spec["scheduler"], loras, unet=unet)
        out = _open_outputs(_submit_and_wait(graph, st, step_offset=offset,
                                             node_offsets={"3": 0}))[0]
        offset += steps

        patched = out.resize((crop.width, crop.height), Image.LANCZOS)
        paste_mask = cmask.filter(ImageFilter.GaussianBlur(blur)) if blur > 0 else cmask
        region = result.crop((x0, y0, x1, y1))
        region = Image.composite(patched, region, paste_mask)
        result.paste(region, (x0, y0))
        done += 1

    _log(f"ADetailer[{unit.get('ad_model')}]: {done}/{len(masks)} region(s) inpainted")
    return result, done


def _make_infotext(spec, prompt, neg, steps, sampler, scheduler, cfg, seed, w, h,
                   model, loras):
    parts = [
        f"Steps: {steps}", f"Sampler: {sampler}", f"Schedule type: {scheduler}",
        f"CFG scale: {cfg}", f"Seed: {seed}", f"Size: {w}x{h}",
        f"Model: {model}", f"Backend: comfy/{spec['name']}",
    ]
    if loras:
        parts.append("Lora: " + ", ".join(f"{n}:{w_}" for n, w_ in loras))
    head = prompt
    if neg:
        head += f"\nNegative prompt: {neg}"
    return head + "\n" + ", ".join(parts)


def _generate(req):
    """base -> HR -> ADetailer を1ジョブとして実行する。

    フェーズ間で comfy の VRAM は解放しない(同じモデルを使い回す)。解放は
    ラッパーの finally で1回だけ行う。
    """
    import base64 as _b64
    import io as _io

    spec = _pick_spec(req)
    prompt, loras = _extract_loras(getattr(req, "prompt", "") or "")
    neg, _ = _extract_loras(getattr(req, "negative_prompt", "") or "")

    model = _ckpt_name(req)
    # 振り分けは名前のパターンで行うが、その名前が comfy に実在するなら
    # 量子化違いの指定として尊重する(例: z_image_turbo_bf16 を明示)。
    unet = _resolve_unet(spec, model)
    # RAW 判定は実際にロードするファイル名で行う。振り分けキー(checkpoint 名)は
    # モデル名由来で "Turbo" と入っていても中身が RAW のことがあるため。
    is_raw = _is_raw_variant(unet)

    steps, cfg = _normalize_sampling(req, spec, is_raw=is_raw)
    # 16px 丸めは latent の実装都合であってクライアントが知るべきことではない。
    # 丸めた解像度で生成し、最後に要求解像度へ戻す(I/F として入出力を整合させる)。
    req_w = int(getattr(req, "width", 1024) or 1024)
    req_h = int(getattr(req, "height", 1024) or 1024)
    w, h = _align(req_w), _align(req_h)
    batch = int(getattr(req, "batch_size", 1) or 1)
    n_iter = int(getattr(req, "n_iter", 1) or 1)
    if n_iter > 1:
        raise RuntimeError(
            f"n_iter={n_iter} は未対応です(ルーターは 1 バッチしか実行しません)。"
            f"黙って枚数が減るのを避けるためエラーにしています。"
            f"batch_size で指定するか、リクエストを分けてください。")
    seed = int(getattr(req, "seed", -1) or -1)
    if seed < 0:
        seed = int.from_bytes(os.urandom(4), "big")
    sampler, scheduler = _map_sampler(req, spec)

    _warn_unsupported(req, spec)
    sampler, scheduler = _normalize_sampler(sampler, scheduler, spec)
    hr = _hr_params(req, spec, w, h, steps)
    units = _adetailer_units(req)

    # 進捗の総ステップ数。ADetailer は検出数が事前に分からないので 1領域/ユニットで
    # 見積もっておき、実際に増えたぶんは _submit_and_wait 側で max を押し上げる。
    total = steps + sum(spec["steps"] for _ in units)
    if hr and spec.get("hr_mode") == "esrgan_2pass":
        total += int(spec["hr_steps"] * spec["hr_denoise"]) + 1

    _wait_forge_idle()

    st = shared.state
    try:
        st.begin(f"comfy-{spec['name']}")
        st.job_count = 1
        st.job_no = 0
        st.sampling_steps = max(1, total)
        st.sampling_step = 0
    except Exception:
        pass

    _log(f"route -> {spec['name']} | {w}x{h} steps={steps} cfg={cfg} "
         f"sampler={sampler}/{scheduler} seed={seed} loras={len(loras)} "
         f"hr={'on' if hr else 'off'} adetailer={len(units)}")
    _free_forge_vram()

    t0 = time.time()
    offset = 0

    # --- 生成 ---
    # HR の実現方法はモデルごとに違う(実測で決めた。詳細は _hr_params の docstring)。
    #   direct       : 目標解像度で1パス生成
    #   esrgan_2pass : base 生成 -> ESRGAN 拡大 -> 低 denoise で焼き直し
    # 現状は Krea2 / Z-Image とも esrgan_2pass。Krea2 は direct でも破綻しないが、
    # 同一シードで比較して画質に有意差が無く速度も 2パスがわずかに速かったため統一した。
    gw, gh = w, h
    hr2pass = None
    if hr:
        if spec.get("hr_mode") == "esrgan_2pass":
            # ESRGAN は 4x 固定なので、目標倍率になるよう縮小率を逆算する
            hr2pass = {"model": spec["hr_upscale_model"],
                       "shrink": hr["scale"] / 4.0,
                       "denoise": spec["hr_denoise"],
                       "steps": spec["hr_steps"]}
        else:
            gw, gh = hr["w"], hr["h"]

    graph = _build_txt2img(spec, prompt, neg, gw, gh, steps, cfg, seed, batch,
                           sampler, scheduler, loras, hr2pass=hr2pass, unet=unet)
    # 進捗はサンプラーノードだけを対象にする("3"=base, "44"=HR の 2nd パス)
    node_offsets = {"3": 0}
    if hr2pass:
        node_offsets["44"] = steps
    images = _open_outputs(_submit_and_wait(graph, st, step_offset=offset,
                                            node_offsets=node_offsets))
    offset += steps + (int(hr2pass["steps"] * hr2pass["denoise"]) + 1 if hr2pass else 0)
    # 実出力の解像度(HR 有効時は 2 パス目の結果)。ログにのみ使う。
    # parameters / info / infotext は入力解像度(gw, gh)を返す(A1111 準拠)。
    out_w, out_h = images[0].width, images[0].height
    mode = "direct" if not hr2pass else f"esrgan2pass({hr2pass['model']})"
    _log(f"  generate {gw}x{gh} -> {out_w}x{out_h} hr={mode} "
         f"done {time.time() - t0:.1f}s ({len(images)} image(s))")

    # --- ADetailer ---
    ad_used = []
    for unit in units:
        t2 = time.time()
        new_images = []
        for im in images:
            out, done = _run_adetailer_unit(spec, im, unit, prompt, neg, cfg, seed,
                                            loras, st, offset, unet=unet,
                                            pw=gw, ph=gh)
            new_images.append(out)
            offset += spec["steps"] * max(1, done)
        images = new_images
        ad_used.append(unit.get("ad_model"))
        _log(f"  adetailer {unit.get('ad_model')} done {time.time() - t2:.1f}s")

    # PIL -> base64(PNG)
    b64 = []
    for im in images:
        buf = _io.BytesIO()
        im.save(buf, format="PNG")
        b64.append(_b64.b64encode(buf.getvalue()).decode())

    _log(f"done in {time.time() - t0:.1f}s ({len(b64)} image(s))")

    # 丸めて生成したぶんを要求解像度の比率へ戻す。HR 有効時は倍率が乗ったまま。
    if (w, h) != (req_w, req_h):
        from PIL import Image as _Im
        images = [im.resize((max(1, round(im.width * req_w / w)),
                             max(1, round(im.height * req_h / h))), _Im.LANCZOS)
                  for im in images]
        _log(f"  resolution restore: {w}x{h} -> {req_w}x{req_h} 相当へリサイズ")

    extra = {"Backend": f"comfy/{spec['name']}-raw" if is_raw
             else f"comfy/{spec['name']}"}
    if hr:
        extra.update({"Hires upscale": hr["scale"],
                      "Hires mode": spec.get("hr_mode", "direct")})
        if spec.get("hr_mode") == "esrgan_2pass":
            extra.update({"Hires upscaler": spec["hr_upscale_model"],
                          "Hires steps": spec["hr_steps"],
                          "Denoising strength": spec["hr_denoise"]})
    if ad_used:
        extra["ADetailer model"] = ", ".join(m for m in ad_used if m)
    if loras:
        extra["Lora"] = ", ".join(f"{n}:{v}" for n, v in loras)

    infotext = _make_infotext(spec, prompt, neg, steps, sampler, scheduler, cfg,
                              seed, req_w, req_h, model, loras)
    count = len(b64)
    info = {
        "prompt": prompt, "all_prompts": [prompt] * count,
        "negative_prompt": neg, "all_negative_prompts": [neg] * count,
        "seed": seed, "all_seeds": [seed] * count,
        "subseed": seed, "all_subseeds": [seed] * count,
        # ネイティブは int の 0 を返す。float だと型が変わるので合わせる。
        "subseed_strength": 0,
        "width": req_w, "height": req_h, "sampler_name": sampler, "cfg_scale": cfg,
        "steps": steps, "batch_size": batch, "restore_faces": False,
        "face_restoration_model": None, "sd_model_name": model,
        "sd_model_hash": _model_hash(model), "sd_vae_name": None, "sd_vae_hash": None,
        "seed_resize_from_w": -1, "seed_resize_from_h": -1,
        # ネイティブは送った値をそのまま返す。HR の実現方式が違うため生成には
        # 使っていないが、値としてはエコーバックして互換を保つ。
        "denoising_strength": getattr(req, "denoising_strength", None),
        "extra_generation_params": extra,
        "index_of_first_image": 0, "infotexts": [infotext] * count, "styles": [],
        "job_timestamp": time.strftime("%Y%m%d%H%M%S"),
        # ネイティブは override_settings.CLIP_stop_at_last_layers をそのまま返す。
        # Krea2/Z-Image に clip skip の概念は無いが、値はエコーバックしておく。
        "clip_skip": int((getattr(req, "override_settings", None) or {})
                         .get("CLIP_stop_at_last_layers", 1) or 0),
        "is_using_inpainting_conditioning": False, "version": "forge-comfy-router",
    }
    # `parameters` は送信リクエストのエコーバックだが、こちらは turbo 向けに
    # steps/cfg/sampler を矯正し解像度も丸めている。送信値のまま返すと、クライアントが
    # これを EXIF に保存して group_hash の計算元にしているため、保存メタが実際の生成条件と
    # 食い違い、同一条件の画像がハッシュ違いで別グループに割れる。
    #
    # ただし**クライアントが送っていないキーは埋めない**。ネイティブ forge は送られた
    # ものだけを返し、未送信の任意フィールドは null のままにする。埋めてしまうと、
    # 矯正が一切走っていない場合でもネイティブと差分が出て、同じ設定なのにハッシュが
    # 変わる(例: sampler_index だけ送ったのに sampler_name まで埋まる)。
    raw = _req_dict(req)
    params = dict(raw)

    # **ルーターは送信されたパラメータを書き換えない。** 値の決定はクライアントの
    # 責任で、ルーターが決めると EXIF に送っていない値が残り、クライアントから
    # 制御する手段も無くなる。したがって params は送信値そのまま(dict(raw))。
    #
    #   生成に使う値(steps / cfg_scale / sampler / scheduler)-> 送信値をそのまま使う
    #   HR の 2nd パス設定(denoising_strength / hr_* )-> 無視するが送信値を返す
    #     実際に使った内部値は info.extra_generation_params に載せる
    #   width / height -> 16px 丸めは内部で吸収し、要求解像度を返す
    # A1111 は hires 有効時も `parameters` / infotext の Size は 1 パス目の解像度を保ち、
    # 倍率は `Hires upscale` で別に持つ(実ファイルだけが 2 倍になる)。
    # つまり width/height は「1 パス目の要求解像度」を指す。
    return b64, info, params


def _wrap_txt2img(app: FastAPI):
    from modules.api import models as api_models

    target = None
    for route in app.routes:
        if getattr(route, "path", None) == "/sdapi/v1/txt2img":
            target = route
            break
    if target is None:
        _log("WARN: /sdapi/v1/txt2img not found; router disabled")
        return

    original = target.endpoint

    def wrapped(txt2imgreq, *args, **kwargs):
        if _pick_spec(txt2imgreq) is None:
            return original(txt2imgreq, *args, **kwargs)
        try:
            images, info, params = _generate(txt2imgreq)
            return api_models.TextToImageResponse(
                images=images, parameters=params, info=json.dumps(info))
        except _Interrupted:
            # ネイティブ forge は中断を例外にせず、生成済みのぶんを返して正常終了する。
            # comfy は中断時に画像を出さないので images は空になる。
            _log("interrupted; returning empty result (native forge と同じ扱い)")
            return api_models.TextToImageResponse(
                images=[], parameters=_req_dict(txt2imgreq),
                info=json.dumps({"infotexts": [], "interrupted": True}))
        except Exception as e:
            _log(f"ERROR: {e!r}")
            raise
        finally:
            # 成否によらず comfy に VRAM を返させる。ここを飛ばすと次の forge ジョブが
            # ロードに失敗するため、例外経路でも必ず通す。
            _free_comfy_vram()
            try:
                shared.state.end()
                shared.state.sampling_step = 0
            except Exception:
                pass

    target.endpoint = wrapped
    if hasattr(target, "dependant"):
        target.dependant.call = wrapped
    _log(f"wrapped /sdapi/v1/txt2img -> {COMFY_URL} "
         f"(patterns: {', '.join(s['name'] for s in MODEL_SPECS)})")


def on_app_started(demo: gr.Blocks, app: FastAPI):
    _wrap_txt2img(app)


script_callbacks.on_app_started(on_app_started)
