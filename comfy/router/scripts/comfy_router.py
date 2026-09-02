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
        # HR の 2nd pass の既定も base と同じサンプラーにする。公式アップスケーラの
        # テンプレートは dpmpp_2m_sde/beta だが、テンプレートは step 数もモデルも
        # 前提が違うため、そのまま持ち込むと破綻するのを実測した。
        # あくまで未送信時の既定で、クライアントが送ればその値を使う。
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
    {
        "name": "anima",
        # civitdl の stem は "homosimileAnima_v10" "waiANIMA_v10Base10"
        # "anima_turboV11" "maleGgmixIllustrious_anima11Base" のように表記がゆれる。
        # ただの r"anima" だと animagine(SDXL) / animal / animated まで拾ってしまうので、
        # 直後に小文字が続く語を除く。大文字が続く形(animaGay17)は残したいので
        # IGNORECASE はスコープ指定にして、後読みには効かせない。
        "pattern": re.compile(r"(?i:anima)(?![a-z])"),
        # 振り分けキーと checkpoint 名は同じフィールド(override_settings.
        # sd_model_checkpoint)なので、spec が選ばれた時点で名前は必ず非空。
        # つまり _resolve_unet の「名前が空なら既定」分岐には到達しない。
        # 他アーキと構造を揃えるために置くだけなので、専用ファイルは持たず
        # civitdl 管理下の公式 turbo を指す。
        "unet": "Anima-mid_2458426-vid_3263843/"
                "anima_turboV11-mid_2458426-vid_3263843.safetensors",
        # Anima の TE は Qwen3-0.6B。CLIPLoader の type は公式テンプレートどおり
        # "stable_diffusion"(実際の種別は state_dict から検出される)。
        "clip": "qwen_3_06b_base.safetensors",
        "clip_type": "stable_diffusion",
        # Qwen-Image と同じ VAE。krea2 と同一ファイルを共有する(HF 側とサイズ一致)。
        "vae": "qwen_image_vae.safetensors",
        "latent_node": "EmptyLatentImage",
        # shift=3.0 は supported_models.Anima の sampling_settings に入っており、
        # ModelSamplingAuraFlow を挟む必要はない(公式テンプレートにも無い)。
        "aura_shift": None,
        # 未送信時の既定。運用するのは turbo 系なので公式の蒸留版推奨値
        # (cfg 1 / 8-12 steps)に合わせる。非蒸留の base/aesthetic は 30-50 steps /
        # cfg 4-5 だが、どのバリアントかはルーターが判断しない(モデルの中身を
        # 知らない)。値はクライアントが送る。
        #
        # なお steps / cfg / sampler は base パスでは**到達しない**。forge の API
        # モデルが steps=50 / cfg_scale=7.0 / sampler_index="Euler" を必ず埋めるため
        # (稼働中の /openapi.json で確認)。到達するのは scheduler(既定 None)と、
        # 進捗の総ステップ見積もりだけ。cfg は hr_cfg が既定 1.0 で埋まるので
        # どこからも到達しないが、_hr2pass が参照するのでキーは残す。
        "sampler": "euler",
        "scheduler": "simple",
        "steps": 8,
        "cfg": 1.0,
        # HR 2nd パスの既定。hr_upscaler / hr_sampler_name / denoising_strength が
        # 未送信のときだけ使われる(いずれも API 既定は None なので到達する)。
        # sampler と denoise はチェックポイント作者(Civitai/HomoSimile Anima)の
        # 推奨由来: er_sde が作者の既定サンプラー、denoise は 0.35-0.4。
        # **ただし代替との比較実測はしていない。**
        "hr_sampler": "er_sde",
        "hr_scheduler": "simple",
        "hr_steps": 8,
        # direct(1664x2432 一発, 260.7s)と esrgan_2pass(83.9s)を同一シードで比較し、
        # 顔を等倍で見て両方とも破綻なし。同等画質で 2パスが3倍速いので 2パスに統一。
        "hr_mode": "esrgan_2pass",
        # 作者は R-ESRGAN 4x+ Anime6B を推奨しているが、同一 seed・同一 HR 設定で
        # 4x-UltraSharp / RealESRGAN_x4plus_anime_6B / 4x_fatal_Anime を比較したところ
        # 画素差 4.09・エッジエネルギー 10.53 vs 10.30 で判別できず、顔を等倍で見ても
        # 差が無かった。替える理由が実測で出なかったので krea2 / z-image と揃える。
        "hr_upscale_model": "4x-UltraSharp.pth",
        "hr_denoise": 0.35,
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
    # A1111 の Restart は comfy に対応が無い。以前は euler へ差し替えていたが、
    # 別のサンプラーで生成されたことにクライアントが気づけないため削除した。
    # 解決できない名前は _map_sampler_pair がエラーにする。
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
    同じアーキで別の量子化を指定したい場合がある。

    **解決できない名前は spec 既定に落とさずエラーにする。** 落とすと、指定したのとは
    別のモデルで生成されたことにクライアントは気づけない(parameters には指定名が返る)。
    LoRA / hr_upscaler / sampler と同じ扱い。
    """
    name = (requested or "").strip()
    if not name:
        return spec["unet"]
    cat = _unet_catalog()
    hit = cat.get(name) or cat.get(os.path.basename(name))
    if hit is None:
        cat = _unet_catalog(refresh=True)  # 直前に落としたものかもしれない
        hit = cat.get(name) or cat.get(os.path.basename(name))
    if hit is None:
        raise RuntimeError(
            f"checkpoint '{name}' が comfy から見えません。"
            f"models/Stable-diffusion 配下にあるか、civitdl の取得が完了しているか"
            f"確認してください。")
    if hit != spec["unet"]:
        _log(f"unet override: {spec['unet']} -> {hit} (クライアント指定)")
    return hit


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
            f"LoRA '{name}' が comfy から見えません。"
            f"models/Lora 配下にあるか、civitdl の取得が完了しているか"
            f"確認してください。")
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
    `"res_multistep simple"` のように comfy 名 + scheduler を1つの文字列にした形も
    同様に受ける(A1111 の `"DPM++ 2M Karras"` と同じ書き方)。
    """
    key = str(raw or "").strip().lower()
    samplers, schedulers = _comfy_sampler_names()

    def _sched(name):
        """scheduler 名を comfy 名へ。解決できなければ None。"""
        k = str(name or "").strip().lower()
        return k if k in schedulers else SCHEDULER_MAP.get(k)

    if not key:
        sampler, scheduler = def_sampler, def_scheduler
    elif key in samplers:
        sampler, scheduler = key, def_scheduler
    elif key in SAMPLER_MAP:
        sampler, scheduler = SAMPLER_MAP[key]
    else:
        # "res_multistep simple" のようにサンプラー名と scheduler を1つの文字列で
        # 送ってくる形も受ける。A1111 の "DPM++ 2M Karras" と同じ書き方で、
        # SAMPLER_MAP に列挙済みの組み合わせ以外にも適用できるようにする。
        head, _, tail = key.rpartition(" ")
        tail_sched = _sched(tail) if head else None
        if tail_sched and (head in samplers or head in SAMPLER_MAP):
            sampler = head if head in samplers else SAMPLER_MAP[head][0]
            scheduler = tail_sched
        else:
            # 解決できない名前を既定へ落とすと、指定と違う値で生成されたことに
            # クライアントは気づけない(parameters には送信値が返るため)。
            # hr_upscaler と同じくエラーにする。
            raise RuntimeError(
                f"sampler={raw!r} を解決できません。"
                f"A1111 名: {', '.join(sorted(SAMPLER_MAP))} / "
                f"comfy 名: {', '.join(sorted(samplers))}")
    if sched_raw:
        resolved = _sched(sched_raw)
        if resolved is None:
            raise RuntimeError(
                f"scheduler={sched_raw!r} を解決できません。"
                f"A1111 名: {', '.join(sorted(SCHEDULER_MAP))} / "
                f"comfy 名: {', '.join(sorted(schedulers))}")
        scheduler = resolved
    return sampler, scheduler


def _map_sampler(req, spec):
    return _map_sampler_pair(
        getattr(req, "sampler_name", None) or getattr(req, "sampler_index", None),
        getattr(req, "scheduler", None), spec["sampler"], spec["scheduler"])


def _sampling(req, spec):
    """steps / cfg を決める。値の決定はクライアントの責任。

    ルーターはモデルの中身(蒸留の有無、蒸留時の cfg、推奨 step 数)を知らない。
    こちらで推奨値を定義すると、想定と違う使い方をクライアントができなくなる。
    未送信のときだけ spec の値へ落ちる。
    """
    req_steps = int(getattr(req, "steps", 0) or 0)
    req_cfg = float(getattr(req, "cfg_scale", 0) or 0)
    return (req_steps or spec["steps"]), (req_cfg or spec["cfg"])


def _warn_unsupported(req, spec):
    """対応していない指定をサイレントに落とさない。

    絵が変わらないもの(ControlNet など)はログに出すだけ。**別モデルの指定**は
    エラーにする。黙って本体と同じモデルで描き直すと、指定と違うもので生成された
    ことをクライアントが検出できないため(checkpoint / LoRA / sampler と同じ扱い)。
    """
    ov = getattr(req, "override_settings", None) or {}
    if ov.get("sd_vae"):
        _log(f"NOTE: override_settings.sd_vae={ov['sd_vae']!r} は無視される "
             f"(comfy 側は {spec['vae']} 固定)")
    for unit in _adetailer_units(req):
        if unit.get("ad_controlnet_model") not in (None, "None", ""):
            _log("NOTE: ad_controlnet_* は Krea2/Z-Image 用 ControlNet が無いため無視")
        # 判定条件はネイティブ forge の get_override_settings と同じにする。
        # ad_use_checkpoint / ad_use_vae が真でも実名が入っていなければネイティブも
        # 何もしないので、そこで落とすとクライアントの既定形が通らなくなる。
        ck = unit.get("ad_checkpoint")
        if (unit.get("ad_use_checkpoint") and ck
                and ck not in ("None", "Use same checkpoint")):
            raise RuntimeError(
                f"ad_checkpoint={ck!r} は未対応です。comfy 側の ADetailer は本体と"
                f"同じ checkpoint で描き直します。黙って別モデルで生成したことに"
                f"ならないようエラーにしています。")
        vae = unit.get("ad_vae")
        if (unit.get("ad_use_vae") and vae
                and vae not in ("None", "Use same VAE")):
            raise RuntimeError(
                f"ad_vae={vae!r} は未対応です。comfy 側は {spec['vae']} 固定です。")


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


def _post_comfy(path, payload, timeout=120):
    req = urllib.request.Request(
        COMFY_URL + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    body = urllib.request.urlopen(req, timeout=timeout).read()
    try:
        return json.loads(body or b"{}")
    except ValueError:
        return {}


def _free_comfy_vram():
    """comfy は生成後もモデルを VRAM に保持し続けるため、明示的に解放させる。
    これを怠ると次の forge ジョブがロードに失敗する(実測 8GB 保持)。

    まず自前の `/api/router/free_vram` を試す(comfy/custom_nodes/partial_vram_free.py)。
    こちらは VRAM だけを返し、RAM に載せた重みを残すので、forge のジョブが挟まっても
    次の comfy 生成がディスクからの読み直しにならない。

    使えなければ `/api/free` へ倒す。**その場合も free_memory は送らない。** comfy の
    /api/free は2つのフラグで効果が別物で(main.py の flags 処理)、unload_models は
    unload_all_models() を呼ぶのに対し、free_memory は execution の reset() でノード
    出力キャッシュを作り直す。後者を送るとロード済みモデルのオブジェクトごと捨てられ、
    ノード出力の再利用も効かなくなる。VRAM を返す目的には前者で足りる。
    """
    global _partial_free_supported
    if _partial_free_supported:
        try:
            res = _post_comfy("/api/router/free_vram", {"timeout": 60})
            # mode=full は comfy 側で部分解放に失敗して全解放へ倒れた場合。
            # VRAM は返っているが RAM の重みは残っていない。
            kept = res.get("mode") == "partial"
            _log(f"comfy VRAM released ({'RAM の重みは保持' if kept else 'RAM も解放'})")
            return
        except Exception as e:
            # 404 = 拡張の入っていない古いイメージ。504 = worker が時間内に
            # 処理できなかった。どちらも全解放へ倒せば VRAM だけは確実に返る。
            code = getattr(e, "code", None)
            if code == 404:
                _partial_free_supported = False
                _log("NOTE: comfy に /router/free_vram が無い。全解放へフォールバック")
            else:
                _log(f"WARN: partial VRAM release failed ({e!r}); 全解放へフォールバック")
    try:
        _post_comfy("/api/free", {"unload_models": True})
        _log("comfy VRAM released")
    except Exception as e:
        _log(f"WARN: comfy VRAM release failed: {e!r}")


# comfy が VRAM を握ったままかどうか。生成のたびに解放すると comfy の
# DynamicVRAM がステージした重み(実測 11.7GB)まで捨てられ、次の生成が必ず
# ディスクからの読み直しになる(実測 +98秒)。そこで解放は forge が実際に
# VRAM を要求するまで遅らせる。
_comfy_holds_vram = False
_lazy_release_installed = False
# 部分解放エンドポイントの有無。404 を一度見たら以降は問い合わせない。
_partial_free_supported = True


def _release_comfy_if_held():
    global _comfy_holds_vram
    if not _comfy_holds_vram:
        return
    _comfy_holds_vram = False
    _free_comfy_vram()


def _install_lazy_release():
    """forge が VRAM へモデルを載せる直前に comfy を解放させる。

    load_models_gpu は forge が GPU を使う唯一の入口(load_model_gpu もここを
    通る)なので、txt2img 以外の経路(img2img など)も漏れなく拾える。
    フックできなかった場合は従来どおり生成ごとに即解放する(下の finally 参照)。
    """
    global _lazy_release_installed
    try:
        from backend import memory_management
        original = memory_management.load_models_gpu

        def patched(*a, **kw):
            _release_comfy_if_held()
            return original(*a, **kw)

        memory_management.load_models_gpu = patched
        _lazy_release_installed = True
        _log("lazy comfy VRAM release installed (hook: load_models_gpu)")
    except Exception as e:
        _log(f"WARN: lazy release hook failed; falling back to eager free: {e!r}")


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
        tw, th = hr2pass["target"]
        if hr2pass["kind"] == "model":
            # モデル拡大は倍率が固定(4x など)なので、目標解像度へ縮小して合わせる
            g["40"] = {"class_type": "UpscaleModelLoader",
                       "inputs": {"model_name": hr2pass["model"]}}
            g["41"] = {"class_type": "ImageUpscaleWithModel",
                       "inputs": {"upscale_model": ["40", 0], "image": ["8", 0]}}
            g["42"] = {"class_type": "ImageScale",
                       "inputs": {"image": ["41", 0], "upscale_method": "lanczos",
                                  "width": tw, "height": th, "crop": "disabled"}}
        else:
            # 単純リサイズ(Lanczos など)。モデルを経由しない
            g["42"] = {"class_type": "ImageScale",
                       "inputs": {"image": ["8", 0],
                                  "upscale_method": hr2pass["model"],
                                  "width": tw, "height": th, "crop": "disabled"}}
        g["43"] = {"class_type": "VAEEncode",
                   "inputs": {"pixels": ["42", 0], "vae": ["12", 0]}}
        g["44"] = {"class_type": "KSampler",
                   "inputs": {"model": model_out, "seed": seed,
                              "steps": hr2pass["steps"], "cfg": hr2pass["cfg"],
                              "sampler_name": hr2pass["sampler"],
                              "scheduler": hr2pass["scheduler"],
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
        # キーが1つも当たらない LoRA(アーキ違い)は、警告だけ出して適用せずに通る。
        # 一度これをエラーにする custom node を入れたが、**ネイティブ forge も同じく
        # 黙って無視して生成を続ける**(sd_forge_lora/networks.py: 未マッチが13件を
        # 超えると元のモデルをそのまま返す)。片方の経路だけ厳しくすると I/F が
        # 食い違うため、ネイティブに合わせて素のローダに戻した。
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

# A1111 のアップスケーラ名のうち、モデルを使わない単純リサイズのもの。
# comfy の ImageScale の upscale_method へ写す。
PIXEL_UPSCALERS = {
    "lanczos": "lanczos", "nearest": "nearest-exact", "nearest-exact": "nearest-exact",
    "bilinear": "bilinear", "bicubic": "bicubic", "area": "area",
    # Latent 系は latent 空間での拡大だが、ここでは同等の画素拡大で代替する
    # (VAE 往復を挟む都合上、latent のまま引き渡す経路が無い)。
    "latent": "bilinear", "latent (bicubic)": "bicubic",
    "latent (nearest)": "nearest-exact", "latent (nearest-exact)": "nearest-exact",
}


def _resolve_upscaler(name, spec):
    """hr_upscaler を「単純リサイズ」か「モデル拡大」に解決する。

    値の良し悪しはクライアントが決める。Lanczos は Z-Image で破綻すると実測して
    いるが、指定されたらそのまま使う(ルーターが good/bad を判断しない)。
    解決できない名前は黙って spec 既定に落とさずエラーにする。
    """
    if not name:
        return ("model", spec["hr_upscale_model"])
    key = str(name).strip()
    if key.lower() in PIXEL_UPSCALERS:
        return ("pixel", PIXEL_UPSCALERS[key.lower()])
    catalog = _upscale_catalog()
    for cand in catalog:
        if cand.lower() == key.lower() or os.path.splitext(cand)[0].lower() == key.lower():
            return ("model", cand)
    raise RuntimeError(
        f"hr_upscaler={name!r} を解決できません。"
        f"単純リサイズ: {', '.join(sorted(PIXEL_UPSCALERS))} / "
        f"モデル: {', '.join(catalog) if catalog else '(なし)'}")


_UPSCALE_CATALOG = None


def _upscale_catalog():
    global _UPSCALE_CATALOG
    if _UPSCALE_CATALOG is None:
        try:
            _UPSCALE_CATALOG = json.loads(urllib.request.urlopen(
                COMFY_URL + "/api/models/upscale_models", timeout=30).read())
        except Exception as e:
            _log(f"WARN: upscale model catalog fetch failed: {e}")
            _UPSCALE_CATALOG = []
    return _UPSCALE_CATALOG


def _hr2pass(req, spec, hr):
    """HR の 2nd パス設定を解決する。値はクライアントのもの。

    `hr_upscaler` / `hr_sampler_name` / `hr_scheduler` / `hr_second_pass_steps` /
    `hr_cfg` は未送信(None / 空 / 0)なら spec の既定へ落ちる。**`denoising_strength`
    だけは別で、forge の API モデルが既定 0.75 を必ず埋めるため spec["hr_denoise"] には
    落ちない。**ネイティブ forge も同じ 0.75 で焼き直すので、これで挙動が揃う。

    hr_cfg は forge でも「HR パスの cfg 実値」であって 1パス目からの継承ではない
    (processing.py の `if self.hr_cfg == 1: self.hr_uc = None`)。
    """
    kind, up = _resolve_upscaler(getattr(req, "hr_upscaler", None), spec)
    den = getattr(req, "denoising_strength", None)
    steps = int(getattr(req, "hr_second_pass_steps", 0) or 0)
    sampler, scheduler = _map_sampler_pair(
        getattr(req, "hr_sampler_name", None), getattr(req, "hr_scheduler", None),
        spec["hr_sampler"], spec["hr_scheduler"])
    cfg = getattr(req, "hr_cfg", None)
    hr2pass = {"kind": kind, "model": up, "target": (hr["w"], hr["h"]),
               "denoise": float(den) if den is not None else spec["hr_denoise"],
               "steps": steps or spec["hr_steps"],
               "cfg": float(cfg) if cfg not in (None, "") else spec["cfg"],
               "sampler": sampler, "scheduler": scheduler}
    _log(f"  hr 2nd pass: {kind}={up} sampler={sampler}/{scheduler} "
         f"steps={hr2pass['steps']} cfg={hr2pass['cfg']} denoise={hr2pass['denoise']}")
    return hr2pass


def _hr2pass_steps(hr2pass):
    """2nd パスが実際に回すサンプリングステップ数(進捗の見積もり用)。

    **denoise では割り引かない。** A1111 の img2img は denoising_strength で
    実ステップ数が減るが、comfy はスケジュールの後ろだけを切り出すので回す本数は
    `steps` ちょうど(samplers.py の set_steps: `sigmas[-(steps + 1):]`)。
    割り引くと HR パスの進捗が実際より少なく見積もられ、ETA がずれる。
    """
    if not hr2pass:
        return 0
    return hr2pass["steps"]


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

    **2nd パスの値はクライアントのもの**(解決は _hr2pass)。denoising_strength /
    hr_second_pass_steps / hr_sampler_name / hr_scheduler / hr_upscaler / hr_cfg は
    送信値をそのまま使う。
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
        # 黙ってスキップすると ADetailer が効いていないことに気づけない。
        raise RuntimeError(
            f"ADetailer model {name!r} が見つかりません。"
            f"models/adetailer 配下にあるか確認してください。"
            f"利用可能: {', '.join(sorted(models)) if models else '(なし)'}")

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


def _ad_steps(unit, base_steps):
    """ADetailer の inpaint steps。ad_use_steps が偽なら **リクエストの steps** に落ちる。

    ネイティブの ADetailer も同じで、ad_use_steps が偽なら p.steps を使う。
    以前はここが spec["steps"] に落ちていたが、同一アーキに蒸留版(8-12 steps)と
    非蒸留版(25-35 steps)が併存すると、base パスは送信値どおりなのに ADetailer だけ
    要求と違う steps で回る。しかも info には「ADetailer 適用」としか出ないので
    クライアントからは検出できない(Anima で顕在化。Krea 2 の RAW でも同じ)。
    ad_cfg が cfg に落ちるのと揃える意味もある。
    """
    s = int(unit.get("ad_steps", 0) or 0) if unit.get("ad_use_steps") else 0
    return s or base_steps


def _run_adetailer_unit(spec, pil, unit, prompt, neg, cfg, seed, loras, st, offset,
                        base_steps, unet=None, pw=None, ph=None):
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
    steps = _ad_steps(unit, base_steps)
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

    _log(f"ADetailer[{unit.get('ad_model')}]: {done}/{len(masks)} region(s) inpainted "
         f"(steps={steps} cfg={ad_cfg} denoise={denoise} {iw}x{ih})")
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
    steps, cfg = _sampling(req, spec)
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
    hr = _hr_params(req, spec, w, h, steps)
    units = _adetailer_units(req)

    # HR の実現方法はモデルごとに違う(実測で決めた。詳細は _hr_params の docstring)。
    #   direct       : 目標解像度で1パス生成
    #   esrgan_2pass : base 生成 -> 拡大 -> 低 denoise で焼き直し
    # 現状は Krea2 / Z-Image とも esrgan_2pass。Krea2 は direct でも破綻しないが、
    # 同一シードで比較して画質に有意差が無く速度も 2パスがわずかに速かったため統一した。
    # 進捗の総ステップ数を出すのに要るので、グラフ組み立てより先に解決しておく。
    # 名前の解決に失敗した場合もここで落ちる(st.begin より前なので job を掴まない)。
    gw, gh = w, h
    hr2pass = None
    if hr:
        if spec.get("hr_mode") == "esrgan_2pass":
            hr2pass = _hr2pass(req, spec, hr)
        else:
            gw, gh = hr["w"], hr["h"]

    # 進捗の総ステップ数。ADetailer は検出数が事前に分からないので 1領域/ユニットで
    # 見積もっておき、実際に増えたぶんは _submit_and_wait 側で max を押し上げる。
    total = steps + sum(_ad_steps(u, steps) for u in units) + _hr2pass_steps(hr2pass)

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
    graph = _build_txt2img(spec, prompt, neg, gw, gh, steps, cfg, seed, batch,
                           sampler, scheduler, loras, hr2pass=hr2pass, unet=unet)
    # 進捗はサンプラーノードだけを対象にする("3"=base, "44"=HR の 2nd パス)
    node_offsets = {"3": 0}
    if hr2pass:
        node_offsets["44"] = steps
    images = _open_outputs(_submit_and_wait(graph, st, step_offset=offset,
                                            node_offsets=node_offsets))
    offset += steps + _hr2pass_steps(hr2pass)
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
                                            loras, st, offset, steps, unet=unet,
                                            pw=gw, ph=gh)
            new_images.append(out)
            offset += _ad_steps(unit, steps) * max(1, done)
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

    extra = {"Backend": f"comfy/{spec['name']}"}
    if hr:
        extra.update({"Hires upscale": hr["scale"],
                      "Hires mode": spec.get("hr_mode", "direct")})
        if hr2pass:
            extra.update({"Hires upscaler": hr2pass["model"],
                          "Hires sampler": f"{hr2pass['sampler']}/{hr2pass['scheduler']}",
                          "Hires CFG Scale": hr2pass["cfg"],
                          "Hires steps": hr2pass["steps"],
                          "Denoising strength": hr2pass["denoise"]})
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
        # ネイティブ(processed.js())はリクエストの値をそのまま返す。ここに
        # リテラルを埋めると型が食い違う(int 0 を埋めていたが、実測ではネイティブ
        # SDXL が float 0.0 を返しており、クライアントの機械照合で差分として出た)。
        # 値そのものと同じく型もエコーバックに任せる。
        "subseed_strength": getattr(req, "subseed_strength", 0),
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
    # `parameters` は送信リクエストのエコーバック。クライアントはこれを EXIF に保存し
    # group_hash の計算元にするため、実際の生成条件と食い違うと同一条件の画像が
    # ハッシュ違いで別グループに割れる。書き換えないので食い違いは起きない。
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
    #   HR の 2nd パス設定(denoising_strength / hr_upscaler / hr_sampler_name /
    #     hr_scheduler / hr_second_pass_steps / hr_cfg)-> 送信値をそのまま使う
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
            # 成否によらず comfy の VRAM を「解放予定」にする。実際の解放は forge が
            # VRAM を要求した時点(_install_lazy_release のフック)。こうすると comfy の
            # 連続生成でステージ済みの重みが残り、ディスク読み直しが起きない。
            # フックを入れられなかった場合だけ従来どおり即解放する。
            global _comfy_holds_vram
            if _lazy_release_installed:
                _comfy_holds_vram = True
            else:
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
    _install_lazy_release()


script_callbacks.on_app_started(on_app_started)
