"""チェックポイントのアーキを判定して返すエンドポイント。

ルーターの振り分けは `override_settings.sd_model_checkpoint` の**名前**を正規表現で
見ていた。名前は当てにならない:

  - モデル名に別アーキの名前が入る(名前が Illustrious で中身が Anima、を実在で確認)
  - 同じ Civitai モデル(mid)でも版によってアーキが違う
  - アーキ名が他の語の接頭辞になる(anima / animagine / animal / animated)

食い違うと **どちらの向きでも静かに壊れる**。中身が新アーキなのに forge へ流れれば
"Failed to recognize model type!"、別アーキなのに comfy へ流れれば壊れた絵が出る。

そこで**モデル自身に聞く**。アーキの知識は ComfyUI にしか無いのでここに置く。
判定は `comfy.model_detection` に任せるので、**新しいアーキが増えても手を入れなくてよい**
(キー名からアーキを推測するコードを書かない)。

重みは読まない。safetensors は先頭8バイトの長さ + JSON ヘッダなので、テンソル名と形状
だけを読んで `device="meta"` の空テンソルを組み、それを判定に渡す。4GB のファイルでも
実際に読むのは数百KB で、メモリも消費しない。

  GET /router/detect_arch?name=<comfy の diffusion_models 相対パス>
  -> {"arch": "Anima", "mro": ["Anima", "BASE", "object"], "keys": 879}
  -> 判定できなければ {"arch": null, ...}
"""
import json
import logging
import struct

import torch
from aiohttp import web

import comfy.model_detection
import comfy.utils
import folder_paths
from server import PromptServer

# safetensors のヘッダに載る dtype 文字列 -> torch の型。
# 判定は形状しか見ないが、dtype で分岐する検出器があるので合わせておく。
_DTYPES = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
    "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
    "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
}
for _name in ("float8_e4m3fn", "float8_e5m2"):
    if hasattr(torch, _name):
        _DTYPES["F8_" + _name.split("_")[-1].upper()] = getattr(torch, _name)

_cache = {}


def _meta_state_dict(path):
    """safetensors のヘッダだけを読み、meta テンソルの state_dict を組む。"""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)
    sd = {}
    for key, info in header.items():
        dtype = _DTYPES.get(info.get("dtype"), torch.float16)
        sd[key] = torch.empty(info["shape"], dtype=dtype, device="meta")
    return sd


def _detect(path):
    sd = _meta_state_dict(path)
    total = len(sd)
    # チェックポイント形式(model.diffusion_model.*)と diffusion model 単体の両方が来る。
    # comfy.sd.load_diffusion_model_state_dict と同じ手順で前置きを剥がす。
    prefix = comfy.model_detection.unet_prefix_from_state_dict(sd)
    stripped = comfy.utils.state_dict_prefix_replace(sd, {prefix: ""}, filter_keys=True)
    if len(stripped) > 0:
        sd = stripped
    config = comfy.model_detection.model_config_from_unet(sd, "")
    if config is None:
        return None, [], total
    cls = type(config)
    return cls.__name__, [c.__name__ for c in cls.__mro__], total


@PromptServer.instance.routes.get("/router/detect_arch")
async def detect_arch(request):
    name = request.query.get("name", "")
    if not name:
        return web.json_response({"error": "name is required"}, status=400)
    if name in _cache:
        return web.json_response(_cache[name])

    path = folder_paths.get_full_path("diffusion_models", name)
    if path is None:
        return web.json_response(
            {"error": f"{name!r} is not visible under diffusion_models"}, status=404)
    try:
        arch, mro, keys = _detect(path)
    except Exception as e:
        logging.exception("detect_arch failed for %s", name)
        return web.json_response({"error": f"{type(e).__name__}: {e}"}, status=500)

    result = {"arch": arch, "mro": mro, "keys": keys}
    _cache[name] = result
    return web.json_response(result)
