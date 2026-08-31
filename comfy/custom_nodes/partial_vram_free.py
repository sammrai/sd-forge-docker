"""VRAM だけを返し、RAM に載せた重みは残す解放エンドポイント。

GPU は forge(sdui)と共有で、forge がモデルを載せる直前にルーターが comfy へ
解放を要求する。これまでは `POST /free {"unload_models": true}` を使っていたが、
これは VRAM だけでなく **RAM 上のコピーごと** 捨ててしまう:

    unload_all_models()                       model_management.py
      -> free_memory(1e30, device)
      -> LoadedModel.model_unload(unpatch_weights=True)
      -> ModelPatcher.detach(unpatch_all=True)
      -> unpatch_model(unpatch_weights=True)
           +- unpin_all_weights()             model_patcher.py
           +- partially_unload_ram(1e32)      <- RAM のコピーを全部捨てる

そのため forge のジョブが1本挟まるだけで、次の comfy 生成はチェックポイントを
ディスクから読み直す。実測でアーキ 1 本 18GB / 172 秒。連続生成なら 175〜182 秒で
終わる krea2 が、間に forge が入ると 335〜406 秒に伸びていた。

`ModelPatcher.partially_unload()` は VRAM 側(vbar とバックアップ)しか触らず
`unpin_all_weights` も `partially_unload_ram` も呼ばないので、こちらを使えば
RAM のコピーを残したまま VRAM を返せる。`/free` は真偽値2つしか受け取らず
部分解放を頼む口が無いため、ここで足す。

`free_memory()` は経由しない。あれは `memory_to_free < loaded_size()` のときだけ
部分解放に入る(model_management.py の LoadedModel.model_unload)ので、VRAM を
全部返そうとすると必ず detach 側に落ちてしまう。

解放そのものは comfy 本体のフラグ機構に相乗りする。`/free` と同じく prompt worker
スレッドが生成の合間に処理するので、生成中に VRAM を剥がす事故が起きない。
"""

import asyncio
import logging
import threading

import comfy.model_management as mm
from aiohttp import web
from server import PromptServer

# comfy が全部を捨てる 1e32 と同じ意味の「全部」。partially_unload は
# 1e30 以上を "requested=all" として扱う。
FREE_ALL = 1e32

_partial_requested = False
_done = threading.Event()
_last_mode = None
_original_unload_all_models = mm.unload_all_models


def _partially_unload_all():
    """ロード済みモデルの VRAM を解放する。detach しないので RAM のピンは残る。

    1つでも失敗したら None を返し、呼び出し側が本来の全解放へフォールバックする。
    VRAM が返らないまま forge がロードすると OOM で落ちるため、
    中途半端に成功した状態で先に進ませない。
    """
    freed = 0
    for loaded in list(mm.current_loaded_models):
        model = loaded.model
        if model is None:  # weakref が切れている(GC 待ち)
            continue
        try:
            freed += model.partially_unload(model.offload_device, FREE_ALL)
        except Exception:
            logging.exception("partial_vram_free: partially_unload failed")
            return None
    mm.soft_empty_cache(force=True)
    return freed


def _unload_all_models():
    """comfy 本体の unload_all_models を包む。

    直前に /router/free_vram が呼ばれていたときだけ部分解放にする。
    それ以外の呼び出し(comfy 本体や他の拡張)は素通しする。
    """
    global _partial_requested, _last_mode
    if not _partial_requested:
        return _original_unload_all_models()
    _partial_requested = False
    _last_mode = "partial"
    try:
        freed = _partially_unload_all()
        if freed is None:
            _last_mode = "full"
            return _original_unload_all_models()
        logging.info(
            "partial_vram_free: released %.1fMB of VRAM, %d model(s) kept in RAM",
            freed / (1024 ** 2), len(mm.current_loaded_models))
    finally:
        _done.set()


mm.unload_all_models = _unload_all_models


@PromptServer.instance.routes.post("/router/free_vram")
async def free_vram(request):
    """VRAM を返す。RAM に載せた重みは残す。

    worker が実際に解放し終えるまで待ってから返す。呼び出し側(ルーター)は
    この応答を見てから forge にモデルを載せるので、待たずに返すと解放前に
    forge がロードを始める競合が残る。
    """
    global _partial_requested, _last_mode
    timeout = 60.0
    try:
        body = await request.json()
        timeout = float(body.get("timeout", timeout))
    except Exception:
        pass  # ボディ無しも許す

    _done.clear()
    _last_mode = None
    _partial_requested = True
    PromptServer.instance.prompt_queue.set_flag("unload_models", True)

    completed = await asyncio.get_running_loop().run_in_executor(
        None, _done.wait, timeout)
    if not completed:
        # 生成が長引いているなど。ルーター側は 200 以外をフォールバック扱いに
        # するので、ここで失敗を明示して従来の全解放へ倒す。
        logging.warning("partial_vram_free: timed out after %.0fs", timeout)
        return web.json_response({"status": "timeout"}, status=504)
    # mode=full は部分解放に失敗して全解放へ倒れた場合。VRAM は返っているので
    # 成功だが、RAM の重みは残っていない。
    return web.json_response({"status": "ok", "mode": _last_mode})


# comfy はノードを持たないモジュールにも警告を出さないが、慣例に合わせて空で置く。
NODE_CLASS_MAPPINGS = {}
