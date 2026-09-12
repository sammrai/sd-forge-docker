#!/usr/bin/env python3
"""焼き直し後の差分検証。sdui コンテナ内で実行する。

  python3 verify_bake.py <新しいckpt> <バックアップのckpt>

VAE 以外が1バイトも変わっていないことを確認する。RAM を 14GB ほど使う。
"""
import sys, collections, hashlib, torch
from safetensors.torch import load_file


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    new, old = load_file(sys.argv[1]), load_file(sys.argv[2])

    print("tensors: new %d / old %d" % (len(new), len(old)))
    for k, v in sorted(collections.Counter(
            (k.split(".")[0], str(t.dtype)) for k, t in new.items()).items()):
        print("   %-20s %-15s %d" % (k[0], k[1], v))

    nk = {k for k in new if not k.startswith("first_stage_model.")}
    ok = {k for k in old if not k.startswith("first_stage_model.")}
    h = lambda sd, ks: hashlib.sha256(b"".join(
        k.encode() + sd[k].contiguous().view(torch.uint8).numpy().tobytes()
        for k in sorted(ks))).hexdigest()
    print("UNet/TE キー集合が同一 :", nk == ok, "(%d keys)" % len(nk))
    print("UNet/TE バイト列が不変 :", nk == ok and h(new, nk) == h(old, ok))
    print("VAE キー数             : %d -> %d" % (
        len(old) - len(ok), len(new) - len(nk)))
    stray = [k for k in new if "model_ema" in k or k.startswith("first_stage_model.loss.")]
    print("余計なキー             :", stray or "なし")


if __name__ == "__main__":
    main()
