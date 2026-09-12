#!/usr/bin/env python3
"""VAE ラウンドトリップ健全性チェック。sdui コンテナ内で実行する。

  python3 vae_roundtrip.py <画像> <target.safetensors> [<target> ...]

target はチェックポイント（`first_stage_model.*` を抜き出す）でも
VAE 単体ファイルでもよい。正常なら入力の彩度/std をほぼ保つ。
SD1.5 と SDXL の VAE は同じ構造なので config は共通で足りる
（encode/decode に scaling_factor は効かない）。
"""
import sys, torch, numpy as np
from safetensors.torch import load_file
from diffusers import AutoencoderKL
from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
from PIL import Image

CFG = dict(in_channels=3, out_channels=3,
           down_block_types=["DownEncoderBlock2D"] * 4,
           up_block_types=["UpDecoderBlock2D"] * 4,
           block_out_channels=[128, 256, 512, 512],
           layers_per_block=2, latent_channels=4, norm_num_groups=32,
           sample_size=1024, scaling_factor=0.13025, act_fn="silu")


def build(path):
    sd = load_file(path)
    vae = {k: v for k, v in sd.items() if k.startswith("first_stage_model.")}
    sd = vae or sd
    if not any(k.endswith("decoder.conv_out.weight") for k in sd):
        return None, "VAE キーなし（別アーキか、VAE未焼き込み）"
    m = AutoencoderKL(**CFG)
    miss, unexp = m.load_state_dict(convert_ldm_vae_checkpoint(sd, m.config), strict=False)
    return m.eval().float(), f"missing {len(miss)} unexpected {len(unexp)}"


def metrics(a):
    a = np.asarray(a).astype(np.float32) / 255.0
    mx, mn = a.max(2), a.min(2)
    return float(np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0).mean()), float(a.std())


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    img_path, targets = sys.argv[1], sys.argv[2:]
    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    im = im.resize((512, 512), Image.LANCZOS)
    x = torch.from_numpy(np.array(im)).float().permute(2, 0, 1)[None] / 127.5 - 1.0

    base_sat, base_std = metrics(im)
    print("%-4s %-38s %10s %8s" % ("#", "", "saturation", "std"))
    print("%-4s %-38s %10.4f %8.4f   <- 入力" % ("", "INPUT", base_sat, base_std))
    for i, t in enumerate(targets, 1):
        name = t.split("/")[-1]
        name = name if len(name) <= 38 else name[:35] + "..."
        m, note = build(t)
        if m is None:
            print("[%d]  %-38s %s" % (i, name, note))
            continue
        with torch.no_grad():
            y = m.decode(m.encode(x).latent_dist.mode()).sample
        arr = ((y.clamp(-1, 1)[0].permute(1, 2, 0).numpy() + 1) * 127.5).round().astype(np.uint8)
        sat, std = metrics(arr)
        ratio = sat / max(1e-6, base_sat)
        verdict = "OK" if ratio > 0.9 else "*** 破損 ***"
        print("[%d]  %-38s %10.4f %8.4f   %s (彩度比 %.2f, %s)"
              % (i, name, sat, std, verdict, ratio, note))
    print()
    for i, t in enumerate(targets, 1):
        print("[%d] %s" % (i, t))


if __name__ == "__main__":
    main()
