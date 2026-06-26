#!/usr/bin/env python3
# Neo の HiRes 2パス目 denoise を環境変数 FORGE_HR_DENOISE_MULT で補正する微修正。
# comfy-kitchen バックエンドの前方計算が forge(lllyasviel) より滑らかで、同一 denoise だと
# 高周波ディテールが不足する。hires denoise を倍率補正して forge 相当のディテールに合わせる。
# 既定 1.0 = 無効（安全）。compose env で 1.16 等を指定して有効化。
import io, sys

PATH = "/app/neo/modules/processing.py"
src = io.open(PATH, encoding="utf-8").read()

needle = ("        samples = self.sampler.sample_img2img(self, samples, noise, self.hr_c, self.hr_uc, "
          "steps=self.hr_second_pass_steps or self.steps, image_conditioning=image_conditioning)")

if needle not in src:
    print("HR_DENOISE_PATCH: needle not found -- aborting (no change)", file=sys.stderr)
    sys.exit(1)

replacement = (
    "        import os as _oshr\n"
    "        _hrm = float(_oshr.environ.get('FORGE_HR_DENOISE_MULT', '1.0'))\n"
    "        _odn = self.denoising_strength\n"
    "        if _hrm != 1.0:\n"
    "            self.denoising_strength = min(_odn * _hrm, 0.999)\n"
    "            print(f'[HR_DENOISE_PATCH] hires denoise {_odn} -> {self.denoising_strength} (x{_hrm})')\n"
    "        samples = self.sampler.sample_img2img(self, samples, noise, self.hr_c, self.hr_uc, "
    "steps=self.hr_second_pass_steps or self.steps, image_conditioning=image_conditioning)\n"
    "        self.denoising_strength = _odn"
)

src = src.replace(needle, replacement, 1)

# TF32 無効化フック（env FORGE_DISABLE_TF32=1 で fp32演算を真の full precision に）。
# `from __future__` は必ずファイル先頭でなければならないので、その直後に挿入する。
tf32_hook = (
    "\nimport os as _ostf, torch as _ttf\n"
    "if _ostf.environ.get('FORGE_DISABLE_TF32', '0') == '1':\n"
    "    _ttf.backends.cudnn.allow_tf32 = False\n"
    "    _ttf.backends.cuda.matmul.allow_tf32 = False\n"
    "    try: _ttf.set_float32_matmul_precision('highest')\n"
    "    except Exception: pass\n"
    "    print('[TF32_PATCH] TF32 disabled (full fp32 precision)')\n"
)
future_line = "from __future__ import annotations"
if future_line in src:
    src = src.replace(future_line, future_line + "\n" + tf32_hook, 1)
else:
    src = tf32_hook + src

io.open(PATH, "w", encoding="utf-8").write(src)
print("HR_DENOISE_PATCH: applied (+TF32 hook)")
