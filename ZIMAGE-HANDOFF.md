# Z-Image 統合 引き継ぎ資料（作業中断スナップショット）

最終更新: 2026-06-26
対象ブランチ: `zimage-wip`（このブランチ）
本番 main は安定版 `be5e8dc`（z-image 統合前 = 素の forge）へ巻き戻し済み。

---

## 0. このブランチは何か

`sammrai/sd-forge-docker:12.4.0` = lllyasviel forge(SDXL, torch2.5) と diffusers
Z-Image-Turbo を**同一プロセスに同居**させた統合イメージ。拡張
`zimage_diffusers_ext`(extensions-builtin/zimage-diffusers)が `/sdapi/v1/txt2img`
をラップし、`override_settings.sd_model_checkpoint` が正規表現 `z[_\- ]?image` に
マッチすれば diffusers `ZImagePipeline` へ、それ以外は forge ネイティブへ透過ルーティング。

含まれるコミット:
- `0e283ca` Integrate diffusers Z-Image-Turbo into the 12.4.0 forge image
- `76a09a4` Fix Z-Image HiRes OOM and add forge-compatible response/progress
- （このブランチの WIP コミット）max_seq cap + `_release_zimage_gpu` ハード解放化

GPU は RTX 3060 **12GB**(11.62GB usable)。transformer ~12GB / text_encoder(Qwen3) ~7.5GB /
vae ~160MB(いずれも bf16)。**大きい2モデルは同時常駐できない**のが全ての制約の根。

---

## 1. なぜ main を巻き戻したか（中断理由 = 未解決のクリティカル不具合）

**症状**: Civitai タスクが `RuntimeError: Unknown error` で「ずっと」失敗する。
forge ネイティブ SDXL(jibMixRealisticXL)タスクすら失敗する。

**根本原因（2026-06-26 ログ解析で確定）**:
1. Z-Image 生成が OOM で途中クラッシュすると、**約11.5GB が GPU に張り付いたまま戻らない**。
   - ログ: `OutOfMemoryError('CUDA out of memory. Tried to allocate 20.00 MiB ...
     11.54 GiB memory in use')` が3連続。＝生成開始前から GPU 満杯。
2. `_release_zimage_gpu()` は「released」とログするのに**実際は解放できていない**
   （コンポーネント単位の `.to('cpu')` / group offload 再適用ではクラッシュ残骸が剥がれない。
   Python 参照が残ると `empty_cache()` も効かない）。
3. 残骸が居座るため、後続の **forge ネイティブ SDXL** も空き1GB前後の劣化モード
   （"Low GPU VRAM Warning" / "tiled VAE encoding" フォールバック, 約10倍遅）で動き、
   最終的に hires VAE で OOM → `RuntimeError`。**Z-Image も SDXL も全部巻き込まれる**。

確認コマンド: `nvidia-smi` でコンテナ pid が 11018MiB/12288MiB 占有を確認済。

→ 本番を不安定なまま放置できないので、安定版 `be5e8dc` に戻し、この不具合を腰を据えて
直してから再投入する方針。

---

## 2. このブランチで適用済みの修正（未デプロイ・未検証）

`zimage_diffusers_ext/scripts/zimage_api.py`:

### (a) `_encode()` に max_sequence_length cap
512 固定だと padding で encode 活性化が ~3.8G に膨らみ、TE 一括 GPU ロード(7.7G)と
合わせて OOM する。実トークン長 +16 に詰めて活性化を ~1/3 に。
```python
n_p = len(pipe.tokenizer(prompt or "").input_ids)
n_n = len(pipe.tokenizer(neg).input_ids) if neg else 0
max_seq = min(512, max(64, max(n_p, n_n) + 16))
# encode_prompt(..., max_sequence_length=max_seq)
```

### (b) `_release_zimage_gpu()` をハード解放にフォールバック化（★本命修正・未検証）
ソフト解放(コンポーネント CPU 退避)後に `mem_get_info()` で空きを確認し、6GB 未満なら
`_hard_reset_pipe()` でパイプ globals(`_pipe/_img2img/_inpaint`)を None にして
全参照を断ち、`gc.collect()+empty_cache()+synchronize()` で実メモリを解放する。
次回リクエストで `from_pretrained` 再ロード(ディスクキャッシュ済・数十秒)。
クラッシュ時のみ発生するコストなので許容。

```python
def _hard_reset_pipe():
    global _pipe, _img2img, _inpaint
    _pipe = _img2img = _inpaint = None
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
```

**注意**: `import gc` を追加済み。この (b) は**ログ上の挙動を本番/隔離で未検証**。
次の作業者はまずここを検証すること。

---

## 3. これまでに確定した知見（再発明防止・無駄足回避）

### offload
- transformer の **group offload は正常動作**（隔離テスト: base 53s / hires 167s, 残留0.16G）。
  `transformer.enable_group_offload(offload_type="block_level", num_blocks_per_group=1,
  use_stream=True, record_stream=False)`。
- **`record_stream=True` は有害**: prefetch したブロックの解放が遅れ runaway → OOM。必ず False。
- **text_encoder(transformers Qwen3)の group offload は壊れている**: onload はするが
  offload されず OOM（use_stream True/False どちらも）。→ TE は group offload しない。
- **TE は CPU 常駐 + encode 時だけ丸ごと GPU swap** で運用（`pipe.text_encoder.to("cuda")`→
  encode→`.to("cpu")`）。CPU encode は bf16 で約38分(2263s)と病的に遅く却下。
- vae は常駐（bf16 キャスト必須・`enable_tiling()/enable_slicing()` で hires OOM 回避）。

### LoRA（動作する・forge 資産を再利用）
- パース: `modules.extra_networks.parse_prompt(text)` → `data["lora"]` の ExtraNetworkParams。
- 解決: `networks.available_networks` / `available_network_aliases`(NetworkOnDisk.filename)。
- 適用: `pipe.load_lora_weights(path, adapter_name=...)` + `pipe.set_adapters(names, weights)`。
- diffusers `lora_state_dict` が `diffusion_model.`→`transformer.` を自動変換（480 keys, 全 transformer, TE LoRA 無し）。
- **group offload + LoRA は両立する**（diffusers が load 後に
  `_maybe_remove_and_reapply_group_offloading` を自動再適用）。**fuse/焼き込みは不採用**
  （遅い・バグ温床）。

### ADetailer
- 方針転換で **ADetailer は不要**（env `ZIMAGE_ADETAILER=1` のときだけ動く既定OFF実装は残置）。
  person inpaint がフル解像度パスになり +約960s でクライアント idle timeout(6分)を誘発するため。
- 実装は forge の adetailer 資産(ultralytics_predict, mask_preprocess 等)を流用済み。

### クライアント timeout（`/app/src/lib/forge.py` `_gen`）
- **idle ベース**: `while idle < 60*6`(360s)。idle は status=="pending" か progress 変化でリセット。
  総時間ではない。進捗を出さないフェーズ(encode/model-load/adetailer)が6分超で timeout。

### 環境
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（断片化 OOM 緩和）。
- Z-Image 開始前に `_free_forge_vram()`（`backend.memory_management.unload_all_models()` +
  `soft_empty_cache(force=True)`）で forge 常駐 SDXL を退避。
- `ZIMAGE_OFFLOAD` env 既定 `group`（他 sequential/model/none）。

---

## 4. 次にやること（推奨順）

1. **(b) ハード解放の検証**。`neo-verify/` の隔離テスト(zoff_test.py/zenc_test.py)で
   「OOM を意図的に起こす → `_release_zimage_gpu` 相当 → `mem_get_info` で空きが戻るか」を確認。
   本番 GPU でユーザーが生成中はテスト禁止（GPU 競合）。
2. (a) max_seq cap がクリーン GPU で TE 一括 encode を収めるか確認（zenc_test.py）。
3. 上記2点が OK なら rebuild + recreate で再デプロイ（hot-patch/docker exec 改変は禁止、
   必ず rebuild+recreate）。Civitai #133666368 で base+hires+LoRA(ADetailer なし)を通す。
4. forge ネイティブ SDXL が無影響で動くことを確認してから main へマージ。

---

## 5. 隔離テスト資産（neo-verify/）
- `zoff_test.py`: pipe ロード→group offload→LoRA2枚→base+hires。各段で VRAM ログ。
- `zenc_test.py`: encode のみ。capped vs full512 max_seq を実プロンプトで比較。
- 実行は GPU が空いている時のみ（`nvidia-smi` で空き>11000MiB を確認してから）。

## 6. 関連メモリ
- `forge25-unified-image`（構成とハマりどころ）
- `zimage-turbo-models`（モデル一式と生成パラメータ）
- `neo-migration-config-blocker`（Neo は精細度低下で不採用）
