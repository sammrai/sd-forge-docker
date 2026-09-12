---
name: vae-bake
description: Diagnose a broken baked-in VAE in a Stable Diffusion checkpoint and re-bake a known-good one in place. Use when generated images come out グレー / 彩度が低い / 眠い / 白っぽい / コントラストが出ない / 色が濁る, when the user asks whether a checkpoint has a VAE 焼き込まれているか, when a civitai model page says "a VAE is needed for better results", or when they want to VAE を焼き込む / 焼き直す / 差し替える / bake a VAE into a checkpoint. Bundles vae_bake.py plus round-trip and diff verification scripts. Covers the header-only structure check, why cosine similarity silently misjudges a broken VAE and round-trip measurement is the only reliable test, and verifying that only the VAE changed.
---

# チェックポイントの VAE 診断と焼き直し

「グレーで彩度が異常に低い」「眠い」「コントラストが出ない」は、**チェックポイントに
焼き込まれた VAE が壊れている**ときの典型症状。VAE マージで重みだけスケールされ bias が
対応していない状態になりやすく、civitai の説明に "a VAE is needed for better results" と
書いてあるモデルは大抵これ。

作業は `sdui` コンテナ内で行う（ホストの `python3` には numpy も torch も無い）。
`/data2/forge-data` が `/app/data` にマウントされている。

```bash
docker compose exec -T sdui python3 ...   # torch 2.4.0 / diffusers 0.31.0 / safetensors 0.4.2
```

## 1. 構造を見る（7GB でも一瞬）

safetensors のヘッダだけ読む。ホストで動く。

```python
import json, struct, collections
f = open(PATH, "rb"); n = struct.unpack("<Q", f.read(8))[0]
hdr = json.loads(f.read(n)); hdr.pop("__metadata__", None)
print(collections.Counter((k.split(".")[0], v["dtype"]) for k, v in hdr.items()))
```

SDXL なら `model` 1680 / `conditioner` 585 / **`first_stage_model` 248** が揃う。
248 キー（SD1.5 も同数）が無ければ VAE 未焼き込みか、Flux/Z-Image/Krea2 など別アーキ
（それらは VAE のキー構造が違うのでこのスキルの対象外）。

**キーが揃っていても「正常」の証明にはならない。** ここで止めずに次へ進む。

## 2. 判定はラウンドトリップ実測でやる

画像 → encode → decode させて、彩度とコントラストが保たれるかを測る。これが唯一
信頼できる判定。比較対象として `models/VAE/sdxl_vae.safetensors` と、正常だと分かって
いる他のチェックポイントを一緒に並べる。テスト画像はそのモデル自身のプレビュー
（`extra_data-vid_*/*.jpeg`）を使えばよい。

```bash
docker compose cp .claude/skills/vae-bake/scripts/vae_roundtrip.py sdui:/tmp/
docker compose exec -T sdui python3 /tmp/vae_roundtrip.py <画像> <ckpt or vae> [...]
```

実測例（壊れた ASTRAL [NoobAI] と、その焼き直し後）:

| | saturation | std |
|---|---|---|
| 入力画像 | 0.4918 | 0.2030 |
| **壊れた焼き込みVAE** | **0.2300** | **0.0767** |
| 焼き直し後 / `sdxl_vae.safetensors` | 0.4824 | 0.1988 |
| AutismMix・AnimeBoysXL（正常な対照） | 0.482〜0.487 | 0.199 |

正常なら彩度比 0.95 以上になる（スクリプトは 0.9 未満を破損と判定する）。
**0.5 前後まで落ちていたら破損。** 引数は checkpoint でも VAE 単体
ファイルでもよい（`first_stage_model.` があれば抜き出す）。encode / decode に
`scaling_factor` は効かないので SD1.5 と SDXL で config を分ける必要は無い。

### cos 類似度で判定してはいけない

テンソルごとの cos 類似度は**スケール不変**。VAE マージ破損は「重みの向きはそのまま、
ノルムだけ 0.29 倍、bias は対応せず」という形で起きるので、cos で見ると
`1.00000` が並び、**正常な fp16-fix 版と誤認する**。実際にこれで一度誤判定した。

- 壊れた ASTRAL vs 正常な AutismMix … cos 中央値 **0.99994**（ほぼ同一に見える）
- 同じペアのラウンドトリップ … 彩度 **0.2300 vs 0.4825**（別物）

ノルム比まで見るか、素直にラウンドトリップする。後者の方が速くて確実。

## 3. 焼き直す

同梱の `scripts/vae_bake.py` を使う。第1引数は**フォルダ名**（`name-mid_X-vid_Y` 形式）、
第2引数は `models/VAE/` 内のファイル名。

```bash
docker compose cp .claude/skills/vae-bake/scripts/vae_bake.py sdui:/tmp/
docker compose exec -T sdui python3 /tmp/vae_bake.py "<フォルダ名>" sdxl_vae.safetensors
```

同じものが Dockerfile で `/app/vae_bake.py` にも焼き込まれている（README の手順はこちら）。
**イメージ内のコピーは次の再ビルドまで古いまま**なので、スクリプトを直したら上の
`docker compose cp` で回す。後始末は不要 — 以下はすべてスクリプト側で処理される。

| 挙動 | 理由 |
|---|---|
| バックアップは `/app/data/model-backup/` | `models/` 配下に置くと **Forge が別モデルとして拾う**。作業用の一時ディレクトリ `/app/data/.vae-bake-tmp` も同じ理由で models 外（同一FSなので move は一瞬） |
| 既存バックアップがあれば中止 | 2回焼くと原本を上書きして復元不能になる |
| `.safetensors` 以外の sidecar を全部引き継ぐ | csv・`extra_data-vid_*/` のプレビューと model_dict は**原本を退避したら取り直せない** |
| 元ディレクトリの uid/gid を複製 | コンテナ内 root で書くため。他のモデルと所有者が揃わなくなる |
| VAE 側の `loss.*` と `model_ema.*` を除外 | 学習時の残骸。`first_stage_model.` 配下の unexpected key になる |

必要なもの: ディスク **+7GB**（/data2 は常に逼迫しているので `df -h /data2` を先に見る）、
RAM **15GB 程度**（全テンソルをメモリに載せて保存する）。`sdxl_vae.safetensors` は fp32
なので焼くと **+167MB**（6.94 → 7.11GB）になる。Forge は VAE を fp32 で回すので
fp16 に落とす必要は無い。

## 4. 検証する

**VAE 以外が1バイトも変わっていないこと**を確かめる。UNet と text encoder に手が
入っていたら焼き直しではなく別物になっている。

```bash
docker compose cp .claude/skills/vae-bake/scripts/verify_bake.py sdui:/tmp/
docker compose exec -T sdui python3 /tmp/verify_bake.py "<新ckpt>" "<バックアップckpt>"
```

`UNet/TE バイト列が不変 : True`、VAE キー数 248 → 248、余計なキーなし、を確認する。
そのうえで **2. のラウンドトリップをもう一度回して彩度が戻ったことを見る**。
「焼けた」と「直った」は別。

## 5. Forge への反映

ファイル差し替えだけなので**再起動は不要**。ただし対象がロード中だとメモリ上の古い VAE が
使われ続ける。

```bash
curl -s 127.0.0.1:7680/sdapi/v1/options | python3 -c \
  "import sys,json;print(json.load(sys.stdin)['sd_model_checkpoint'])"
```

これが差し替えたモデルなら、別モデルへ切り替えてから戻してもらう（Forge のハッシュ/メタデータ
キャッシュは mtime 比較で無効化される — `modules/hashes.py:46`, `modules/cache.py:110` — ので、
焼き直しで mtime が新しくなれば次回ロード時に自動で読み直される）。
他のセッションが生成中でも影響しないので、アイドル待ちも不要。
再起動が要る変更と混ざっている場合だけ `forge-restart` スキルへ。

## 環境メモ

- `models/VAE/` にある SDXL 用は `sdxl_vae.safetensors`（本家 fp32）のみ。
  `ae` / `qwen_image_vae` / `z_image_ae` は別アーキ用なので SDXL には焼けない
- 他の VAE（sdxl-vae-fp16-fix など）を使いたい場合は `models/VAE/` に置いてから同じ手順
- コンテナ名は compose のサービス名 `sdui`（実体は `sd-forge-docker-sdui-1`）
- Forge の API は traefik 経由で `127.0.0.1:7680`
