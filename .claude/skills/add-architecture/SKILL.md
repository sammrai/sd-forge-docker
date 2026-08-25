---
name: add-architecture
description: Add support for a NEW model architecture (like Krea 2 or Z-Image) that forge cannot load, by routing it to the ComfyUI backend. Use when the user wants to 導入 / 対応させる / 使えるようにする a new arch, when a checkpoint fails with "unknown model type", or when a newly-supported arch produces broken images. Covers the router internals and where to edit, extracting correct workflow values from official templates, the sampler/resolution traps, and the compatibility checklist against the client.
---

# 新アーキの導入

既存アーキ(SD1.5/SDXL/Pony/Illustrious/Flux/Krea 2/Z-Image)のチェックポイントを足すだけなら
civitdl で落とすだけ。コード変更は要らない。**このスキルは新アーキ対応の作業用。**

forge が判別できるアーキは固定されている。

```bash
docker exec sd-forge-docker-sdui-1 sh -c 'ls /app/webui/backend/diffusion_engine/'
docker exec sd-forge-docker-sdui-1 sh -c \
  'grep "^class " /app/webui/repositories/huggingface_guess/huggingface_guess/model_list.py'
```

ここに無いアーキは ComfyUI へ振り分ける。

---

# 対象コードの構造

実体は **`comfy/router/scripts/comfy_router.py`**（このリポジトリ）。
sdui へは `./comfy/router:/app/data/extensions/comfy-router:ro` でバインドマウントされる。
**リポジトリ側を編集する。** `/data2/forge-data/extensions/` にコピーを置かない。

```
comfy_router.py
├── MODEL_SPECS[]              アーキごとの定義。★新アーキはここに1エントリ足す
├── SAMPLER_MAP / SCHEDULER_MAP  A1111 名 → comfy 名
├── TURBO_SAFE_SAMPLERS        蒸留モデルで実用になるサンプラー
├── RAW_MIN_STEPS / RAW_MIN_CFG  RAW に turbo 値が来たときのエラー閾値
│
├── _pick_spec(req)            checkpoint 名 → spec（振り分け）
├── _resolve_unet(spec, name)  指定名を comfy の実ファイルへ解決
│                              **見つからなければ黙って spec 既定にフォールバック**
├── _is_raw_variant(name)      蒸留前バリアントの検出（ファイル名の "raw" トークン）
├── _align(v, m=16)            解像度の丸め。**全アーキ一律16の倍数**
├── _hr_params()               enable_hr → 目標解像度の算出
├── _normalize_sampling()      steps/cfg の矯正（RAW ならエラー）
├── _normalize_sampler()       サンプラーの矯正
├── _wait_forge_idle()         forge の生成中に VRAM を剥がさないためのガード
│
├── _loaders()                 UNET/CLIP/VAE + LoRA + ModelSampling
├── _build_txt2img()           base（hr2pass 指定で HR の 2nd パスも同グラフに）
├── _build_inpaint()           ADetailer の inpaint
│
├── _submit_and_wait()         comfy へ投げ ws で進捗を受ける
├── _run_adetailer_unit()      検出→クロップ→inpaint→貼り戻し
├── _generate(req)             統括。base → HR → ADetailer
└── _wrap_txt2img(app)         /sdapi/v1/txt2img の endpoint を差し替え
```

## 処理の流れ

```
client → /sdapi/queue/txt2img → sd-queue のワーカー
                                    ↓ 同じ route.endpoint を呼ぶ
                              _wrap_txt2img の wrapped()
                                    ↓ _pick_spec が None なら forge ネイティブへ素通し
                              _generate()
                                    ↓ _free_forge_vram()
                              comfy /api/prompt → ws で進捗 → /api/history
                                    ↓ finally: _free_comfy_vram()
                              TextToImageResponse(images, parameters, info)
```

GPU は単一キューで直列化される。**`finally` の `_free_comfy_vram()` を通さないと
comfy が VRAM を保持し続け、次の forge ジョブが OOM で落ちる。**

---

# 設計判断: コードを書く前に決める2つ

**ここを飛ばすと、後で個別の不具合として1つずつ潰す羽目になる。**

## (1) 互換性はルーターが吸収する

**クライアントは改修しない。** 差分が出たらルーター側で吸収する。
クライアント改修を検討するのは、ルーター側で吸収不可能だと確認できてからのみ。
（詳しい責任分解は後半の E2E フェーズを参照）

## (2) `parameters` の意味論を宣言する

**`parameters` はその画像を実際に作った条件を表す。**

この一文を最初に置けば、矯正の反映・丸めの反映・HR 時の解像度は、個別の不具合ではなく
一つの設計として最初から揃う。

例外の規則も同時に決める。基準は「**クライアントがその値に合わせにいけるか**」。

- 合わせにいける（矯正）→ `parameters` に反映する
- 合わせにいけない（ルーター内部の固定値）→ 送信値のまま返し、実態は
  `info.extra_generation_params` で追わせる

後者を反映するとクライアントは合わせる手段がなく、group_hash の予測が永久に外れる。
具体的なキーの分類はチェックリスト B にある。

# 実装手順

**手順5・7・8 は実測を伴うので、先に反映（手順9）が要る。** 線形には進まない。

```
1〜4（調べて spec を書く）
   ↓
9（反映）              ← まずテンプレート値のまま入れて動かす
   ↓
5・7・8（実測で較正） ←┐
   ↓                    │ 変更のたびに再反映
9（反映）             ──┘
   ↓
検証 → E2E
```

生成は必ずキュー経由（`/sdapi/queue/txt2img`）で行うため、spec が反映されていないと
実測そのものができない。手順4まで終えたら一度手順9へ飛ぶこと。

## 1. ComfyUI の対応を確認

```bash
docker compose exec -T comfy sh -c \
  'ls /opt/venv/lib/python3.12/site-packages/comfyui_workflow_templates_json/templates/ | grep -i <arch>'
```

無ければ ComfyUI を上げる。**sdui の再起動とは別手順**。

```bash
# 1. Dockerfile.comfy の COMFY_REF を更新（v 付きのタグ: v0.33.3 → v0.34.0）
# 2. docker-compose.yml の image: も合わせる（v を落とす: sd-forge-docker-comfy:0.34.0）
#    build が旧タグを上書きするので動きはするが、ピンを正直に保つために更新する
docker compose build comfy

# GPU を sdui と共有しているので、生成中でないことを確認してから起動する
curl -s http://127.0.0.1:7680/sdapi/v1/progress | python3 -c \
  "import json,sys; print(json.load(sys.stdin)['state']['job'] or 'IDLE')"
docker compose up -d --no-deps comfy

docker compose exec -T comfy curl -s http://127.0.0.1:8188/api/object_info | python3 -c \
  "import json,sys; print('<新ノード名>' in json.load(sys.stdin))"
```

## 2. ワークフローの正解値をテンプレートから取り出す

**推測で書くと例外は出ずに静かに壊れる。** テンプレートは subgraph 形式。

```bash
docker compose exec -T comfy python3 -c "
import json
d=json.load(open('/opt/venv/lib/python3.12/site-packages/comfyui_workflow_templates_json/templates/<file>.json'))
for sg in d.get('definitions',{}).get('subgraphs',[]):
    for n in sg.get('nodes',[]): print(n['type'], n.get('widgets_values'))
    for l in sg.get('links',[]): print(' link', l)"
```

| 取る値 | Z-Image | Krea 2 |
|---|---|---|
| `CLIPLoader` の `type` | `lumina2` | `krea2` |
| latent ノード | `EmptySD3LatentImage` | `EmptyLatentImage` |
| `ModelSamplingAuraFlow` | shift=3 | なし |
| sampler / scheduler | `res_multistep` / `simple` | `euler` / `simple` |
| steps / cfg | 8 / 1.0 | 8 / 1.0 |
| negative | `ConditioningZeroOut` | 同左 |

引数名は実機で確認する。

```bash
docker compose exec -T comfy curl -s http://127.0.0.1:8188/api/object_info | python3 -c "
import json,sys; d=json.load(sys.stdin)
for n in ['UNETLoader','CLIPLoader','KSampler']: print(n, list(d[n]['input']['required'].keys()))"
```

## 3. 重みを揃える

チェックポイント本体は **civitdl 経由でクライアントが落とす**前提。手で置くと forge の
モデル一覧に載らず、クライアントの `parse_model_string` が落ちる。

ただし spec の `"unet"` に書く**既定ファイル**は自分で用意する。これは
**指定名が comfy の catalog で解決できなかった場合**のフォールバック。
逆に civitdl で落ちた実ファイルは catalog にヒットするので**そちらが優先される**
（ログに `unet override: <既定> -> <実ファイル>` が出る）。
置き場は共有ツリーの `models/Stable-diffusion`。
`extra_model_paths.yaml` の `a111` ブロックにより `diffusion_models` として見える。

```bash
docker compose exec -T comfy curl -s http://127.0.0.1:8188/api/models/diffusion_models | python3 -c \
  "import json,sys; [print(x) for x in json.load(sys.stdin) if '<arch>' in x.lower()]"
```

**`_resolve_unet` は catalog に無い名前を黙って spec 既定にフォールバックする。**
配置ミスに気づきにくいので、必ず上のコマンドで載ったことを確認する。

text encoder は Civitai に無いので HF から取る(`Comfy-Org/<model>` の
`split_files/text_encoders/`)。置き場は forge と共有しているモデルツリー(`models/text_encoder`, `models/VAE`)。
civitdl 管理下と同じ場所に置き、compose のマウントを増やさない。

- **fp8 版の TE があれば優先**。VRAM に収まるかで速度が2倍変わる
  （Z-Image: full 7.49GB で 65秒 → fp8 5.25GB で 31秒）
- 汎用的なファイル名(`ae.safetensors`)は衝突するのでリネームする

## 4. MODEL_SPECS にエントリを足す

```python
{
    "name": "<arch>",
    "pattern": re.compile(r"...", re.IGNORECASE),  # civitdl の命名にマッチする形
    "unet": "<既定の実ファイル名>",
    "clip": "<TE>", "clip_type": "<CLIPLoader の type>", "vae": "<VAE>",
    "latent_node": "EmptyLatentImage",             # or EmptySD3LatentImage
    "aura_shift": None,                            # or 3.0
    "sampler": "euler", "scheduler": "simple",
    "steps": 8, "cfg": 1.0,                        # turbo なら
    "hr_mode": "esrgan_2pass",                     # 手順8 で決める
    "hr_sampler": "dpmpp_2m_sde", "hr_scheduler": "beta", "hr_steps": 5,
    # upscale_models 配下に実在すること。**4x モデル前提**（_generate が
    # shrink = hr_scale / 4.0 で縮小率を逆算する）。2x を指定すると解像度がズレる
    "hr_upscale_model": "4x-UltraSharp.pth", "hr_denoise": 0.33,
}
```

### pattern の書き方

civitdl は `<モデル名>-mid_<id>-vid_<id>/<ファイル名>-mid_<id>-vid_<id>.safetensors`
という名前を付ける。クライアントが送る `sd_model_checkpoint` はこの stem。

```
krea2Turbo_v10-mid_2732656-vid_3072332
zImageTurbo_turbo-mid_2168935-vid_2442439
```

モデル名部分は作者依存で表記ゆれするが、**アーキ名は必ず含まれる**。
だから既存の pattern はアーキ名だけを、区切り文字のゆれを吸収する形で書いてある。

```python
r"krea\s*-?_?2"      # "Krea 2" "krea-2" "krea2" のいずれにもマッチ
r"z[_\- ]?image"     # "z_image" "z-image" "zImage" のいずれにも
```

**実際に civitdl で1本落として名前を確認してから書く。**
`MODEL_SPECS` は先頭から順に照合するので、他のアーキと衝突させない。

### spec に新しいキーを足す場合

参照箇所は11関数に分散している。全部洗う:

```bash
grep -n 'spec\[' comfy/router/scripts/comfy_router.py
```

## 5. サンプラーの安全性を確認

**最も事故りやすい。破綻したらまずここを疑う。**
8step 蒸留モデルに `dpmpp_2m`+`karras` を当てると顔が完全に潰れる（目が黒い塊になる）。
これを解像度や量子化の問題と誤診し、HR 方式の比較を丸ごとやり直した。

### 調べ方

同一プロンプト・同一シードで、**spec の既定 + 既存の安全リスト5種**を生成し、
顔を等倍で見る（手順は「自分での検証」）。潰れなければ安全。

```
euler / euler_ancestral / res_multistep / er_sde / lcm
```

comfy の生サンプラー名を `sampler_index` にそのまま渡せる（`_map_sampler_pair` が
A1111 名で引けなければ comfy 名として素通しする）。A1111 に対応名の無い
`res_multistep` などはこの形で指定する。

**安全リスト外の候補を試すときは、先に `TURBO_SAFE_SAMPLERS` へ仮追加して反映する。**
していないと `_normalize_sampler` が spec 既定へ黙って矯正するので、
「候補で生成したつもりが既定と同じ絵」を見て誤判定する（ログに `turbo normalize` が
出るので、生成後に必ず確認する）。

### TURBO_SAFE_SAMPLERS に足すときの注意

**この集合は全アーキ共通。** anima で安全なものを足すと、既存の Krea2 / Z-Image でも
そのサンプラーが素通しになる。逆に anima では危険だが既存アーキで安全なものは外せない。

アーキごとに分けたくなったら、集合を spec のキーへ移す（`_normalize_sampler` が
`spec` を受け取っているので改修は局所で済む）。

## 6. RAW（蒸留前）バリアントの命名を確認

`_is_raw_variant` は**ロードするファイル名を camelCase 分割して "raw" トークンを探す**
だけ。`krea2RawInt8Convrot` は拾えるが、非蒸留バリアントが `base` / `full` / `pruned`
などの命名だと**検出されない**。

検出されないと、RAW に turbo 用の steps/cfg が当たってもエラーにならず、
**サイレントに壊れた画像が返る**（このスキルが最悪と位置づけている事態）。

Civitai でバリアントの命名を確認し、"raw" 以外なら `_is_raw_variant` を拡張する。
非蒸留バリアントが存在しないアーキなら対応不要。

## 7. 解像度の制約を調べて実装する

公式/コミュニティの情報で「上限画素数」と「寸法の倍数制約」を確認する。
（Z-Image は総画素 1,048,576 が上限で32の倍数必須）

### 実装の現状を知っておくこと

**丸めは `_align(v, m=16)` の16の倍数のみ。spec にアラインメントや画素上限のキーは無い。**
つまり Z-Image の「32の倍数必須」は**コードのどこにも強制されていない**（実運用の解像度が
たまたま32の倍数だったため顕在化していない）。

新アーキの制約が16丸めで満たせない場合は、spec にキーを足して `_generate` の w/h 算出を
拡張する。該当箇所:

```python
w = _align(int(getattr(req, "width", 1024) or 1024))
h = _align(int(getattr(req, "height", 1024) or 1024))
```

## 8. hr_mode を決める

**表で機械的に決めず、実測で決める。**

`esrgan_2pass` を既定とし、同一プロンプト・同一シードで `direct` と両方生成して比較する。

| 結果 | 選択 |
|---|---|
| 画質が同等 | **`esrgan_2pass`**（方式が揃い spec の分岐が減る） |
| `esrgan_2pass` が破綻する | `direct` |
| `direct` が破綻する | `esrgan_2pass`（Z-Image がこれ。4Mpx で網目状に劣化する） |

Krea 2 は 4Mpx を直接生成できるが、実測（direct 346.0s / esrgan_2pass 300.6s、画質に
有意差なし）の結果 `esrgan_2pass` にしている。**「ネイティブに扱えるから direct」
という判断はしていない。**

| `hr_mode` | 内容 |
|---|---|
| `direct` | 目標解像度で1パス |
| `esrgan_2pass` | base → ESRGAN 拡大 → 低 denoise で焼き直し |

**拡大は必ずモデル拡大(ESRGAN)を経由させる。** Lanczos だと情報の無いぼけた画像を
渡すことになり鱗状の反復パターンが出る。latent の bislerp 拡大も破綻した。

## 9. 反映

拡張の編集だけなら `restart`、`docker-compose.yml` を変えたら `up -d`。
手順は `forge-restart` スキル。**再起動前に他セッションへ一報**（切れ目を掴んだ瞬間に
落とすので相手からは予告なく 502 が始まる）。

### 反映済みかを推測しない

バインドマウントなので**再起動した時点でディスク上の最新コードが読まれる**。
「まだ反映していない」と思い込んで不要な再起動や停止依頼をしたことがある。

```bash
docker exec sd-forge-docker-sdui-1 sh -c \
  'grep -n "<足した識別子>" /app/data/extensions/comfy-router/scripts/comfy_router.py'
```

「コード修正済み」と「稼働中に反映済み」を**区別して伝える**。

---

# 自分での検証

**必ず `/sdapi/queue/txt2img` 経由。comfy を直接叩かない。**
直接叩くと本番と並走して両方遅くなるうえ、スクリプトを kill しても comfy 側のジョブは
止まらず VRAM を保持し続け、本番が OOM で落ちる（実際に2回起こした）。
`/api/interrupt` も効かず `docker compose restart comfy` が必要になる。

**顔を等倍で見る。** 全体像やテクスチャだけ見て「良好」と判断しない。背景が綺麗でも
顔が潰れていることがある。網目状・鱗状の反復パターンはサンプラー不適合か解像度超過のサイン。

```bash
docker exec sd-forge-docker-sdui-1 bash -lc "cd /app/webui && python3 -c '
import sys; sys.path.insert(0, \"/app/data/extensions/adetailer\")
from adetailer.common import get_models
from adetailer.ultralytics import ultralytics_predict
from PIL import Image
m = get_models(\"/app/data/models/adetailer\")
p = ultralytics_predict(m[\"face_yolov8n.pt\"], Image.open(\"<path>\").convert(\"RGB\"),
                        confidence=0.3, device=\"cpu\")
print(p.bboxes)'"
```

---

# E2E フェーズ: クライアントとの互換性検証

自分の検証が通ったら、**クライアント側の実経路で E2E 検証する。
ここまでやって導入完了。**

**このフェーズにはクライアント側セッションの協力が要る。** 連絡先はユーザーに確認する。
セッションが起動していない場合は**自分での検証までを完了として報告し、E2E 未実施を
明記して終える**。単独で代替はできない（クライアントのコードを自分で動かす前提を置かない）。

クライアントは `normalize_adetailer_args` の位置引数整形や `build_generation_prompt` の
LoRA タグ付与を経由する。自分で組み立てたペイロードでの確認とは別物なので、
両方あって初めて担保になる。

## 準備: 使えるバリアントの一覧を作る

同一モデル内に spec の違うバリアントが混在する。`Krea 2 Turbo Official ...` の中に
RAW と Turbo の両方があり、**モデル名に "Turbo" と入っていてもバージョンは raw**。
クライアントからは見抜けないので、こちらが一覧を渡す。

```bash
curl -s "https://civitai.com/api/v1/models/{model_id}" | python3 -c "
import json,sys; d=json.load(sys.stdin)
for v in d['modelVersions']:
    print(v['id'], v['name'], '|', [f['name'] for f in v.get('files',[])])"
```

## 責任分解（ユーザーが明示した方針。動かさない）

クライアントは **IF どおり動くかをブラックボックス的にテストする側**で、互換の負担は
請け負わない。ユーザーの実発言（クライアント側セッションのログより）:

> このクライアント側での実装はなにもしない。
> 基本そのまま動作するのかという確認が基本
> こちらからはインタフェースの一致を追求すればいいだけです。
> 想定外の事や質問や不整合は先方に連絡して修正を依頼して。

| | 担当 |
|---|---|
| パラメータの矯正・変換・エコーバック | **ルーター** |
| 非互換が見つかったときの修正 | **ルーター** |
| 想定外・質問・不整合の起票 | クライアント → ルーター |
| 実経路での生成と結果の照合 | クライアント |
| モデル/LoRA の選定と civitdl での導入 | クライアント |
| クライアントのコード変更 | **誰もしない** |

**差分が出たらルーター側のバグ**、が既定の解釈。
ゴールは「クライアントから見て forge ネイティブと区別がつかない状態」。

一度この線を破り、クライアントへサンプラー分岐の実装を依頼した。
ユニットテスト18件まで書かせた後に方針違反と分かり全破棄させている。
**依頼する前に、ルーター側で吸収できないか必ず検討する。**

## 検証の厳密さ（ユーザーが明示した基準）

> **そこまで妥協なく確認できないと終われません。**

- **機能は「有無の差分」で確認する。** メタデータの申告を見て合格にしない
- **2モデル × 3機能 = 6パターン**（アーキが増えたら増える）
- **実データは Civitai が申告する設定をそのまま使う。** 手持ちモデルで代用しない
  （合成プロンプトの生成物を元画像にして「なぜ civitai の設定で生成しないの？」と
  指摘された。実データを流したからこそ RAW バリアント混入という本番バグが見つかった）
- **ユーザーの短い示唆は最小の読みで受ける。** 拡大解釈して計画を作り替える前に、
  現在の全体マトリクスを再掲して合意を取る
- **ダウンロードと生成を分離しない。** 通常の生成タスクの一部として civitdl が走る形で
  検証する（「ふつうに生成タスクとして dl も一緒にやるだけという意図」）。
  **分離した瞬間、テストしているのは本番と別の経路になる。**
  フェーズ分けや負荷の集約といった戦略を発明しない — 短い指示を拡大解釈して計画を
  作り替えると、二転三転して確認コストがユーザー側に発生する

## 報告の規律

**「合格」と言えるのは証拠グレードを満たしたときだけ。** 証拠なしの合格宣言は、
後で必ず撤回コストになる。

実際にこの会話で起きたこと: `info.extra_generation_params` にキーが出ただけで
「ADetailer 合格」「LoRA 合格」と報告し、ユーザーから

> civitaiのリンク２つ分も動作したという事？
> adetailer lora hrどれも動作したという事？
> civitaiリンク２枚 + adetailer lora hrどれも動作したという事？

と**言い方を変えて3回**問われ、3回とも「いえ、半分です」と答えることになった。
最後に「どう確認したの」と踏み込まれて申告を見ただけだと露呈し、合格を撤回して
やり直した。**報告が信用されていない状態を自分で作っている。**

守ること:

- 合格宣言には**証拠の種類を併記する**（申告のみ／実測、どのモデル・どの機能か）
- **未確認を聞かれる前に明示する。** 台帳を常に正確に保つ
- 計画を変更したら**マトリクス全体を再掲して抜けを確認する**。方針が揺れた際に
  実際に HR が検証項目から落ち、ユーザーに「hr adetailer lora も忘れないでください」
  と指摘された

同型の規律として「**コード修正済み**」と「**稼働中に反映済み**」も必ず区別する。

## 連絡

クライアント側のセッションへ `SendMessage` で連絡する。**セッション名はユーザーに確認する**
（`ListAgents` で一覧は出るが、本番環境用と開発環境用が並んでいることがあり、名前だけでは
判別できない）。
**今何をしていて次に何をするかをこまめに伝える。** 特に再起動前、自分の判断ミスに
気づいたとき。

**検証の停止・縮小はユーザーの承認事項。エージェント間の合意で確定させない。**
一度、自分から相手に「止めましょうか」と提案し、相手の受諾をもって全ジョブを止めた。
ユーザーからは「なぜ止めたの？」「どう言う要請？」と問われ、続けて
「そこまで妥協なく確認できないと終われません」と言われている。
**相手が選べば人間の頭越しに検証が止まる構図を作らない。**

依頼時に伝える内容:

```
<arch> をルーター経由で使えるようにしました。クライアント無改修での確認をお願いします。

## そちらの役割
インタフェースの一致を追求していただくだけで結構です。**不整合を見つけたら
返信してください。修正はこちらでやります。** クライアント側のコードは変更しないでください
（回避策を入れると、何が非互換なのか分からなくなります）。

## 振り分け
override_settings.sd_model_checkpoint が <pattern> にマッチしたら comfy へ。
クライアントの post_sdapi_queue_txt2img(= /sdapi/queue/txt2img)経由でも振り分けられます。
SDXL 等は従来経路のままです。

## 確認をお願いしたいこと
1. ネイティブ SDXL とのキー単位の機械照合(parameters / info のキー数・型・null)
2. クライアント無改修のまま、実際のプロンプト・LoRA・ADetailer 設定で生成
3. **下記の Civitai 画像を、通常の画像サンプリング経路(img2param)で流してください**
   <arch>: <Civitai 画像 URL>（LoRA 付きのものを選ぶ）
   HR あり / なしの両方でお願いします
4. response["info"] のパースとメタ保存(img2param / civitai2forge_param)が壊れないか
5. group_hash の予測と実測が一致するか
6. 不一致は見つかりしだい個別に報告してください（まとめてでなく）

3 は合成プロンプトでは出ない問題を拾うためです。実際に落札者が持ち込む経路そのもので、
checkpoint と LoRA を civitdl で取得するところから通してください。

## 機能の確認方法について
ADetailer / LoRA / HR は、**info.extra_generation_params に載っただけでは
確認になりません**（ルーターが「適用した」と申告しているだけです）。同一 seed で
有無を生成し、ピクセル差分を取ってください。ADetailer は顔だけ・LoRA は全面・
HR は base の単純拡大との差、という形で局在性まで見ていただけると確実です。

## 注意していただきたいこと
- モデル取得は1プロセスに集約してください。civitdl は --workers 1 で、同じモデルを
  複数プロセスから同時に要求すると固まります（並列生成が同じ新規モデルを指すと発生）
- ダウンロードの進捗は /civitdl/status ではなく .tmp のサイズで見てください。
  ダウンロード中に /status が応答しないのは正常です
- 大きなモデルのロード自体でディスクが飽和します(生成1本で %util 96%)
- group_hash の予測基底は実際の保存 meta と同じ構造(59キー)にしてください

## 使えるバリアント
<上で作った一覧。RAW など spec の違うものが混在する場合は必ず明示する>
```

## 渡す Civitai 画像の選び方

**アーキごとに1枚、LoRA が付いている画像**を選ぶ。Civitai の画像ページが申告する
リソース（checkpoint / LoRA）をクライアントがそのまま civitdl するので、**実データでしか
出ない問題**（RAW バリアントの申告、アーキ違いの LoRA、極端な生成パラメータ）を拾える。

```bash
curl -s "https://civitai.com/api/v1/images?limit=20&modelVersionId=<vid>" | python3 -c "
import json,sys
for i in json.load(sys.stdin)['items']:
    m = i.get('meta') or {}
    print(i['id'], m.get('sampler'), m.get('steps'), m.get('cfgScale'), m.get('Size'))"
```

## E2E チェックリスト

### A. 構造の一致（ネイティブ SDXL と機械照合）

- [ ] トップレベルが `["info", "parameters"]`
- [ ] `parameters` のキー数・キー名が一致（実績 59）／型・null の差分ゼロ
- [ ] `info` のキー数・キー名が一致（実績 32）／型・null の差分ゼロ

`info.sd_model_hash` のみ null 固定（comfy 側で未算出、クライアント未参照で合意済み）。

### B. `parameters` の意味論

- [ ] 矯正した値が反映される（steps / cfg_scale / sampler_index / sampler_name / scheduler / width / height）
- [ ] **送っていないキーは null のまま**（`sampler_index` だけ送って `sampler_name` が埋まらない）
- [ ] 内部固定値は送信値のまま（denoising_strength / hr_*）
- [ ] `seed: -1` は `parameters` に `-1`、`info.seed` に実値
- [ ] 丸めが反映される（`1928` → `1920`）
- [ ] **HR 時は1パス目の解像度**（`832x1216`+`hr_scale:2` → `parameters` は `832x1216`、実ファイル `1664x2432`）

基準は一貫して「モデルに入力した解像度」。丸めは入力が変わるので反映、HR の拡大は出力なので反映しない。

### C. 素通しと不動点

- [ ] spec 準拠の値なら矯正が起きず `parameters` が送信値と一致
- [ ] group_hash の事前予測と実測が一致
- [ ] `/params` で読み戻して再生成すると**ピクセル差 0**

最後が最重要。`parameters` に送信値を返していると読み戻し→再生成でまた矯正が走る往復になる。
**この検証項目はユーザーの問い**（「生成した画像を/paramで取得、再生成してもいっちするの？」）
が起点で、それまで未検証だった。

### D. 機能（クライアント実経路で）

- [ ] base / ADetailer / HR / LoRA
- [ ] **HR + ADetailer + LoRA 同時**（クライアントの既定形）
- [ ] `batch_size > 1` / LoRA 複数同時
- [ ] 画像サンプリング経路（Civitai 画像から `img2param`）

### E. 機能が効いている証明（差分の局在性）

**`info.extra_generation_params` に載っただけでは動作確認にならない。**
ルーターが「適用した」と申告しているだけで、絵が変わった証拠ではない。
同一 seed で有無を生成しピクセル差分を取る。

```python
d = np.abs(np.asarray(a, float) - np.asarray(b, float)).mean(axis=2)
mask = np.zeros(d.shape, bool); mask[y0:y1, x0:x1] = True   # face bbox
print(d[mask].mean(), d[~mask].mean(), (d > 8).mean() * 100)
```

- [ ] **ADetailer**: 顔だけ差分が大きく他はほぼゼロ（実績 13.13 / 0.01）
- [ ] **LoRA**: 全体が変化（実績 平均 17.58、変化画素 64.2%）
- [ ] **HR**: base を単純拡大したものと比較して差が出る（2パス目が描き直している証拠）

### F. アーキ不整合の検出

- [ ] 他アーキの LoRA を指定したらエラーで落ちる
- [ ] RAW バリアントに turbo 相当の値（`steps<=10` / `cfg<=1.5`）が来たらエラーで落ちる

壊れた画像を納品するくらいならタスク失敗のほうがまし、という判断。クライアントは
Civitai 画像のメタから steps/cfg をそのまま取るので、turbo 画像が RAW を申告していると
必ずこの組み合わせが来る。

---

# 502 が出たら切り分ける

```bash
curl -s -m 10 -o /dev/null -w "forge   %{http_code}\n" http://127.0.0.1:7680/sdapi/queue/login
curl -s -m 10 -o /dev/null -w "civitdl %{http_code}\n" http://127.0.0.1:7680/civitdl/models/
iostat -x 1 2 | grep "^sdb"                                       # %util 97% なら飽和
docker inspect -f '{{.State.StartedAt}}' sd-forge-docker-sdui-1   # 再起動直後か
```

実例: forge は 200 なのに 502 と報告された → civitdl のワーカー占有。別の回は
forge の再起動ウィンドウ。**決めつけない。**

---

# 完了条件

- [ ] `MODEL_SPECS` にエントリが入り、拡張がロードされている
      `docker logs sd-forge-docker-sdui-1 2>&1 | grep "comfy-router] wrapped"`
- [ ] キュー経由で base 生成が通り、顔を等倍で見て破綻がない
- [ ] クライアントの E2E チェックリスト A〜F がすべて合格
- [ ] 変更をコミット（**コミットは都度ユーザーの承認を取る**）
- [ ] 知見を memory に記録（アーキ固有の値、踏んだ罠、実測性能）
