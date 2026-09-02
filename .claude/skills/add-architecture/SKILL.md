---
name: add-architecture
description: Add support for a NEW model architecture (like Krea 2 or Z-Image) that forge cannot load, by routing it to the ComfyUI backend. Use when the user wants to 導入 / 対応させる / 使えるようにする a new arch, when a checkpoint fails with "unknown model type", or when a newly-supported arch produces broken images. Covers the router internals and where to edit, extracting correct workflow values from official templates, the resolution and HR traps, where the router must stay out of the client's parameter choices, and the compatibility checklist against the client.
---

# 新アーキの導入

既存アーキ(SD1.5/SDXL/Pony/Illustrious/Flux/Krea 2/Z-Image)のチェックポイントを足すだけなら
civitdl で落とすだけ。コード変更は要らない。**このスキルは新アーキ対応の作業用。**

forge が判別できるアーキは固定されている。

```bash
docker compose exec -T sdui sh -c 'ls /app/webui/backend/diffusion_engine/'
docker compose exec -T sdui sh -c \
  'grep "^class " /app/webui/repositories/huggingface_guess/huggingface_guess/model_list.py'
```

ここに無いアーキは ComfyUI へ振り分ける。

---

# 対象コードの構造

実体は **`comfy/router/scripts/comfy_router.py`**（このリポジトリ）。
**このファイルはイメージに焼き込まれる**（`Dockerfile` の `COPY comfy/router /opt/comfy-router`）。
ENTRYPOINT が起動のたびに `rm -rf` してから `/app/data/extensions/` へ配置し直すので、
コンテナ内を直接編集しても次の起動で消える。**必ずリポジトリ側を編集する。**

```
comfy_router.py
├── MODEL_SPECS[]              アーキごとの定義。★新アーキはここに1エントリ足す
├── SAMPLER_MAP / SCHEDULER_MAP  A1111 名 → comfy 名
│
├── _pick_spec(req)            checkpoint の中身 → spec（振り分け）
│                              判定は comfy の /router/detect_arch に聞く
│                              （comfy/custom_nodes/arch_detect.py）
├── _resolve_unet(spec, name)  指定名を comfy の実ファイルへ解決
│                              **解決できなければエラー（既定へ落とさない）**
├── _map_sampler_pair()        サンプラー/スケジューラ名の解決。解決不能はエラー
├── _align(v, m=16)            解像度の丸め。**全アーキ一律16の倍数**
├── _hr_params() / _hr2pass()  enable_hr → 目標解像度と 2nd パス設定
├── _sampling()                steps/cfg（送信値。未送信時のみ spec 既定）
├── _wait_forge_idle()         forge の生成中に VRAM を剥がさないためのガード
├── _install_lazy_release()    forge の VRAM 要求時に comfy を解放するフック
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
                                    ↓ finally: 解放予定にする(即解放はしない)
                              TextToImageResponse(images, parameters, info)
```

GPU は単一キューで直列化される。comfy は生成後もモデルを VRAM に保持し続けるので、
**解放しないと次の forge ジョブが OOM で落ちる。**

### ただし生成ごとに解放してはいけない

**解放は forge が実際に VRAM を要求するまで遅らせる。** comfy 生成の `finally` で毎回
解放すると、comfy がステージした重み(実測 12.5GB)ごと捨てられ、次の生成が必ず
ディスクからの読み直しになる。**同じモデルの連続生成で1枚あたり 150秒の差が出た**
(237s → 51s)。

```python
finally:
    _comfy_holds_vram = True          # 解放予定にするだけ

# forge が GPU にモデルを載せる唯一の入口をフックする
memory_management.load_models_gpu = patched   # 中で _release_comfy_if_held()
```

`load_models_gpu` を選ぶのは、ルーターが包んでいない経路(img2img など)も通るため。
txt2img の native 分岐に置くと取りこぼして OOM する。フックに失敗したら従来どおり
即解放にフォールバックさせ、壊れる方向へ倒れないようにする。

---

# 設計判断: コードを書く前に決める2つ

**ここを飛ばすと、後で個別の不具合として1つずつ潰す羽目になる。**

## (1) 吸収するのは I/F。パラメータの値ではない

**この線引きが全体の前提。** ここを取り違えると、後で全部やり直しになる。

| | 担当 | 例 |
|---|---|---|
| **I/F**（リクエスト/レスポンスの形） | **ルーターが吸収する** | `parameters` のキー・型・null、`info` のキー、`sd_model_hash`、解像度の丸め、`n_iter`、interrupt |
| **パラメータの値**（どう描くか） | **クライアントの責任** | steps / cfg_scale / sampler / scheduler / ad_prompt / HR の 2nd パス（`denoising_strength` / `hr_upscaler` / `hr_sampler_name` / `hr_scheduler` / `hr_second_pass_steps` / `hr_cfg`） |

I/F は**内部で何を経由してもよい。入出力が整合していればよい。**
例: latent の16px制約は実装都合なのでクライアントは知らなくてよい。丸めた解像度で
生成し、出力を要求解像度へ戻す。`1928` を送れば `1928` が返る。

値は**書き換えない**。書き換えると EXIF に送っていない値が残り、クライアントから
制御する手段も無くなる。**spec の既定値は「送られてこなかったときに使う値」であって、
送信値を上書きしてよい根拠ではない。**

### 不適合な入力: 値域は通す、名前はエラー

**この2つを混同しない。** 扱いが逆になる。

| | 扱い |
|---|---|
| **値域**が不適合（名前は解決できるが、このモデルに合わない値） | **そのまま使う。破綻してよい** |
| **名前**が解決できない（checkpoint / LoRA / サンプラー / スケジューラ / アップスケーラ / 付加処理のモデル） | **エラーで落とす** |

値域が変なのは「クライアントがそう決めた」と解釈できる。名前は解釈しようがない。
迷ったら **「これが起きたことをクライアントは検出できるか」** で決める。既定へ落とすと
`parameters` には送信値が返るので検出できない。これがこのスキルの言う最悪の事態。

### ただし、落とす範囲はネイティブが落ちる範囲を超えない

**ネイティブが黙って通すものを、この経路だけエラーにしない。** 同じ入力が経路によって
成功したり失敗したりすると、クライアントは経路ごとの分岐を持つことになり、互換の目標が
崩れる。ネイティブの挙動が silent failure に見えても、**それを直すのはこの経路の仕事では
ない**。直すなら両経路そろえる。

**迷ったらネイティブの実装を読む。** 「検出できるか」で決めるのは、ネイティブが落ちる
範囲の中だけ。境界は「名前が解決できるか」であって「効くか」ではない。

| | ネイティブ | この経路 |
|---|---|---|
| 名前が解決できない | 失敗する | **エラー**(メッセージを読めるものにするだけ) |
| 名前は解決できるが効かない | 無視して生成を続ける | **同じく続ける** |

### ルーターはモデルの中身を知らない

**値域チェックは書かない。例外も作らない。** 蒸留の有無、蒸留時の cfg、推奨 step 数、
どのサンプラーが破綻するか、バリアントごとの適正レンジ——**モデル固有の知識を
ルーターに持たせない。**

「この組み合わせは絶対に壊れるから落とす」は必ず失敗する。壊れる境界の判断はモデルの
性質に依存し、その知識はすぐ古くなる。実際に、モデル名からバリアントを推測して
steps/cfg のレンジで弾く実装を入れたところ、**正当な組み合わせを弾いてタスクを失敗させた**。

同じ理由で、モデルの素性をファイル名から推測して挙動を変えることもしない
（`info` のラベルに使うのも同じ。推測に基づく記録は嘘をつく）。

推奨値はクライアント側が持つ。ルーターは**要求されたパラメータで生成することだけに
注力する。**

## (2) `parameters` は送信値をそのまま返す

書き換えないので **送信値 = 実際に使った値 = 返却値**。`params = dict(raw)` でよい。

- **送っていないキーは埋めない。** 埋めるとネイティブと差分が出る
  （`sampler_index` だけ送ったのに `sampler_name` まで埋まる、など）
- 実際に使った値は `info.extra_generation_params` にも載せる。ここは送信値のエコーでは
  なく**実測の記録**なので、内部で解決した名前（`hr_upscaler` の実ファイル名など）を書く

# 実装手順

**手順5・6・7 は実測を伴うので、先に反映（手順8）が要る。** 線形には進まない。

```
1〜4（調べて spec を書く）
   ↓
8（反映）              ← まずテンプレート値のまま入れて動かす
   ↓
5・6・7（実測で較正） ←┐
   ↓                    │ 変更のたびに再反映
8（反映）             ──┘
   ↓
検証 → E2E
```

生成は必ずキュー経由（`/sdapi/queue/txt2img`）で行うため、spec が反映されていないと
実測そのものができない。手順4まで終えたら一度手順8へ飛ぶこと。

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
    "hr_mode": "esrgan_2pass",                     # 手順7 で決める
    "hr_sampler": "dpmpp_2m_sde", "hr_scheduler": "beta", "hr_steps": 5,
    # upscale_models 配下に実在すること。倍率は問わない
    # （拡大後に目標解像度へリサイズするので 2x でも 4x でも合う）
    "hr_upscale_model": "4x-UltraSharp.pth", "hr_denoise": 0.33,
}
```

### 振り分けは名前ではなく中身で決まる

`comfy_arch` には **comfy の `model_config` クラス名**を書く(`Anima` / `Krea2` /
`ZImage` など。`supported_models.py` の `class` 名)。`_pick_spec` は comfy の
`/router/detect_arch` に判定させ、返る `mro` にこの名前が含まれる spec を選ぶ。
`mro` で見るのは派生クラスを親クラス名で拾うため。

**アーキ名がファイル名に入っている保証はない。** 手元の 278 件で照合したところ、
アーキ名を含まないマージが 7 件あり、いずれも forge へ流れて
"Failed to recognize model type!" で失敗していた。名前で振り分けると
**どちらの向きでも静かに壊れる**(中身が新アーキなのに forge へ / 別アーキなのに comfy へ)。

`pattern` は**判定できなかったときのフォールバック**として残す(comfy が落ちている、
ファイルがまだ comfy から見えない)。書き方はアーキ名だけを、区切り文字のゆれを
吸収する形にする。

```python
r"krea\s*-?_?2"      # "Krea 2" "krea-2" "krea2" のいずれにもマッチ
r"z[_\- ]?image"     # "z_image" "z-image" "zImage" のいずれにも
```

**判定できているのにどの spec にも当たらない場合は、名前パターンに落とさない。**
そこで落とすと、名前だけ似た別アーキを comfy へ送ってしまう。

判定は `comfy.model_detection` に任せるので、**新アーキで手を入れるのは
`comfy_arch` の1行だけ**。導入前に手元の全チェックポイントで判定を回し、
Civitai の申告 baseModel と食い違わないかを確認する。

### spec に新しいキーを足す場合

参照箇所は11関数に分散している。全部洗う:

```bash
grep -n 'spec\[' comfy/router/scripts/comfy_router.py
```

**リクエストのフィールドと対になるキーなら、API 側の既定値を先に確認する。** forge の
API モデルは多くのフィールドを既定値で埋めるため、「送られてこなかった」を `None` / `0`
では判定できない。埋まるフィールドでは spec のフォールバックに到達せず、**spec を
調整しても絵は変わらない。** 到達しないと分かったらフォールバックを消す。

```bash
docker compose exec -T sdui grep -n '"key": "<field>"' /app/webui/modules/api/models.py
```

## 5. spec の既定値を実測で決める

spec の `sampler` / `scheduler` / `steps` / `cfg` / `hr_*` は、**クライアントが送って
こなかったときにだけ使う値**。生成の良し悪しをルーターが決める値ではない
（設計判断(1)の「モデルの中身を知らない」を参照）。

同一プロンプト・同一シードで数種を生成して選ぶ（手順は「自分での検証」）。
**テンプレートの値をそのまま持ち込まない。** step 数もモデルも前提が違うため、
そのまま使うと破綻することがある。

送信値が必ず埋まるフィールドでは、この既定値には**到達しない**。手順4の
「API 側の既定値を先に確認する」を必ず通すこと。

### 到達可能性は**稼働中のプロセス**に聞く。ソースやコメントで判断しない

既定値を書く前に、そのキーに到達し得るかを確認する。API モデルが埋めるフィールドの
フォールバックは死にコードで、いくら調整しても絵は変わらない。

```bash
curl -s http://127.0.0.1:7680/openapi.json | python3 -c "
import json,sys
s=json.load(sys.stdin)['components']['schemas']['StableDiffusionProcessingTxt2Img']['properties']
for k in ['<対になるリクエストのフィールド>']: print(k, s.get(k,{}).get('default'))"
```

既定が `None` なら到達する。値が入っていれば到達しない。

**リポジトリ内の記述を根拠にしない。** クラス定義もコメントも実際の既定とずれる
(`_hr2pass` の docstring は「0.75 が必ず入る」と書いていたが実際は `None` だった。
サブクラスの既定がモデル生成器に拾われていなかった)。

**推奨値の置き場はクライアント側のスキーマ**であって spec ではない。spec に書くのは
「クライアントが送りようがない値」(TE/VAE のファイル名など)だけと考えてよい。

### サンプラー名の解決（これは I/F の話）

`_map_sampler_pair` は comfy 名 → A1111 名の順に引き、どちらでも引けなければ末尾の
トークンを scheduler として分割する（`"res_multistep simple"` → `res_multistep` +
`simple`）。A1111 の `"DPM++ 2M Karras"` と同じ書き方で、comfy 名にも適用できるように
してある。それでも解決できなければ**エラー**。

名前の形を吸収するのは I/F の仕事。**どのサンプラーが良いかの判断とは別**で、
解決さえできればどんな組み合わせでもそのまま使う。

## 6. 解像度の制約を調べて実装する

公式/コミュニティの情報で「上限画素数」と「寸法の倍数制約」を確認する。
（Z-Image は総画素 1,048,576 が上限で32の倍数必須）

### 実装の現状を知っておくこと

**丸めは `_align(v, m=16)` の16の倍数のみ。spec にアラインメントや画素上限のキーは無い。**
つまり Z-Image の「32の倍数必須」は**コードのどこにも強制されていない**（実運用の解像度が
たまたま32の倍数だったため顕在化していない）。

新アーキの制約が16丸めで満たせない場合は、spec にキーを足して `_generate` の w/h 算出を
拡張する。該当箇所:

```python
req_w = int(getattr(req, "width", 1024) or 1024)
req_h = int(getattr(req, "height", 1024) or 1024)
w, h = _align(req_w), _align(req_h)      # 生成は丸めた解像度で
...
if (w, h) != (req_w, req_h):             # 返す前に要求解像度へ戻す
```

## 7. hr_mode を決める

**表で機械的に決めず、実測で決める。**

`esrgan_2pass` を既定とし、同一プロンプト・同一シードで `direct` と両方生成して比較する。

| 結果 | 選択 |
|---|---|
| 画質が同等 | **`esrgan_2pass`**（方式が揃い spec の分岐が減る） |
| `esrgan_2pass` が破綻する | `direct` |
| `direct` が破綻する | `esrgan_2pass`（高解像度で網目状に劣化するアーキがある） |

**「ネイティブに高解像度を扱えるから direct」という決め方はしない。** 直接生成できる
アーキでも、同一シードで比べて画質差が無く速度も 2パスが上、という結果が出ている。

| `hr_mode` | 内容 |
|---|---|
| `direct` | 目標解像度で1パス |
| `esrgan_2pass` | base → ESRGAN 拡大 → 低 denoise で焼き直し |

**spec の既定（`hr_upscale_model`）はモデル拡大(ESRGAN)にする。** Lanczos だと情報の
無いぼけた画像を渡すことになり鱗状の反復パターンが出る。latent の bislerp 拡大も破綻した。

あくまで**既定値**であって、2nd パスの拡大方式・sampler・steps をクライアントが
送ってきたらそのまま使う。

## 8. 反映

**ルーターはイメージ同梱なので、`restart` では反映されない。再ビルドとコンテナ再作成が要る。**
手順は `forge-restart` スキル。**再起動前に他セッションへ一報**（切れ目を掴んだ瞬間に
落とすので相手からは予告なく 502 が始まる）。

### 反映済みかを推測しない

「コードを直した」と「稼働中のプロセスがそれを読んでいる」は別。ここを混同して、
反映されていないコードを相手にデバッグし続けたことがある。**必ず実物を確認する。**

```bash
docker compose exec -T sdui sh -c \
  'grep -c "<足した識別子>" /app/data/extensions/comfy-router/scripts/comfy_router.py'
```

ただしファイルが新しくても**プロセスは古いことがある**。Python は起動時に import する
ので、コンテナの起動時刻がファイルの mtime より古ければ、そのコードは動いていない。

```bash
docker inspect -f '{{.State.StartedAt}}' $(docker compose ps -q sdui)
stat -c %y comfy/router/scripts/comfy_router.py
```

ログに出る固有の文言が現ファイルと食い違っていないかも有効な判定材料になる。

「コード修正済み」と「稼働中に反映済み」を**区別して伝える**。

### 反映は「本当に要るか」を数えてから

本番共有インスタンスなので、再起動は他者のジョブとキューを落とす。手順は
`forge-restart` スキル。このフェーズ固有の判断は3つ。

- **その変更は今のトラフィックに効くか。** クライアントが送っているキーの spec 既定を
  変えても何も起きない。効かない変更のために断を入れない
- **断の承認はユーザーのもの。** 相手セッションは本番を止める権限を持たない。
  相手が「いいですよ」と言っても、それは承認ではない
- **未コミットのローカルビルドが動いている間は `pull` できない**(自分の変更が消える)。
  タグが CI 産と区別できない状態であることを、伝えずに放置しない

---

# 自分での検証

**必ず `/sdapi/queue/txt2img` 経由。comfy を直接叩かない。**
直接叩くと本番と並走して両方遅くなるうえ、スクリプトを kill しても comfy 側のジョブは
止まらず VRAM を保持し続け、本番が OOM で落ちる（実際に2回起こした）。
`/api/interrupt` も効かず `docker compose restart comfy` が必要になる。

**顔を等倍で見る。** 全体像やテクスチャだけ見て「良好」と判断しない。背景が綺麗でも
顔が潰れていることがある。網目状・鱗状の反復パターンはサンプラー不適合か解像度超過のサイン。

```bash
docker compose exec -T sdui bash -lc "cd /app/webui && python3 -c '
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

## 準備1: 使えるバリアントの一覧を作る

同一モデル内に spec の違うバリアントが混在する。`Krea 2 Turbo Official ...` の中に
RAW と Turbo の両方があり、**モデル名に "Turbo" と入っていてもバージョンは raw**。
クライアントからは見抜けないので、こちらが一覧を渡す。

```bash
curl -s "https://civitai.com/api/v1/models/{model_id}" | python3 -c "
import json,sys; d=json.load(sys.stdin)
for v in d['modelVersions']:
    print(v['id'], v['name'], '|', [f['name'] for f in v.get('files',[])])"
```

`v.get('description')` も読む。蒸留の有無と推奨 step/cfg がそこにあることが多い。

### 一覧ではなく「そのまま貼れる行」を渡す

**相手が判断できない情報は、判断材料ではなく成果物の形で渡す。** 一覧を提示して
止めると「どれを使いどれを読み替えるか」の判断が相手に残り、そこで詰まる。

運用しないバリアントがあるなら `checkpoint_proxy.yml` に貼れる形で渡す。

```yaml
proxies:              # 使わない版 -> 実際に動かす版
  "<非蒸留の stem>": "<蒸留版の stem>"
base_model_fallback:  # Checkpoint リソースを持たない画像用
  "<Civitai の baseModel ラベル>": "<実際に動かす版の stem>"
```

`base_model_fallback` のキーは `/api/v1/model-versions/<vid>` の `baseModel` で
**確認して**書く。**未定義だと該当画像が `ValueError` で落ちる**ため、渡し忘れると
相手が実データで踏む。

## 準備2: 推奨パラメータを調べる(3ソース。1つで済ませない)

**クライアントが持つ既定値の根拠になる。** ルーターは値を書き換えないので、ここを
サボると「動くが絵が破綻する」状態でクライアントに渡すことになる。

| ソース | 取り方 | 何が分かるか |
|---|---|---|
| ① 上流のモデルカード | `curl -sL https://huggingface.co/<repo>/resolve/main/README.md` | アーキ全体の推奨。**バリアント別の step/cfg**、対応解像度、サンプラー、プロンプト規約 |
| ② チェックポイント作者の Civitai 説明 | 手元の `extra_data-vid_*/model_dict-*.json` の `description` / `modelVersions[].description`(HTML なのでタグを剥がす) | そのチェックポイント固有の推奨。**HR 倍率の上限**やアップスケーラ名など、上流に無い情報 |
| ③ 実際の画像の申告値 | `https://civitai.com/api/trpc/image.getGenerationData?input={"json":{"id":<image_id>}}` | 実運用の値と**宣言リソース(checkpoint / LoRA)**。v1 の `/api/v1/images` は `meta: null` を返すので使えない |

**3つが食い違ったら、バリアントが併存しているサイン。** ①がバリアント別に数値を
分けていないか、③が①のどのレンジに当たるかを見る。併存するなら、スキーマを1エントリに
するか2エントリにするかが設計判断になる(下記)。②にしか無い情報がよくある
(HR 倍率の上限、推奨アップスケーラ)ので①だけで済ませない。

## 準備3: クライアントへ渡すスキーマを作る

**「対応しました、確認してください」だけでは足りない。** クライアントはアーキごとの
既定値と値域を YAML で持つ。**定義が無いと受け皿(`match: '.*'` の forge 用エントリ)に
落ち、comfy が解決できない値を送ることになる**ので、検証以前に成立しない。

### クライアント側で触るファイル(名指しで伝える)

| ファイル | 何を足すか |
|---|---|
| `src/generation_defaults.yaml` | **アーキのエントリ**。`name` / `match` / `schema.properties` / `adetailer_params`。先頭から順に照合され、末尾の `match: '.*'` が受け皿 |
| `src/scripts/checkpoint_proxy.yml` | `base_model_fallback` に **Civitai の baseModel ラベル**→ 代替チェックポイント。**未定義だと Checkpoint リソースを持たない画像が `ValueError` で落ちる**。非蒸留版を蒸留版へ寄せるなら `proxies` にも1行 |
| `tests/backend/unit/test_generation_defaults.py` | 代表チェックポイント定数と、`COMFY_CHECKPOINTS` / `DISTILLED_CHECKPOINTS` への追加 |

`match` は **civitdl のモデル名(stem)で照合する。Civitai の baseModel 名ではない。**
`base_model_fallback` の方だけが baseModel ラベルで引く。この2つを混同しない。

### スキーマの骨格(comfy 経由のアーキは全部この形)

**キーを落とさない。** 送らないキーはルーターの spec 既定に落ちるので、クライアントから
制御できなくなる。

**キーを落とさない。値には根拠のコメントを付ける。** コメントが無い値は、あとから
誰も動かせなくなる(実測なのか勘なのか判別できないため)。

```yaml
  # エントリ冒頭に、このアーキの前提をまとめて書く:
  #   バリアントが併存するか / 1エントリにした場合の割り切り
  #   match 式の意図(何を除外しているか、ルーター側と同じ式であること)
  #   値域を絞った/広げた理由
  - name: <arch>
    match: '<ルーターの MODEL_SPECS の pattern と同じ式>'
    schema:
      type: object
      properties:
        steps:
          type: integer
          default:
          minimum:
          maximum:          # 幅を持たせると、その範囲なら元画像の申告値が残る
        cfg_scale:
          type: number
          default:
          minimum:          # 蒸留モデルは min=max=1.0 で固定
          maximum:
        sampler_index:
          type: string
          # 既定の出どころ(実画像の申告値 / 作者常用 など)
          default:
          enum: []
        scheduler:
          type: string
          default:
          # 解決できない名前はルーターがエラーにする(黙って既定へ落とさない)
          enum: []
        # --- Hires(2パス目)。1パス目の値は引き継がれないので全部要る ---
        hr_cfg:
          type: number
          default:          # 蒸留モデルは 1.0 固定(理由は下記)
          minimum:
          maximum:
        hr_scale:
          type: number
          # 上限の根拠(作者の記載 / 対応解像度 / 実測のどれか)を書く
          default:
          minimum:
          maximum:
        hr_upscaler:
          type: string
          # 既定を選んだ根拠。ドキュメントの表記が使えない場合はそれも書く
          default:
          enum: []
        hr_sampler_name:
          type: string
          default:
          enum: []
        hr_scheduler:
          type: string
          # 送らないとルーター側の既定に落ちる。1パス目の scheduler は引き継がれない
          default:
          enum: []
        hr_second_pass_steps:
          type: integer
          default:
          minimum:
          maximum:
        denoising_strength:
          type: number
          # 下げるほどアップスケーラの出力に忠実。「下げれば安全」ではない
          default:
          minimum:
          maximum:
        adetailer:
          type: boolean
          default:
          enum: []
    adetailer_params:
      ad_prompt: ""         # 蒸留モデルで有効にするなら空にしない(理由は下記)
      ad_negative_prompt: ""
```

| 埋めかた | |
|---|---|
| `steps` / `cfg_scale` / `sampler_index` / `scheduler` | 準備2 の3ソース。バリアント別に分かれるのはここ |
| `hr_*` / `denoising_strength` | 準備2 の② + 実測。上流のモデルカードにはまず載っていない |
| すべての `enum` | 下記の実機検証を通したものだけ |

**蒸留モデルでは `cfg_scale` と `hr_cfg` を `minimum: maximum: 1.0` で固定する。**
cfg<=1 だとルーターは負条件を `ConditioningZeroOut` にするので CFG が成立せず、
1.0 以外を選べる状態にしておくと壊れた絵が出るだけになる。

**同じ理由で、蒸留モデルの `adetailer` を有効にするなら `ad_prompt` を必須にする。**
空だとルーターは本体プロンプト全文を inpaint に流し(ネイティブ ADetailer と同一挙動)、
ネガティブでの抑止が効かないため、顔がプロンプト中の別の被写体に描き換えられる。

### ★ enum に書く値は、全部ルーターで解決できることを実機で確かめてから渡す

**解決できない名前はルーターがエラーで落とす。** enum に嘘が1つ混ざると、その値を
選んだ生成が丸ごと失敗する。**ドキュメントに書かれた名前をそのまま写さない** —
上流や作者が使う表記は forge の一覧名であることがあり、comfy の語彙とは別物。

```bash
docker compose exec -T comfy curl -s http://127.0.0.1:8188/api/object_info/KSampler > /tmp/ks.json
docker compose exec -T comfy curl -s http://127.0.0.1:8188/api/models/upscale_models > /tmp/up.json
docker compose exec -T sdui sh -c 'ls /app/data/models/adetailer' > /tmp/ad.txt
python3 - <<'EOF'
import json,re
ks=json.load(open('/tmp/ks.json'))['KSampler']['input']['required']
samplers, scheds = set(ks['sampler_name'][0]), set(ks['scheduler'][0])
upmodels=json.load(open('/tmp/up.json'))
src=open('comfy/router/scripts/comfy_router.py',encoding='utf-8').read()
SAMPLER_MAP=set(re.findall(r'^\s*"([^"]+)": \("', src, re.M))
PIXEL=set(re.findall(r'"([a-z\- \(\)\.]+)": "(?:lanczos|nearest-exact|bilinear|bicubic|area)"', src))
def chk_up(v):
    k=v.strip().lower()
    if k in PIXEL: return 'pixel'
    return next((c for c in upmodels if c.lower()==k or c.rsplit('.',1)[0].lower()==k), None)
for v in ['<enum に書く候補を全部>']:
    print(v, chk_up(v) or (v.lower() in samplers) or (v.lower() in SAMPLER_MAP) or '*** 解決不可 ***')
EOF
```

`hr_upscaler` の画素拡大側(`Lanczos` / `Bicubic` / …)は `_resolve_upscaler` が
小文字化して引くので大小は問わない。

### 1エントリか2エントリか

蒸留版と非蒸留版が併存するなら、**両方の値域を1つのスキーマで表現することはできない**。

- **2エントリ**(絞り込みの強い方を先に置く): 名前で判別できる場合。
  `'(?=.*(?i:<arch>)<境界>)(?=.*(?i:<バリアント名>))'` → `'(?i:<arch>)<境界>'` の順。
  先読みを2つ並べると語順に依存しない
- **1エントリ**: 運用するバリアントを絞る場合。非蒸留を `checkpoint_proxy` の
  `proxies` で蒸留版へ読み替え、読み替え対象外のコミュニティ merge には蒸留用の値が
  当たる、という割り切りを**明示的に文書化する**(Krea 2 RAW と同じ扱い)

**どちらを選ぶかはクライアント側の運用判断。** こちらは選択肢と、絞った場合に何が
起きるか(読み替え対象外のモデルに合わない値が当たる)を提示して、決定は相手に委ねる。

### `match` は「同じ式」ではなく「同じマッチ集合」にする

ルーター側の pattern と**マッチする集合が一致していること**が要件。食い違うと、forge へ
流れるモデルに comfy の語彙を送る(またはその逆)ことになる。

**渡す式に処理系固有の構文を使わない。** ルーターは Python だが、クライアントの
`match` は**フロントエンドへ渡って JS の `RegExp` としても評価される**。Python の
インラインフラグ `(?i:...)` は JS に無く、`Invalid group` で壊れる。

```python
r"(?i:<arch>)(?![a-z])"          # ルーター側(Python)はこれでよい
```
```yaml
match: '[Aa][Nn][Ii][Mm][Aa](?![a-z])'   # 渡す側は文字クラスに展開する
```

境界(`(?![a-z])` など)が要るのは、アーキ名が他の語の接頭辞になっているとき
(例: `anima` は `animagine` / `animal` / `animated` の接頭辞)。Python 側で全体を
`re.IGNORECASE` にすると後読みにも効いて、大文字が続く名前まで落ちる。

**渡す前に両方の処理系で試す。** 手元の全チェックポイント名 + 誤爆候補で照合し、
2つの式が**同じ集合**になることを確認する。

```bash
node -e "new RegExp('<渡す式>')"   # 構文が通るか
python3 -c "import re; print([n for n in NAMES if bool(re.search(A,n))!=bool(re.search(B,n))])"
```

## 責任分解（動かさない）

クライアントは **I/F どおり動くかをブラックボックス的にテストする側**。

| | 担当 |
|---|---|
| I/F の不一致（キー・型・null・解像度・hash・interrupt） | **ルーター** |
| パラメータの値（アーキ別の既定値を持つ） | **クライアント** |
| 想定外・質問・不整合の起票 | クライアント → ルーター |
| 実経路での生成と結果の照合 | クライアント |
| モデル/LoRA の選定と civitdl での導入 | クライアント |

**I/F の差分が出たらルーター側のバグ。** ゴールは「クライアントから見て forge
ネイティブと区別がつかない状態」。ただし**値の面倒は見ない**。

一度この線引きを誤り、「クライアントは何も改修しない」と解釈して steps/cfg/sampler を
ルーターで書き換えた。EXIF に送っていない値が残る問題を指摘され全廃した。
**「互換」と言われたら I/F のことで、値のことではない。**

### 禁止は具体的な行為に限定して書く

「無改修で」のような広い言い方をすると、相手は**必須作業まで禁じられたと読んで止まる**。

| クライアントがやること | 可否 |
|---|---|
| アーキ定義とモデル読み替えの追加 | **必須**。こちらがスキーマを渡すので入れてもらう |
| 不整合を見つけて起票する | お願いしたいこと |
| I/F の不一致を埋める回避策(キーの補正、型変換、値の握り潰し) | **やらないでほしい** |

**禁じるのは3行目だけ。** 相手の作業を止める言い方をしない。

## 検証の厳密さ

**妥協なく確認できるまで終わらせない。** これが基準。

- **機能は「有無の差分」で確認する。** メタデータの申告を見て合格にしない
- **2モデル × 3機能 = 6パターン**（アーキが増えたら増える）
- **実データは Civitai が申告する設定をそのまま使う。** 手持ちモデルで代用しない
  （合成プロンプトの生成物を元画像に使って差し戻された。実データを流したからこそ
  本番バグが見つかった）
- **短い示唆は最小の読みで受ける。** 拡大解釈して計画を作り替える前に、
  現在の全体マトリクスを再掲して合意を取る
- **ダウンロードと生成を分離しない。** 通常の生成タスクの一部として civitdl が走る形で
  検証する（「ふつうに生成タスクとして dl も一緒にやるだけという意図」）。
  **分離した瞬間、テストしているのは本番と別の経路になる。**
  フェーズ分けや負荷の集約といった戦略を発明しない — 短い指示を拡大解釈して計画を
  作り替えると、二転三転して確認コストが相手側に発生する

## 報告の規律

**「合格」と言えるのは証拠グレードを満たしたときだけ。** 証拠なしの合格宣言は、
後で必ず撤回コストになる。

実際に起きたこと: `info.extra_generation_params` にキーが出ただけで「ADetailer 合格」
「LoRA 合格」と報告した。**言い方を変えて3回**「本当に全部動いたのか」と問い直され、
3回とも「いえ、半分です」と答えることになった。最後に確認方法を問われて申告を見た
だけだと露呈し、合格を撤回してやり直した。**報告が信用されていない状態を自分で作っている。**

守ること:

- 合格宣言には**証拠の種類を併記する**（申告のみ／実測、どのモデル・どの機能か）
- **未確認を聞かれる前に明示する。** 台帳を常に正確に保つ
- 計画を変更したら**マトリクス全体を再掲して抜けを確認する**。方針が揺れた際に
  実際に HR が検証項目から落ち、指摘されて拾い直した

同型の規律として「**コード修正済み**」と「**稼働中に反映済み**」も必ず区別する。

### 値ごとに「実測 / ドキュメント由来 / 未検証」を明記する

実測した値と、読んだだけの値を同じ口調で並べない。相手は両方とも検証済みと受け取る。
**ドキュメントの推奨は「調べた」であって「測った」ではない。**

**あるバリアントでの実測を別のバリアントに外挿しない。** 蒸留版と非蒸留版では
負条件の扱い(cfg<=1 では `ConditioningZeroOut`)が変わるため、ネガティブ側に依存する
機能(ADetailer の抑止など)の結論はそのまま移らない。

### 断定する前に、その関数を通しで読む

`grep -n` のヒット行だけで制御フローを再構成しない。**ヒット行番号が飛んでいたら
未読区間がある。** 早期 return やガード節はそこに隠れる。

### 誤りが分かったら、単独で即撤回する

誤った情報は相手の作業計画を変える。**放置は相手の時間を止める。** 次の定期報告に
混ぜず、撤回だけのメッセージを送る。

## 連絡

クライアント側のセッションへ `SendMessage` で連絡する。**セッション名はユーザーに確認する**
（`ListAgents` で一覧は出るが、本番環境用と開発環境用が並んでいることがあり、名前だけでは
判別できない）。
**今何をしていて次に何をするかをこまめに伝える。** 特に再起動前、自分の判断ミスに
気づいたとき。

**検証の停止・縮小はユーザーの承認事項。エージェント間の合意で確定させない。**
一度、自分から相手に停止を提案し、相手の受諾をもって全ジョブを止めた。なぜ止めたのか、
どういう要請だったのかを問われ、妥協なく確認できるまで終わらせないと差し戻された。
**相手が選べば人間の頭越しに検証が止まる構図を作らない。**

**準備1〜3を終えてから声をかける。** スキーマを別便にすると、相手は先に自前で定義を
書き、あとから突き合わせる二度手間になる。

依頼時に伝える内容:

```
<arch> をルーター経由で使えるようにしました。互換性の確認をお願いします。

## 先にやっていただく必要があること（検証の前提）
下に付けたパラメータスキーマを generation_defaults.yaml と checkpoint_proxy.yml へ
入れてください。**これが無いと受け皿(match: '.*')の forge 用エントリに落ちて、
comfy が解決できない値を送ることになり、検証以前に成立しません。**
enum と既定値はこちらで実機検証済みなので、そのまま使えます。

## そちらの役割
上の定義を入れたうえで、インタフェースの一致をブラックボックス的に見ていただきたいです。
**不整合を見つけたら返信してください。修正はこちらでやります。**

やらないでほしいのは**I/F の不一致を埋める回避策**だけです（キーの補正、型の変換、
値の握り潰し）。何が非互換なのか分からなくなるためです。アーキ定義の追加は
「改修」ではなく必須作業なので、遠慮なく入れてください。

## 振り分け
override_settings.sd_model_checkpoint が <pattern> にマッチしたら comfy へ。
クライアントの post_sdapi_queue_txt2img(= /sdapi/queue/txt2img)経由でも振り分けられます。
SDXL 等は従来経路のままです。

## 確認をお願いしたいこと
1. ネイティブ SDXL とのキー単位の機械照合(parameters / info のキー数・型・null)
2. **回避策を入れない状態で**、実際のプロンプト・LoRA・ADetailer 設定で生成
   （生成パラメータの値はそちらのスキーマが持ちます。ルーターは書き換えません）
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
- モデルの初回ロードでディスクが飽和します(1アーキ 17GiB 前後)。同じモデルの連続生成
  なら RAM に残るので発生しませんが、モデルを切り替えるたびに1回かかります
- group_hash の予測基底は実際の保存 meta と同じ構造(59キー)にしてください

## 使えるバリアント
<準備1で作った一覧。RAW / 蒸留版など spec の違うものが混在する場合は必ず明示する。
 名前と中身が食い違うものがあれば必ず添える
 (モデル名に別アーキ名が入っている、同じ mid でも版によってアーキが違う、など)>

## パラメータスキーマ
<準備3で作った YAML をそのまま貼る。あわせて触るファイルを名指しする>
- src/generation_defaults.yaml            … アーキのエントリ
- src/scripts/checkpoint_proxy.yml        … base_model_fallback に baseModel ラベルを追加
                                             (未定義だと Checkpoint リソース無しの画像が
                                              ValueError で落ちます)
- tests/backend/unit/test_generation_defaults.py … 代表チェックポイント定数

enum と既定値は**すべて稼働中のルーター / comfy に問い合わせて解決可能を確認済み**です。
値ごとの根拠（実測 / ドキュメント由来 / 未検証）も併記してあります。

## 再起動について
sd-queue のキューは永続化されていません。再起動するとキューと完了済みタスクの結果が
消え、/sdapi/queue/{id}/status が 404 になります。**再起動が必要になったら事前に
連絡します。** そちらから「落としていい」という判断は不要です(本番への断はユーザーの
裁量なので、こちらからユーザーに上げます)。
```

## 渡す Civitai 画像の選び方

**アーキごとに1枚、LoRA が付いている画像**を選ぶ。Civitai の画像ページが申告する
リソース（checkpoint / LoRA）をクライアントがそのまま civitdl するので、**実データでしか
出ない問題**（想定外のバリアント名、アーキ違いの LoRA、極端な生成パラメータ）を拾える。

**渡す前に、その画像が本当に条件を満たすか API で確認する。** `/api/v1/images` は
`meta: null` を返すので使えない。`trpc` の `image.getGenerationData` で
**resources(checkpoint / LORA)と meta の両方**を見る。

```bash
curl -s -H "Authorization: Bearer $CIVITAI_TOKEN" \
  "https://civitai.com/api/trpc/image.getGenerationData?input=%7B%22json%22%3A%7B%22id%22%3A<image_id>%7D%7D" \
  | python3 -c "
import json,sys
r=json.load(sys.stdin)['result']['data']['json']
res=r.get('resources') or []
print('LORA:', [(x['modelName'], x['versionId'], x.get('strength')) for x in res if x['modelType']=='LORA'])
print('CKPT:', [(x['modelName'], x['versionName'], x['versionId']) for x in res if x['modelType']=='Checkpoint'])
m=r.get('meta') or {}
print('meta:', {k:v for k,v in m.items() if k in ('steps','cfgScale','sampler','Size','seed')} or '空')"
```

**相手に渡す入力は、相手の入口を通ることを自分で確認してから渡す。** 満たすべき条件を
先に列挙し、機械的に確認する。**resources と meta の両方が揃っていること**が最低条件
(`meta` が空の画像はクライアントの入口で例外になる)。

## E2E チェックリスト

### A. 構造の一致（ネイティブ SDXL と機械照合）

- [ ] トップレベルが `["info", "parameters"]`（実績 7キー）
- [ ] `parameters` のキー数・キー名が一致（実績 59）／**型**・null の差分ゼロ
- [ ] `info` のキー数・キー名が一致（実績 32）／**型**・null の差分ゼロ

**キー名だけでなく型を突き合わせる。** `info` に**リテラルを埋めたキーは型が食い違う**
(ネイティブはリクエストの値をそのまま返すため)。`info` を組み立てるときは
**値と同じく型もエコーバックに任せる**(`getattr(req, "<key>", <既定>)`)。

group_hash から pop されるキーなら実害は無いが、pop されないキーで同じことが起きると
**同一条件の画像がハッシュ違いで別グループに割れる**。1件見つかったら、同じ作りの
キーが他に無いか全部見る。

`parameters` 側は `model_dump()` なので自分で型を作らない(ネイティブの
`vars(txt2imgreq)` と等価であることは確認済み)。

### B. `parameters` の意味論

- [ ] **送信値がそのまま返る**（steps / cfg_scale / sampler / scheduler を書き換えていない）
- [ ] **送っていないキーは null のまま**（`sampler_index` だけ送って `sampler_name` が埋まらない）
- [ ] **HR の 2nd パス設定が効く**（`denoising_strength` / `hr_upscaler` / `hr_sampler_name` /
      `hr_scheduler` / `hr_second_pass_steps` を変えると絵が変わり、`parameters` には送信値が返る）
- [ ] `seed: -1` は `parameters` に `-1`、`info.seed` に実値
- [ ] **丸めが吸収される**（`1928` を送れば `1928` が返り、実画像もその比率）
- [ ] `info.sd_model_hash` が埋まる（forge はこのモデルをロードしないので自前で計算する）
- [ ] **HR 時は1パス目の解像度**（`832x1216`+`hr_scale:2` → `parameters` は `832x1216`、実ファイル `1664x2432`）

### C. 素通しと不動点

- [ ] どんな値を送っても `parameters` が送信値と一致（書き換えが残っていない）
- [ ] group_hash の事前予測と実測が一致
- [ ] `/params` で読み戻して再生成すると**ピクセル差 0**

最後が最重要。読み戻し→再生成で値が動くなら、どこかで書き換えが残っている。
この検証項目は「生成物から読み戻して再生成したら一致するのか」という問いが起点で、
それまで未検証だった。

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

**契約項目を「そう作ったつもり」で埋めず、実際にその挙動になることを1件ずつ確認する。**
落とす範囲は設計判断(1)の表に従う(**ネイティブが落ちる範囲を超えない**)。

- [ ] 他アーキの LoRA は**ネイティブと同じく**扱われる(片方だけエラーにしない)
- [ ] **実在しない名前はエラーで落ちる**（checkpoint / LoRA / sampler / scheduler /
      `hr_upscaler` / ADetailer モデル）。既定へ落として黙って別物で生成しない
- [ ] `n_iter > 1` はエラーで落ちる（黙って枚数が減らない）
- [ ] **値域では落ちない**（極端な steps / cfg / 想定外のサンプラーを送っても生成される）

**相手に依頼するコストは、自分の実装のどこでエラーが返るかを確認してから見積もる。**
名前の解決は `_generate` の中なので、エラーは投入時ではなく**実行時**に返る。
異常系も本番ジョブの後ろでキュー待ちする。「安価だから」と言って依頼しない。

落とす基準は**「クライアントがそれに気づけるか」**の一点。名前が解決できないまま既定へ
落とすと、指定と違うもので生成されたことを検出する手段が無い。一方、値域は
クライアント自身が選んだ値なので、絵が期待と違っても原因を追える。だから落とさない。

---

# 502 が出たら切り分ける

```bash
curl -s -m 10 -o /dev/null -w "forge   %{http_code}\n" http://127.0.0.1:7680/sdapi/queue/login
curl -s -m 10 -o /dev/null -w "civitdl %{http_code}\n" http://127.0.0.1:7680/civitdl/models/
iostat -x 1 2 | grep "^sdb"                                       # %util 97% なら飽和
docker inspect -f '{{.State.StartedAt}}' $(docker compose ps -q sdui)   # 再起動直後か
```

実例: forge は 200 なのに 502 と報告された → civitdl のワーカー占有。別の回は
forge の再起動ウィンドウ。**決めつけない。**

---

# 完了条件

- [ ] `MODEL_SPECS` にエントリが入り、拡張がロードされている
      `docker compose logs sdui 2>&1 | grep "comfy-router] wrapped"`
- [ ] キュー経由で base 生成が通り、顔を等倍で見て破綻がない
- [ ] クライアントの E2E チェックリスト A〜F がすべて合格
- [ ] クライアントへ**スキーマを渡し**、enum の全値が解決可能であることを実機検証済み
- [ ] 変更をコミット（**コミットは都度ユーザーの承認を取る**）
- [ ] 知見を memory に記録（アーキ固有の値、踏んだ罠、実測性能）
