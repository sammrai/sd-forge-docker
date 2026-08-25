---
name: forge-restart
description: Restart or rebuild the Forge (sd-ui) container safely without killing an in-flight generation. Use whenever the user asks to 再起動 / restart / reload Forge, sdui, or sd-forge-docker, when a change to the comfy-router extension, a model file, or an ADetailer patch needs to take effect, or when edited code does not seem to be running. Covers deciding between restart and rebuild (the router is baked into the image, so restart alone does not pick it up), waiting for the queue to drain, local build overrides, waiting for the API to really be up, and verifying what the running process actually loaded.
---

# Forge (sd-ui) の再起動

他セッションやフロントエンドが同時に生成を投げてくる**共有インスタンス**。再起動は
進行中のジョブを問答無用で殺す。

## restart か、再ビルドか

| 変えたもの | 必要な操作 |
|---|---|
| `comfy/router/`（イメージ同梱） | **再ビルド + 作り直し**。`restart` では反映されない |
| `/app/data/extensions/` 配下のマウント上の拡張 | `restart` |
| `models/adetailer/` に**新しい名前**のモデルを追加 | `restart`（`model_mapping` は起動時に構築） |
| 既存モデルの中身を差し替え（同じパス） | 不要（`YOLO(path)` は毎回作り直される） |
| `docker-compose.yml` | `up -d` |

`comfy-router` はイメージに焼き込まれ、ENTRYPOINT が起動のたびに `rm -rf` してから
`/app/data/extensions/` へ置き直す。**コンテナ内を直接編集しても次の起動で消える。**
稼働中コンテナの状態ではなく compose の定義で判断すること。

## 手順

**ビルドを先に済ませる。** アイドルを掴んでからビルドすると、その数分で切れ目が消える。
同じ理由で `up -d --build` の一体形は使わない。

### 1. ビルド（無停止。再ビルドが要る場合のみ）

```bash
docker compose build sdui
```

`sdui` に `build:` は無いので、`docker-compose.override.yml` に足す（compose が自動で
マージする。**`-f` を付けると自動読み込みが無効になる**）。`.gitignore` 済み。

```yaml
services:
  sdui:
    build:
      context: .
      dockerfile: Dockerfile
      args: {CUDA_VERSION: "12.4.0", PYTORCH_VERSION: "2.4"}
      cache_from: [sammrai/sd-forge-docker:cache-12.4.0]
```

`cache_from` が無いと **torch から全ビルド**になる。publish 済みのタグを指すこと。

### 2. アイドルを掴んで作り直す

切れ目は短い。**0.2秒間隔で連続3回**アイドルを確認して即座に落とす。`curl` の起動コストが
効くので Python で回す。長引くのでバックグラウンド実行に回す。

```python
import time, urllib.request, json, subprocess
idle = 0
while idle < 3:
    try:
        d = json.load(urllib.request.urlopen(
            "http://127.0.0.1:7680/sdapi/v1/progress", timeout=5))
        idle = idle + 1 if not (d.get("state", {}).get("job") or "") else 0
    except Exception:
        idle = 0
    if idle < 3:
        time.sleep(0.2)
subprocess.run(["docker", "compose", "up", "-d", "--no-deps", "sdui"], cwd="<repo>")
```

掴めるのは「直前まで空だった」ことだけ。**キューに積まれた分は落ちる**ので他セッションへ
一報を入れる。`--no-deps` は `comfy` を巻き込まないため（GPU を握ったまま両方起動する）。

「今すぐ落とせ」と言われた場合のみ待たずに実行し、**潰したジョブを必ず報告する**
（落とす前に `/sdapi/v1/progress` の `job` と `progress` を記録）。

### 3. 復帰は **HTTP 200** で待つ

`curl` は 502 でも終了コード 0 を返す。到達性だけで判定すると起動途中を完了と誤判定する。
起動には1〜2分かかる。

```bash
until [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:7680/sdapi/v1/options)" = "200" ]; do sleep 1; done
```

### 4. 何が動いているか確認する

「コードを直した」と「プロセスがそれを読んでいる」は別。ファイルが新しくても、コンテナの
起動時刻が mtime より古ければそのコードは動いていない。

```bash
docker compose exec -T sdui sh -c 'grep -c "<足した識別子>" /app/data/extensions/comfy-router/scripts/comfy_router.py'
docker inspect -f '{{.State.StartedAt}}' $(docker compose ps -q sdui)
docker compose logs sdui 2>&1 | grep -a "comfy-router\|ADetailer initialized" | tail -2
```

### 5. 終わったら override を消す

このマシンは**開発環境と本番環境が同一**。`build:` を残すと、ローカルイメージが無い状態の
`up -d` が CI のイメージを取りに行かず**ソースからビルドする**。しかも同じタグ名を
上書きするので CI 産と区別が付かなくなる。

```bash
rm docker-compose.override.yml
docker compose pull sdui     # 本番へ入れるときは CI が publish したものを使う
```

## 環境メモ

- ポートは traefik 経由で `127.0.0.1:7680`。コンテナは compose のサービス名 `sdui` で指す
- `/data2/forge-data` を `/app/data` にバインドマウント。モデルはこの下
- ADetailer と sd-queue は別リポジトリの拡張
