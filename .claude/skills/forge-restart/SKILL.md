---
name: forge-restart
description: Restart the Forge (sd-ui) container safely without killing an in-flight generation. Use whenever the user asks to 再起動 / restart / reload Forge, sdui, or sd-forge-docker, or when a change to an extension, model file, or ADetailer patch needs to take effect. Covers waiting for the queue to drain, the correct compose command, waiting for the API to really be up, and verifying the reload.
---

# Forge (sd-ui) の再起動

このホストの Forge は**共有インスタンス**で、他のセッションやフロントエンドが同時に生成を投げてくる。再起動は進行中のジョブを問答無用で殺すため、手順を守ること。

## 何をしたら再起動が要るか

| 変更 | 再起動 |
|---|---|
| 拡張の `.py` を編集した | **要る**（モジュールは起動時に1回 import される） |
| `models/adetailer/` に**新しい名前**のモデルを置いた | **要る**（`model_mapping` は起動時に構築） |
| 既存モデルの**中身を差し替え**た（同じパス） | 不要（`YOLO(path)` は毎回作り直される） |
| `docker-compose.yml` を変えた | `restart` では反映されない → 下記参照 |

## 手順

### 1. キューが空くのを待ってから再起動する

`sleep` 5秒間隔のような粗いポーリングだと、ジョブの切れ目を逃して待ち続けるか、逆に走行中に殺す。**1秒間隔**で、**連続2回**アイドルを確認してから落とす。

```bash
cd /home/ssakurai/sd-forge-docker
idle=0
while [ $idle -lt 2 ]; do
  j=$(curl -s -m 5 http://127.0.0.1:7680/sdapi/v1/progress \
      | python3 -c 'import sys,json;print(json.load(sys.stdin).get("state",{}).get("job") or "")' 2>/dev/null)
  if [ -z "$j" ]; then idle=$((idle+1)); else idle=0; fi
  sleep 1
done
docker compose restart sdui
```

長引くことがあるので、この待機はバックグラウンド実行（`run_in_background: true`）に回してよい。

ユーザーが「今すぐ落とせ」と明示した場合のみ待たずに実行する。その場合、**進行中のジョブを潰した事実は必ず報告する**（`/sdapi/v1/progress` の `job` と `progress` を落とす前に記録しておく）。

### 2. `restart` を使う。`up -d` や `down` は使わない

```bash
docker compose restart sdui      # これ
```

`up -d` は `docker-compose.yml` の未コミット変更を巻き込んで反映してしまう。バインドマウント上のファイル変更を反映したいだけなら `restart` で足りる（プロセスが再起動＝再 import されるため）。

`docker-compose.yml` 自体の変更（イメージ、ボリューム、`ARGS` 等）を反映する場合だけ `up -d` が必要。その場合は他に未コミットの変更が混ざっていないか `git diff docker-compose.yml` で先に確認する。

### 3. API の復帰は **HTTP 200** で待つ

`curl` は 502 を受け取っても**終了コード 0** を返す。到達性だけで判定すると、traefik が 502 を返している起動途中を「準備完了」と誤判定し、後続のリクエストが空応答になる。

```bash
until [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:7680/sdapi/v1/options)" = "200" ]; do
  sleep 1
done
```

### 4. 反映を確認する

```bash
docker logs sd-forge-docker-sdui-1 2>&1 | grep -a "ADetailer initialized" | tail -1
# → [-] ADetailer initialized. version: 24.11.1, num models: 12
```

モデルを追加したなら件数が増えているはず。一覧そのものは:

```bash
curl -s http://127.0.0.1:7680/adetailer/v1/ad_model
```

## 環境メモ

- コンテナ名 `sd-forge-docker-sdui-1`、ポートは traefik 経由で `127.0.0.1:7680`
- データは `/data2/forge-data` を `/app/data` にバインドマウント。拡張もモデルもこの下なので、**イメージを更新してもローカル改変は消えない**
- 拡張は独立リポジトリ: `sammrai/adetailer`（private）、`sammrai/sd-queue`。拡張を編集したらそちらでコミットする。本体 `sd-forge-docker` とは別
- 起動には1〜2分かかる（チェックポイントのロードを含む）
