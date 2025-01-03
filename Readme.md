# Docker Compose ファイルの説明とトンネル設定方法


## `docker-compose.yml` の設定内容

### WebUIサービス
- **イメージ**: コメントアウトされていますが、`ghcr.io/yummiii/sd-webui-forge-docker:latest` を使用できます。  
  また、`build: .` としてローカルのDockerfileを使用してビルド可能です。
- **ボリュームマウント**: ローカルの`data`ディレクトリをコンテナ内の`/app/sd-webui`にマウント。
- **環境変数**:
  - `ARGS`: WebUIに渡す引数。例: `--listen --enable-insecure-extension-access`
  - `venv_dir`: 仮想環境のディレクトリ指定。
  - `TORCH_INDEX_URL`: PyTorchのCUDA 12.1対応ホイールを取得するURL。
  - `TORCH_COMMAND`: 必要なPyTorchライブラリをインストールするコマンド。
  - `COMMIT_HASH`: 固定バージョンのコミットを指定。
- **GPUリソース**: NVIDIA GPUを利用する設定を含む。

### Tunnelサービス
- **イメージ**: Cloudflareの公式Dockerイメージ `cloudflare/cloudflared` を使用。
- **再起動ポリシー**: `unless-stopped`（停止しない限り自動再起動）。
- **コマンド**:
  - `tunnel`: Cloudflareトンネルを開始。
  - `--no-autoupdate`: 自動更新を無効化。
  - `run`: トンネルを実行。
- **環境変数**:  
  - `TUNNEL_TOKEN`: Cloudflareトンネルの認証に必要なトークン（シークレットとして設定）。

## 起動

1. git clone https://github.com/sammrai/sd-forge-docker.git
1. cd sd-forge-docker
1. echo "TOKEN=xxxxxx" > .env
1. sudo docker compose build
1. sudo docker compose up -d
1. sudo docker compose logs -f

https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/13507