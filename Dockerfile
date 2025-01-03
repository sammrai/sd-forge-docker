# ベースイメージの指定
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# 環境変数の設定（非対話モードでのインストールを可能に）
ENV DEBIAN_FRONTEND=noninteractive

# ビルド引数の定義（デフォルトのコミットIDを設定）
ARG COMMIT_ID=f53307881bfd824dbdce6ac0d4bba04d9a74ab36

# システムパッケージのインストールと Python パッケージのインストールを一つの RUN 命令に統合
RUN apt update && \
    apt upgrade -y && \
    apt install -y \
        wget \
        git \
        python3 \
        python3-venv \
        libgl1 \
        libglib2.0-0 \
        apt-transport-https \
        libgoogle-perftools-dev \
        bc \
        python3-pip && \
    # pip のアップグレード
    python3 -m pip install --upgrade pip && \
    # 必要な Python パッケージのインストール
    python3 -m pip install torch torchvision torchaudio packaging --extra-index-url https://download.pytorch.org/whl/cu124

# 作業ディレクトリの設定
WORKDIR /app

# リポジトリのクローンと特定のコミットへのチェックアウト
RUN git clone https://github.com/lllyasviel/stable-diffusion-webui-forge.git webui && \
    cd webui && \
    git checkout ${COMMIT_ID}

# Python の依存関係をインストール
RUN pip install -r webui/requirements_versions.txt

# run.sh スクリプトのコピーと実行権限の付与
COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh

# webui ユーザーの作成と権限設定
RUN useradd -m webui && \
    chown webui:webui /app/webui -R

# 実行ユーザーの切り替え
USER webui

ENV venv_dir="-"
# ENTRYPOINTをシェル形式に変更してARGSを渡す
ENTRYPOINT ["/bin/bash", "-c", "/app/webui/webui.sh $ARGS"]
