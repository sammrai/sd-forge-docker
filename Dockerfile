# ベースイメージの指定
ARG CUDA_VERSION=12.4.0
ARG PYTORCH_VERSION=2.4
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

# 環境変数の設定（非対話モードでのインストールを可能に）
ENV DEBIAN_FRONTEND=noninteractive

# ビルド引数の定義（デフォルトのコミットIDを設定）
ARG COMMIT_ID=f53307881bfd824dbdce6ac0d4bba04d9a74ab36

RUN echo $PYTORCH_VERSION

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
    short_cuda_version="$(echo ${CUDA_VERSION} | cut -d. -f1-2 |tr -d .)" && \
    echo $PYTORCH_VERSION &&\
    echo $CUDA_VERSION &&\
    python3 -m pip install torch==${PYTORCH_VERSION} torchvision torchaudio packaging --extra-index-url https://download.pytorch.org/whl/cu${short_cuda_version}


# 作業ディレクトリの設定
WORKDIR /app

# リポジトリのクローンと特定のコミットへのチェックアウト
RUN git clone https://github.com/lllyasviel/stable-diffusion-webui-forge.git webui && \
    cd webui && \
    git checkout ${COMMIT_ID}

# Python の依存関係をインストール
RUN pip install -r webui/requirements_versions.txt

# モデル管理ツールのインストール
RUN pip install civitdl


# webui ユーザーの作成と権限設定
RUN useradd -m webui && \
    chown webui:webui /app/webui -R

# ユーザディレクトリの位置を定義
ENV USERDATA_DIR=/app/data
RUN mkdir -p $USERDATA_DIR && chown webui:webui $USERDATA_DIR

# 実行ユーザーの切り替え
ENV venv_dir="-"
ENV no_proxy="localhost, 127.0.0.1, ::1"

ENTRYPOINT ["/bin/bash", "-c", "\
  chown -R 1000:1000 $USERDATA_DIR && \
  civitconfig default --api-key $CIVITAI_TOKEN || true; \
  civitconfig alias --add @lora $USERDATA_DIR/models/Lora && \
  civitconfig alias --add @vae $USERDATA_DIR/models/VAE && \
  civitconfig alias --add @embed $USERDATA_DIR/models/text_encoder && \
  civitconfig alias --add @checkpoint $USERDATA_DIR/models/Stable-diffusion && \
  su webui -c \"/app/webui/webui.sh --gradio-allowed-path . --data-dir $USERDATA_DIR $ARGS\""]