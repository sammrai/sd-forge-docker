# ベースイメージの指定
ARG CUDA_VERSION=12.4.0
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

# 環境変数の設定（非対話モードでのインストールを可能に）
ENV DEBIAN_FRONTEND=noninteractive

# ビルド引数の定義（デフォルトのコミットIDを設定）
ARG COMMIT_ID=f53307881bfd824dbdce6ac0d4bba04d9a74ab36
ARG PYTORCH_VERSION=2.4


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
        python3-pip

# setuptools 70+ で pkg_resources が分離され、CLIP など旧来 setup.py が壊れるため
# build-isolation も含め全 pip install で setuptools<70 にピン
RUN echo 'setuptools<70' > /etc/pip-constraints.txt
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt

RUN python3 -m pip install --upgrade pip && \
    short_cuda_version="$(echo ${CUDA_VERSION} | cut -d. -f1-2 |tr -d .)" && \
    python3 -m pip install torch==${PYTORCH_VERSION} torchvision torchaudio packaging --extra-index-url https://download.pytorch.org/whl/cu${short_cuda_version}


# 作業ディレクトリの設定
WORKDIR /app

# リポジトリのクローンと特定のコミットへのチェックアウト
RUN git clone https://github.com/lllyasviel/stable-diffusion-webui-forge.git webui && \
    cd webui && \
    git checkout ${COMMIT_ID}

# Python の依存関係をインストール
RUN pip install -r webui/requirements_versions.txt

# Forge 起動時に毎回 pip install されないよう CLIP をシステム側へプリインストール
RUN pip install https://github.com/openai/CLIP/archive/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1.zip

# ADetailer 拡張(data ボリューム常駐)の依存をシステム側へプリインストール。
# forge は webui(非root)で起動するため拡張の自動 install は --user(~/.local)に入り
# ランタイムから import できない。ここで root=system に入れておけば forge が確実に import 可能。
# mediapipe は <=0.10.15 にピン(0.10.16+ は numpy>=2 を要求し forge/blendmodes(numpy<2)と競合)。
RUN pip install "ultralytics>=8.3.0,!=8.3.41,!=8.3.42,!=8.3.45,!=8.3.46" \
    "mediapipe>=0.10.13,<=0.10.15" "rich>=13.0.0"

# モデル管理ツールのインストール
RUN pip install civitdl

# VAE Baking Toolのコピー
COPY vae_bake.py /app/vae_bake.py
RUN chmod +x /app/vae_bake.py

# ControlNet fix for txt2img + Tile (HiRes Fix support)
COPY controlnet.py /app/webui/extensions-builtin/sd_forge_controlnet/scripts/controlnet.py

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