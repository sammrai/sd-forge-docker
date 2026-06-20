# ベースイメージの指定
ARG CUDA_VERSION=12.4.0
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

# 環境変数の設定（非対話モードでのインストールを可能に）
ENV DEBIAN_FRONTEND=noninteractive

# ビルド引数の定義（デフォルトのコミットIDを設定）
ARG COMMIT_ID=f53307881bfd824dbdce6ac0d4bba04d9a74ab36
# forge25: Z-Image(diffusers) 同居のため cu124 / torch 2.5.1 固定。
# 後段で torchaudio==2.5.1+cu124 を決め打ち force-reinstall するため両者を一致させる。
ARG PYTORCH_VERSION=2.5.1


# システムパッケージのインストールと Python パッケージのインストールを一つの RUN 命令に統合
# torch が CUDA を同梱するため NVIDIA CUDA apt リポジトリは不要。ミラー不整合回避のため除去。
RUN rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia*.list 2>/dev/null; \
    apt update && \
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
# build-isolation も含め全 pip install で setuptools<70 にピン。
# huggingface_hub は 1.0 で HfFolder を削除し forge の旧 gradio が import 失敗するため
# 0.34.4 に固定。0.34.4 は HfFolder を持ち、transformers>=4.51(<1.0要件)/diffusers>=0.36
# (>=0.34要件)とも両立。ENV PIP_CONSTRAINT で runtime(launch.py 再install)にも効かせる。
RUN printf 'setuptools<70\nhuggingface_hub==0.34.4\n' > /etc/pip-constraints.txt
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
# Z-Image(diffusers) 用に forge の依存ピンを引き上げる。これをしないと起動時 launch.py が
# requirements を再installして diffusers 等を古い版にダウングレードし ZImagePipeline が消える。
RUN sed -i 's/\r$//' webui/requirements_versions.txt && \
    sed -i -E \
    -e 's/^diffusers==.*/diffusers>=0.36.0/' \
    -e 's/^transformers==.*/transformers>=4.51.0/' \
    -e 's/^peft==.*/peft>=0.17.0/' \
    -e 's/^accelerate==.*/accelerate>=1.2.0/' \
    -e 's/^kornia==.*/kornia>=0.7.3/' \
    -e 's/^huggingface[_-]hub==.*/huggingface_hub==0.34.4/' \
    -e 's/^safetensors==.*/safetensors>=0.4.3/' \
    webui/requirements_versions.txt && \
    echo "=== patched requirements ===" && \
    grep -iE "diffusers|transformers|peft|accelerate|kornia" webui/requirements_versions.txt
RUN pip install -r webui/requirements_versions.txt

# Forge 起動時に毎回 pip install されないよう CLIP をシステム側へプリインストール
RUN pip install https://github.com/openai/CLIP/archive/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1.zip

# モデル管理ツールのインストール
RUN pip install civitdl

# Z-Image を diffusers で実装するための依存（comfy-kitchen不要・forge精度を保ったまま併設）。
# diffusers は z_image パイプラインを含む新しめのもの、transformers は Qwen3 用に >=4.51。
# 元のtorch行がtorchaudioをピン無しで入れ2.11.0(CUDA13要求)になり、diffusers z_image import時に
# libcudart.so.13エラーになるため、入っている torch+CUDA に一致する torchaudio に入れ直す。
# 版とindexは CUDA_VERSION / PYTORCH_VERSION から導出し、matrix の全CUDA系統で不一致を防ぐ
# (12.4.0/2.5.1 -> torchaudio 2.5.1+cu124 / 12.1.0/2.3.1 -> 2.3.1+cu121)。
RUN short_cuda_version="$(echo ${CUDA_VERSION} | cut -d. -f1-2 | tr -d .)" && \
    pip install --force-reinstall --no-deps "torchaudio==${PYTORCH_VERSION}+cu${short_cuda_version}" \
        --index-url "https://download.pytorch.org/whl/cu${short_cuda_version}"
RUN pip install "diffusers>=0.36.0" "transformers>=4.51.0" "peft>=0.17.0" "accelerate>=1.2.0" "kornia>=0.7.3" sentencepiece protobuf

# ADetailer は ultralytics(YOLO)を要求するが、起動時の拡張 install が PIP_CONSTRAINT 等で
# 入らず ModuleNotFoundError になる。イメージに明示プリインストールして確実に使えるようにする。
RUN pip install ultralytics

# huggingface_hub 1.0 で削除された HfFolder を復元するシム（forgeの旧gradio用）。
# sitecustomize として配置し python 起動時に自動適用 → 新hf_hub と旧gradio を両立。
COPY hf_compat_shim.py /usr/local/lib/python3.10/dist-packages/sitecustomize.py

# Z-Image 透過統合拡張（z_image checkpoint を diffusers ZImagePipeline へ振り分け）
COPY zimage_diffusers_ext /app/webui/extensions-builtin/zimage-diffusers
# diffusers モデルキャッシュを data ボリュームに置き永続化＆再DL回避
ENV HF_HOME=/app/data/hf-cache

# VAE Baking Toolのコピー
COPY vae_bake.py /app/vae_bake.py
RUN chmod +x /app/vae_bake.py

# ControlNet fix for txt2img + Tile (HiRes Fix support)
COPY controlnet.py /app/webui/extensions-builtin/sd_forge_controlnet/scripts/controlnet.py

# forge の huggingface_hub ダウンロードパッチ(Windowsロングパス用)を前方互換化。
# 固定シグネチャが hf_hub 0.34.4 の etag 引数を弾き diffusers の Z-Image DL が落ちるため、
# *args/**kwargs で透過するよう差し替える。
COPY patch_basic.py /app/webui/modules_forge/patch_basic.py

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
  chown -R 1000:1000 $USERDATA_DIR || true; \
  sed -i 's/\\r$//' /app/webui/requirements_versions.txt; \
  sed -i -E -e 's/^diffusers==.*/diffusers>=0.36.0/' -e 's/^transformers==.*/transformers>=4.51.0/' -e 's/^peft==.*/peft>=0.17.0/' -e 's/^accelerate==.*/accelerate>=1.2.0/' -e 's/^kornia==.*/kornia>=0.7.3/' -e 's/^huggingface[_-]hub==.*/huggingface_hub==0.34.4/' -e 's/^safetensors==.*/safetensors>=0.4.3/' /app/webui/requirements_versions.txt; \
  civitconfig default --api-key $CIVITAI_TOKEN || true; \
  civitconfig alias --add @lora $USERDATA_DIR/models/Lora || true; \
  civitconfig alias --add @vae $USERDATA_DIR/models/VAE || true; \
  civitconfig alias --add @embed $USERDATA_DIR/models/text_encoder || true; \
  civitconfig alias --add @checkpoint $USERDATA_DIR/models/Stable-diffusion || true; \
  su webui -c \"/app/webui/webui.sh --gradio-allowed-path . --data-dir $USERDATA_DIR $ARGS\""]