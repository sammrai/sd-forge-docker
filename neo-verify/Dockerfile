# Forge Neo 検証用イメージ（既存 sdui イメージとは完全に別物）
# Neo は Python 3.13 / CUDA 13 / torch 2.11 を要求するため、本体とは別系統で隔離する。
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Python 3.13 は Ubuntu 24.04 標準に無いため deadsnakes から導入
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.13 python3.13-venv python3.13-dev \
        git wget bc \
        build-essential \
        libgl1 libglib2.0-0 libgoogle-perftools-dev && \
    rm -rf /var/lib/apt/lists/*

# python3 / python を 3.13 に向ける
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1 && \
    update-alternatives --install /usr/bin/python  python  /usr/bin/python3.13 1 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.13

# Neo (neo ブランチ) を取得
RUN git clone -b neo --depth 1 \
        https://github.com/Haoming02/sd-webui-forge-classic.git /app/neo
WORKDIR /app/neo

# torch は cu130 index から。venv は named volume にして初回のみ依存導入する。
ENV TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
ENV VENV_DIR=/app/neo/venv
ENV no_proxy="localhost, 127.0.0.1, ::1"

EXPOSE 7690
ENTRYPOINT ["bash", "webui.sh"]
