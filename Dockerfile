FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

WORKDIR /app
RUN apt update && apt upgrade -y
RUN apt install -y wget git python3 python3-venv libgl1 libglib2.0-0 apt-transport-https libgoogle-perftools-dev bc python3-pip

# スクリプトをコピーして実行権限を付与
COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh

COPY webui-user.sh /webui-user.sh

# webui ユーザーの作成と権限設定
RUN useradd -m webui

# ENTRYPOINT
ENTRYPOINT ["/app/run.sh"]
