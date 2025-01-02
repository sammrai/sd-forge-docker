#!/bin/bash
echo "Starting Stable Diffusion WebUI"

# VENV_DIRの親ディレクトリを取得
PARENT_DIR=$(dirname "${venv_dir}")

# 親ディレクトリの所有権を変更
chown -R webui:webui "${PARENT_DIR}"

# webuiユーザーでリポジトリをクローン
su webui -c "git clone https://github.com/lllyasviel/stable-diffusion-webui-forge.git ${PARENT_DIR}"

# WebUIの起動
exec su webui -c "/app/sd-webui/webui.sh $ARGS"
