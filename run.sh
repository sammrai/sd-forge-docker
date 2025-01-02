#!/bin/bash
echo "Starting Stable Diffusion WebUI"

# VENV_DIRの親ディレクトリを取得
PARENT_DIR=$(dirname "${VENV_DIR}")

# 親ディレクトリの所有権を変更
chown -R webui:webui "${PARENT_DIR}"

# webuiユーザーでリポジトリをクローン
su webui -c "git clone https://github.com/lllyasviel/stable-diffusion-webui-forge.git ${PARENT_DIR}"

# webuiユーザーでファイルをコピー
su webui -c "cp /webui-user.sh ${PARENT_DIR}"

# WebUIの起動
exec su webui -c "/app/sd-webui/webui.sh $ARGS"
