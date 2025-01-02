#!/bin/bash
echo "Starting Stable Diffusion WebUI"

# VENV_DIRの親ディレクトリを取得
PARENT_DIR=$(dirname "${WEBUI_DIR}")

# 親ディレクトリの所有権を変更
cp -r /app/webui ${WEBUI_DIR}
chown -R webui:webui "${PARENT_DIR}"

# WebUIの起動
export venv_dir="-"
exec su webui -c "${WEBUI_DIR}/webui/webui.sh $ARGS"
