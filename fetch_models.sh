#!/bin/bash

# 必要なツールの確認
if ! command -v civitdl &> /dev/null || ! command -v yq &> /dev/null; then
    echo "Required tools (civitdl, yq) are not installed. Please install them."
    exit 1
fi

# 引数の確認
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <yaml_file> <base_directory> <parallel_downloads>"
    exit 1
fi

# 引数の取得
YAML_FILE="$1"
BASE_DIR="$2"
PARALLEL_DOWNLOADS="$3"

# PARALLEL_DOWNLOADSが0の場合、1に設定
if [ "$PARALLEL_DOWNLOADS" -eq 0 ]; then
    PARALLEL_DOWNLOADS=1
fi

# .env ファイルの読み込み
if [ -f .env ]; then
    source .env
else
    echo ".env file not found. Please create it with your CIVITAI_API_TOKEN."
    exit 1
fi

if [ -z "$CIVITAI_API_TOKEN" ]; then
    echo "CIVITAI_API_TOKEN not set in .env file."
    exit 1
fi

# 配置先ディレクトリの相対パス
VAE_DIR="${BASE_DIR}/models/VAE"
EMBEDDING_DIR="${BASE_DIR}/models/text_encoder"
CHECKPOINT_DIR="${BASE_DIR}/models/Stable-diffusion"
LORA_DIR="${BASE_DIR}/models/LoRA"
LYCORIS_DIR="${BASE_DIR}/models/LyCORIS"

# ダウンロード関数
download_model() {
    local id="$1"
    local type="$2"
    local target_dir="$3"

    mkdir -p "$target_dir"
    echo "Downloading model ID: $id of type: $type to $target_dir"
    civitdl "$id" "$target_dir" -k "$CIVITAI_API_TOKEN"
    
    # ダウンロードの成功確認
    if [ "$?" -ne 0 ]; then
        echo "Failed to download model ID: $id of type: $type"
    else
        echo "Successfully downloaded model ID: $id of type: $type"
    fi
}

export CIVITAI_API_TOKEN
export -f download_model

# YAMLをパースしてダウンロードキューを作成
yq -r '.models[] | "\(.id) \(.type)"' "$YAML_FILE" | while IFS=' ' read -r id type; do
    case "$type" in
        "vae") target_dir="$VAE_DIR" ;;
        "embed") target_dir="$EMBEDDING_DIR" ;;
        "checkpoint") target_dir="$CHECKPOINT_DIR" ;;
        "lora") target_dir="$LORA_DIR" ;;
        "lycoris") target_dir="$LYCORIS_DIR" ;;
        *) echo "Unknown type: $type" && continue ;;
    esac
    echo "$id $type $target_dir"
done | xargs -n 3 -P "$PARALLEL_DOWNLOADS" bash -c 'download_model "$0" "$1" "$2"'

echo "All downloads completed."
