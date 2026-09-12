#!/usr/bin/env python3
"""
VAE Baking Tool for Stable Diffusion Models
Standalone Python script that can be executed via docker exec
"""

import os
import re
import sys
import json
import shutil
import argparse
from pathlib import Path

def extract_model_info(model_name):
    """Extract model_id and version_id from model name"""
    pattern = r'^(.+)-mid_(\d+)-vid_(\d+)$'
    match = re.match(pattern, model_name)

    if not match:
        raise ValueError(f"Invalid model name format: {model_name}. Expected: name-mid_XXXXXX-vid_YYYYYY")

    return match.group(1), match.group(2), int(match.group(3))

def find_model_file(model_dir):
    """Find the safetensors model file in the directory"""
    safetensors_files = list(Path(model_dir).glob("*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")

    return str(safetensors_files[0])

def bake_vae_standalone(model_path, vae_path, output_path):
    """Standalone VAE baking using direct safetensors manipulation"""
    import safetensors.torch

    try:
        print(f"Loading model: {model_path}")
        model_data = safetensors.torch.load_file(model_path)

        print(f"Loading VAE: {vae_path}")
        vae_data = safetensors.torch.load_file(vae_path)

        # Remove existing VAE keys from model
        model_keys_to_remove = [k for k in model_data.keys() if k.startswith(('first_stage_model.', 'vae.'))]
        for key in model_keys_to_remove:
            del model_data[key]

        print(f"Removed {len(model_keys_to_remove)} existing VAE keys")

        # Add VAE data with proper prefixing.
        # loss.* is the training-time discriminator; model_ema.* is the EMA bookkeeping
        # (decay / num_updates). Neither belongs in a checkpoint -- forge would carry
        # them as unexpected keys under first_stage_model.
        vae_prefix = "first_stage_model."
        added_keys = 0
        for key, tensor in vae_data.items():
            if not key.startswith(('loss.', 'model_ema.')):
                model_data[vae_prefix + key] = tensor
                added_keys += 1

        print(f"Added {added_keys} VAE keys")

        print(f"Saving merged model to: {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        safetensors.torch.save_file(model_data, output_path)

        print("VAE baking completed successfully!")
        return True

    except Exception as e:
        print(f"Error during VAE baking: {e}")
        return False

def copy_metadata_files(original_model_dir, output_dir):
    """Copy every sidecar the original carried, except the weights themselves.

    The CSV, the civitai model_dict and the preview JPEGs under extra_data-vid_* all
    have to survive the swap -- forge and the model browser read them, and they are
    not re-downloadable once the original directory is moved away.
    """
    copied = 0
    for entry in sorted(os.listdir(original_model_dir)):
        if entry.endswith(".safetensors"):
            continue
        src = os.path.join(original_model_dir, entry)
        dst = os.path.join(output_dir, entry)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            copied += sum(len(f) for _, _, f in os.walk(src))
        else:
            shutil.copy2(src, dst)
            copied += 1
    print(f"Copied {copied} sidecar file(s) from the original model directory")

def apply_ownership(path, uid, gid):
    """Mirror the original directory's owner onto the replacement.

    This script runs as root inside the container while the model tree is owned by
    the host user; without this, forge's non-root user ends up with files it cannot
    manage and the directory stands out from every sibling.
    """
    os.chown(path, uid, gid)
    for root, dirs, files in os.walk(path):
        for name in dirs + files:
            os.chown(os.path.join(root, name), uid, gid)

def validate_model_directory(model_dir, model_name):
    """Validate that model directory has required CSV and JSON files"""

    safetensors_files = list(Path(model_dir).glob("*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")

    original_base_name = safetensors_files[0].stem

    csv_file = os.path.join(model_dir, f"{original_base_name}.csv")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Required CSV file not found: {csv_file}")

    _, model_id, version_id = extract_model_info(model_name)

    extra_data_dir = os.path.join(model_dir, f"extra_data-vid_{version_id}")
    model_dict_file = os.path.join(extra_data_dir, f"model_dict-mid_{model_id}-vid_{version_id}.json")

    if not os.path.exists(extra_data_dir):
        raise FileNotFoundError(f"Required extra_data directory not found: {extra_data_dir}")

    if not os.path.exists(model_dict_file):
        raise FileNotFoundError(f"Required model_dict JSON file not found: {model_dict_file}")

    print(f"✅ Validation passed:")
    print(f"  - CSV file: {csv_file}")
    print(f"  - Model dict: {model_dict_file}")

    return original_base_name

def main():
    parser = argparse.ArgumentParser(description="VAE Baking Tool for Stable Diffusion Models")
    parser.add_argument("model_folder", help="Model folder name in format: name-mid_XXXXXX-vid_YYYYYY")
    parser.add_argument("vae_name", help="VAE filename")
    parser.add_argument("--models-dir", default="/app/data/models/Stable-diffusion", help="Models directory path")
    parser.add_argument("--vae-dir", default="/app/data/models/VAE", help="VAE directory path")
    # Both of these deliberately sit outside --models-dir: forge scans that tree
    # recursively, so a work directory or a backup left inside it shows up in the
    # checkpoint dropdown as a second, stale copy of the model.
    parser.add_argument("--backup-dir", default="/app/data/model-backup", help="Where the original model is moved to")
    parser.add_argument("--work-dir", default="/app/data/.vae-bake-tmp", help="Scratch directory for the new model (same filesystem as --models-dir)")

    args = parser.parse_args()

    try:
        base_name, model_id, version_id = extract_model_info(args.model_folder)

        print(f"Original model: {base_name}")
        print(f"Model ID: {model_id}, Version ID: {version_id}")

        original_model_dir = os.path.join(args.models_dir, args.model_folder)
        original_base_name = validate_model_directory(original_model_dir, args.model_folder)

        stat = os.stat(original_model_dir)

        model_file = find_model_file(original_model_dir)
        vae_file = os.path.join(args.vae_dir, args.vae_name)

        if not os.path.exists(vae_file):
            raise FileNotFoundError(f"VAE file not found: {vae_file}")

        backup_path = os.path.join(args.backup_dir, args.model_folder)
        if os.path.exists(backup_path):
            raise FileExistsError(f"Backup already exists, refusing to overwrite it: {backup_path}")

        output_dir = os.path.join(args.work_dir, args.model_folder)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        output_path = os.path.join(output_dir, f"{original_base_name}.safetensors")

        print(f"Work directory: {output_dir}")

        copy_metadata_files(original_model_dir, output_dir)

        print(f"\nStarting VAE baking...")
        print(f"Model file: {model_file}")
        print(f"VAE file: {vae_file}")
        print(f"Output: {output_path}")

        if not bake_vae_standalone(model_file, vae_file, output_path):
            print(f"\n❌ VAE baking failed!")
            return 1

        print(f"\n✅ VAE baking completed successfully!")

        apply_ownership(output_dir, stat.st_uid, stat.st_gid)

        os.makedirs(args.backup_dir, exist_ok=True)
        print(f"\n📦 Moving original model to backup: {backup_path}")
        shutil.move(original_model_dir, backup_path)

        print(f"🔄 Swapping processed model to original location: {original_model_dir}")
        shutil.move(output_dir, original_model_dir)

        print(f"\n🎯 File swap completed!")
        print(f"📁 VAE-baked model now at: {original_model_dir}")
        print(f"💾 Original model backed up to: {backup_path}")
        print(f"🆔 Model ID: {model_id}, Version ID: {version_id}")
        print(f"\n⚠️  Verify before trusting it:")
        print(f"    vae_roundtrip.py <preview.jpeg> <new ckpt> <backup ckpt>   # saturation must recover")
        print(f"    verify_bake.py <new ckpt> <backup ckpt>                    # UNet/TE must be byte-identical")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
