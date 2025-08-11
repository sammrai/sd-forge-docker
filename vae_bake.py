#!/usr/bin/env python3
"""
VAE Baking Tool for Stable Diffusion Models
Standalone Python script that can be executed via docker exec
"""

import os
import sys
import re
import json
import shutil
import argparse
import tempfile
from pathlib import Path

def extract_model_info(model_name):
    """Extract model_id and version_id from model name"""
    pattern = r'^(.+)-mid_(\d+)-vid_(\d+)$'
    match = re.match(pattern, model_name)
    
    if not match:
        raise ValueError(f"Invalid model name format: {model_name}. Expected: name-mid_XXXXXX-vid_YYYYYY")
    
    base_name = match.group(1)
    model_id = match.group(2)
    version_id = int(match.group(3))
    
    return base_name, model_id, version_id

def find_model_file(model_dir):
    """Find the safetensors model file in the directory"""
    model_path = Path(model_dir)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    safetensors_files = list(model_path.glob("*.safetensors"))
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
        
        # Add VAE data with proper prefixing
        vae_prefix = "first_stage_model."
        added_keys = 0
        for key, tensor in vae_data.items():
            if not key.startswith('loss.'):  # Skip discriminator loss components
                new_key = vae_prefix + key
                model_data[new_key] = tensor
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

def copy_metadata_files(original_model_dir, output_dir, original_base_name, new_base_name, model_id, original_version_id, new_version_id):
    """Copy and update metadata files from original model"""
    
    # Copy original JSON file if it exists
    original_json = os.path.join(original_model_dir, f"{original_base_name}.json")
    new_json = os.path.join(output_dir, f"{new_base_name}.json")
    
    if os.path.exists(original_json):
        shutil.copy2(original_json, new_json)
        print(f"Copied JSON metadata: {original_json} -> {new_json}")
    else:
        with open(new_json, 'w') as f:
            json.dump({}, f)
        print(f"Created empty JSON file: {new_json}")
    
    # Copy and update CSV file if it exists
    original_csv = os.path.join(original_model_dir, f"{original_base_name}.csv")
    new_csv = os.path.join(output_dir, f"{new_base_name}.csv")
    
    if os.path.exists(original_csv):
        with open(original_csv, 'r') as f:
            csv_content = f.read()
        
        # Update filename references in CSV
        updated_content = csv_content.replace(f"{original_base_name}.safetensors", f"{new_base_name}.safetensors")
        
        with open(new_csv, 'w') as f:
            f.write(updated_content)
        print(f"Copied and updated CSV metadata: {original_csv} -> {new_csv}")
    else:
        # Create basic CSV file as fallback
        with open(new_csv, 'w') as f:
            f.write("file_name,model_type,base_model,trigger_words,description\n")
            f.write(f"{new_base_name}.safetensors,checkpoint,SD 1.5,,VAE baked version\n")
        print(f"Created basic CSV file: {new_csv}")
    
    # Copy model_dict JSON from original model's extra_data directory
    original_extra_data_dir = os.path.join(original_model_dir, f"extra_data-vid_{original_version_id}")
    new_extra_data_dir = os.path.join(output_dir, f"extra_data-vid_{new_version_id}")
    os.makedirs(new_extra_data_dir, exist_ok=True)
    
    original_model_dict = os.path.join(original_extra_data_dir, f"model_dict-mid_{model_id}-vid_{original_version_id}.json")
    new_model_dict = os.path.join(new_extra_data_dir, f"model_dict-mid_{model_id}-vid_{new_version_id}.json")
    
    if os.path.exists(original_model_dict):
        shutil.copy2(original_model_dict, new_model_dict)
        print(f"Copied model_dict JSON: {original_model_dict} -> {new_model_dict}")
    else:
        # Create basic model_dict as fallback
        model_dict = {
            "type": "Checkpoint",
            "base_model": "SD1.5",
            "description": "VAE baked version"
        }
        with open(new_model_dict, 'w') as f:
            json.dump(model_dict, f, indent=2)
        print(f"Created basic model_dict JSON: {new_model_dict}")

def validate_model_directory(model_dir, model_name):
    """Validate that model directory has required CSV and JSON files"""
    
    # Find safetensors file to get the base name
    safetensors_files = list(Path(model_dir).glob("*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")
    
    original_safetensors_file = safetensors_files[0]
    original_base_name = original_safetensors_file.stem
    
    # Check for CSV file
    csv_file = os.path.join(model_dir, f"{original_base_name}.csv")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Required CSV file not found: {csv_file}")
    
    # Check for JSON file (optional, will create empty one if missing)
    json_file = os.path.join(model_dir, f"{original_base_name}.json")
    json_exists = os.path.exists(json_file)
    
    # Extract model info for extra_data validation
    base_name, model_id, version_id = extract_model_info(model_name)
    
    # Check for extra_data directory and model_dict JSON
    extra_data_dir = os.path.join(model_dir, f"extra_data-vid_{version_id}")
    model_dict_file = os.path.join(extra_data_dir, f"model_dict-mid_{model_id}-vid_{version_id}.json")
    
    if not os.path.exists(extra_data_dir):
        raise FileNotFoundError(f"Required extra_data directory not found: {extra_data_dir}")
    
    if not os.path.exists(model_dict_file):
        raise FileNotFoundError(f"Required model_dict JSON file not found: {model_dict_file}")
    
    print(f"✅ Validation passed:")
    print(f"  - CSV file: {csv_file}")
    print(f"  - JSON file: {json_file} {'(exists)' if json_exists else '(will be created)'}")
    print(f"  - Model dict: {model_dict_file}")
    
    return original_base_name

def main():
    parser = argparse.ArgumentParser(description="VAE Baking Tool for Stable Diffusion Models")
    parser.add_argument("model_folder", help="Model folder name in format: name-mid_XXXXXX-vid_YYYYYY")
    parser.add_argument("vae_name", help="VAE filename")
    parser.add_argument("--models-dir", default="/app/data/models/Stable-diffusion", help="Models directory path")
    parser.add_argument("--vae-dir", default="/app/data/models/VAE", help="VAE directory path")
    
    args = parser.parse_args()
    
    try:
        # Extract model information
        base_name, model_id, original_version_id = extract_model_info(args.model_folder)
        
        print(f"Original model: {base_name}")
        print(f"Model ID: {model_id}, Version ID: {original_version_id}")
        
        # Set up paths
        original_model_dir = os.path.join(args.models_dir, args.model_folder)
        
        # Validate model directory structure
        original_base_name = validate_model_directory(original_model_dir, args.model_folder)
        
        model_file = find_model_file(original_model_dir)
        vae_file = os.path.join(args.vae_dir, args.vae_name)
        
        if not os.path.exists(vae_file):
            raise FileNotFoundError(f"VAE file not found: {vae_file}")
        
        # Output will use .processing suffix with same version
        output_dir_name = f"{args.model_folder}.processing"
        output_dir = os.path.join(args.models_dir, output_dir_name)
        output_filename = f"{original_base_name}.safetensors"
        output_path = os.path.join(output_dir, output_filename)
        
        # Keep the same base name for metadata files
        new_base_name = original_base_name
        
        print(f"Output folder: {output_dir_name}")
        print(f"Output filename: {output_filename}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Copy metadata files from original model
        copy_metadata_files(
            original_model_dir, output_dir, original_base_name, new_base_name,
            model_id, original_version_id, original_version_id  # Keep same version
        )
        
        # Perform VAE baking
        print(f"\nStarting VAE baking...")
        print(f"Model file: {model_file}")
        print(f"VAE file: {vae_file}")
        print(f"Output: {output_path}")
        
        success = bake_vae_standalone(model_file, vae_file, output_path)
        
        if success:
            print(f"\n✅ VAE baking completed successfully!")
            print(f"📁 Processing directory: {output_dir}")
            print(f"📄 VAE-baked model file: {output_path}")
            
            # Create backup directory
            backup_dir = os.path.join(args.models_dir, "bak")
            os.makedirs(backup_dir, exist_ok=True)
            
            # Move original model to backup
            backup_path = os.path.join(backup_dir, args.model_folder)
            print(f"\n📦 Moving original model to backup: {backup_path}")
            shutil.move(original_model_dir, backup_path)
            
            # Rename .processing folder to original name
            print(f"🔄 Swapping processed model to original location: {original_model_dir}")
            shutil.move(output_dir, original_model_dir)
            
            print(f"\n🎯 File swap completed!")
            print(f"📁 VAE-baked model now at: {original_model_dir}")
            print(f"💾 Original model backed up to: {backup_path}")
            print(f"🆔 Model ID: {model_id}, Version ID: {original_version_id}")
        else:
            print(f"\n❌ VAE baking failed!")
            return 1
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())