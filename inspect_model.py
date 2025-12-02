"""
inspect_model.py - Utility to check for required files in a Hugging Face model repo.

Usage:
    python inspect_model.py --model_id yisol/IDM-VTON

This requires `huggingface_hub` to be installed.
"""
import argparse
import os
import sys
from huggingface_hub import list_repo_files


def check_model(model_id: str):
    try:
        files = list_repo_files(model_id)
    except Exception as e:
        print(f"Failed to list files for model {model_id}: {e}")
        return 1

    # Look for VAE, UNet, and other expected files
    have_vae = any(p.startswith("vae/") for p in files)
    have_unet = any(p.startswith("unet/") for p in files)

    print(f"Model: {model_id}")
    print(f"Has 'vae/' folder: {have_vae}")
    print(f"Has 'unet/' folder: {have_unet}")

    vae_diffusion = [p for p in files if p.startswith("vae/") and (p.endswith(".safetensors") or p.endswith(".bin"))]
    unet_diffusion = [p for p in files if p.startswith("unet/") and (p.endswith(".safetensors") or p.endswith(".bin"))]

    print('VAE distribution files found:')
    for f in vae_diffusion:
        print('  ', f)

    print('UNet distribution files found:')
    for f in unet_diffusion:
        print('  ', f)

    if not vae_diffusion:
        print('\nNo VAE weights found. This is likely why `from_pretrained` fails with missing file errors.')
        return 2

    if not unet_diffusion:
        print('\nNo UNet weights found. the model will likely fail to load.')
        return 3

    print('\nModel looks like it contains the expected weights.')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', default=os.environ.get('MODEL_ID', 'yisol/IDM-VTON'))
    args = parser.parse_args()
    sys.exit(check_model(args.model_id))
