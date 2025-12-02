# XMAS-OOTDiffusion

This project demonstrates a Runpod serverless handler that loads a Stable Diffusion XL Inpainting pipeline and runs a garment swap (IDM-VTON style) with a person and cloth images.

## Troubleshooting

If you see an error like:

```
Error no file named diffusion_pytorch_model.bin found in directory .../vae
```

This typically means the model repository does not include VAE weights (or the model archive is missing the `vae` subfolder). To resolve:

- Ensure the `MODEL_ID` points to a valid diffusers-compatible repo with `vae` and `unet` subfolders containing `diffusion_pytorch_model.safetensors` or `diffusion_pytorch_model.bin` files.
- If the model is private, provide an authentication token or use a public model.
	- To set an authentication token for private repos, export `HF_AUTH_TOKEN` in your environment. The handler will pass this to `from_pretrained`.
- Alternatively, set a fallback model using the `FALLBACK_MODEL_ID` env var (default: `stabilityai/stable-diffusion-xl-inpainting`).

## Local testing

Set `MODEL_ID` and `FALLBACK_MODEL_ID` as required, then run `local_test.py`:

```bash
export MODEL_ID="yisol/IDM-VTON"
export FALLBACK_MODEL_ID="stabilityai/stable-diffusion-xl-inpainting"
export HF_AUTH_TOKEN="<your-hf-token>"
python local_test.py
```

If the model fails to load due to missing files, the handler returns a JSON error explaining the issue instead of leaving the worker with a full stack trace.

## Inspect a model repo for missing weights

Use the included `inspect_model.py` to list the model's files and check for missing VAE/UNet weights:

```bash
python inspect_model.py --model_id yisol/IDM-VTON
```
If the script reports missing `vae/` files, you'll need to provide a compatible VAE or use a different model.
