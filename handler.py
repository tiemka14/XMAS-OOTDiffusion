"""
handler.py - Runpod serverless handler for IDM-VTON model

Notes:
- This handler loads a Stable Diffusion XL inpainting pipeline from the
    `yisol/IDM-VTON` model. The model requires `diffusers>=0.25.0`.
"""

import base64
import io
import os
try:
    from PIL import Image
except Exception as e:
    raise ImportError(
        "Failed to import PIL (Pillow). Please install it, e.g., `pip install Pillow`."
    ) from e
import torch
import logging
logging.basicConfig(level=logging.INFO)
try:
    from diffusers import StableDiffusionXLInpaintPipeline
    IDM_PIPELINE = StableDiffusionXLInpaintPipeline
except Exception as e:
    raise ImportError(
        "Failed to import StableDiffusionXLInpaintPipeline from diffusers. "
        "Please install or upgrade diffusers to >=0.25.0 (e.g., `pip install -U diffusers`)."
    ) from e
import runpod

# Lazy-load pipeline so imports succeed even if dependencies are missing until runtime
pipe = None
MODEL_ID = os.environ.get("MODEL_ID", "yisol/IDM-VTON")
HF_AUTH_TOKEN = os.environ.get("HF_AUTH_TOKEN")
FALLBACK_MODEL_ID = os.environ.get("FALLBACK_MODEL_ID", "stabilityai/stable-diffusion-xl-inpainting")

def _log_missing_file_error(e: Exception, model_id: str) -> None:
    # Provide a clear log message for typical HF model missing-file errors
    logging.error("Failed to load model '%s': %s", model_id, str(e))
    logging.error("Common causes: model repo is missing files (e.g., VAE, UNet weights), the model is private, or you need a different revision/subfolder.\n"
                  "Check the model on Hugging Face and ensure the 'vae' and other subfolders contain the expected 'diffusion_pytorch_model(.safetensors|.bin)' files.")

def get_pipe():
    global pipe
    if pipe is None:
        # Try to load the user-specified model first. If it fails due to missing
        # files in the model repository, provide a clearer error and attempt
        # to fall back to a known working inpainting model.
        model_id = MODEL_ID
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            logging.info("Loading model '%s' on device '%s' (dtype=%s)", model_id, device, dtype)
            pipe = IDM_PIPELINE.from_pretrained(
                model_id,
                torch_dtype=dtype,
                safety_checker=None,
                use_auth_token=HF_AUTH_TOKEN,
            ).to(device)
        except EnvironmentError as e:
            _log_missing_file_error(e, model_id)
            # Attempt a fallback to a known inpainting model if available
            if FALLBACK_MODEL_ID and FALLBACK_MODEL_ID != model_id:
                try:
                    logging.info("Attempting to load fallback model '%s'", FALLBACK_MODEL_ID)
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    dtype = torch.float16 if device == "cuda" else torch.float32
                    logging.info("Loading fallback model '%s' on device '%s' (dtype=%s)", FALLBACK_MODEL_ID, device, dtype)
                    pipe = IDM_PIPELINE.from_pretrained(
                        FALLBACK_MODEL_ID,
                        torch_dtype=dtype,
                        safety_checker=None,
                        use_auth_token=HF_AUTH_TOKEN,
                    ).to(device)
                except Exception as e2:
                    _log_missing_file_error(e2, FALLBACK_MODEL_ID)
                    raise
            else:
                # Reraise the original error if fallback is disabled or identical
                raise
    return pipe


def b64_to_img(b64_str: str) -> Image.Image:
    data = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(data)).convert("RGB")


def handler(job: dict) -> dict:
    inp = job.get("input", {})
    person_b64 = inp.get("person")
    cloth_b64 = inp.get("cloth")
    if not person_b64 or not cloth_b64:
        return {"error": "Missing 'person' or 'cloth' in input."}

    person_img = b64_to_img(person_b64).resize((512, 768))
    cloth_img = b64_to_img(cloth_b64).resize((512, 768))

    # Ensure the pipeline is loaded before running. Return a descriptive
    # error if we fail to initialize or run the pipeline to avoid noisy
    # stack traces from the worker environment.
    try:
        pipe = get_pipe()
    except Exception as e:
        logging.exception("Failed to load the pipeline")
        return {"error": "Failed to load pipeline: %s" % str(e)}

    try:
        result = pipe(
            image=person_img,
            mask_image=cloth_img,
            cloth=cloth_img
        )
    except Exception as e:
        logging.exception("Error while running the pipeline")
        return {"error": "Pipeline execution failed: %s" % str(e)}

    out_img = result.images[0]
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    out_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {"result": out_b64}


if __name__ == "__main__":
    # Local testing example
    import json, sys
    if "--test_input" in sys.argv:
        idx = sys.argv.index("--test_input")
        test = json.loads(sys.argv[idx + 1])
        print(handler(test))
    else:
        runpod.serverless.start({"handler": handler})
