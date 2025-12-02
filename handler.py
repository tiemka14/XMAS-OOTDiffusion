"""
handler.py - Runpod serverless handler for IDM-VTON model

Notes:
- This handler loads a Stable Diffusion XL inpainting pipeline from the
    `yisol/IDM-VTON` model. The model requires `diffusers>=0.25.0`.
"""

import base64
import io
from PIL import Image
import torch
try:
    from diffusers import StableDiffusionXLInpaintPipeline
    IDM_PIPELINE = StableDiffusionXLInpaintPipeline
    from diffusers import AutoencoderKL
except Exception as e:
    raise ImportError(
        "Failed to import StableDiffusionXLInpaintPipeline from diffusers. "
        "Please install or upgrade diffusers to >=0.25.0 (e.g., `pip install -U diffusers`)."
    ) from e
import runpod

# Lazy-load pipeline so imports succeed even if dependencies are missing until runtime
pipe = None

def get_pipe():
    global pipe
    if pipe is None:
        print("Loading IDM-VTON pipeline…")

        # Load an external VAE!
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            torch_dtype=torch.float16
        )

        pipe = IDM_PIPELINE.from_pretrained(
            "yisol/IDM-VTON",
            vae=vae,                        # ← IMPORTANT FIX
            torch_dtype=torch.float16,
            safety_checker=None
        ).to("cuda")

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

    # Ensure the pipeline is loaded before running
    result = get_pipe()(
        image=person_img,
        mask_image=cloth_img,
        cloth=cloth_img
    )

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
