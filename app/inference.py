from io import BytesIO
from PIL import Image
from oot_diffusion.pipeline import OOTDiffusionPipeline   # example
import torch


# Load model once at startup (important!)
device = "cuda"
pipe = OOTDiffusionPipeline.from_pretrained(
    "yisol/IDM-VTON",
    torch_dtype=torch.float16
).to(device)


def run_tryon(person_bytes, cloth_bytes):
    """Runs OOTDiffusion try-on inference."""

    person_img = Image.open(BytesIO(person_bytes)).convert("RGB")
    cloth_img = Image.open(BytesIO(cloth_bytes)).convert("RGB")

    result = pipe(person_img, cloth_img)

    # result should be a PIL image
    output_img = result.images[0]

    # convert to bytes
    buf = BytesIO()
    output_img.save(buf, format="PNG")
    return buf.getvalue()
