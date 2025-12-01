# app/model.py

import torch
from PIL import Image


class IDMVTONModel:
    def __init__(self, device="cuda"):
        # Defer to CPU if CUDA is not available
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available; falling back to CPU")
            device = "cpu"

        print(f"Loading IDM-VTON model on {device}...")
        try:
            # Import diffusers internals at model load time to keep module import fast
            from diffusers import StableDiffusionIDMInpaintPipeline

            self.pipe = (
                StableDiffusionIDMInpaintPipeline.from_pretrained(
                    "yisol/IDM-VTON",
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    safety_checker=None,
                ).to(device)
            )
        except Exception as exc:
            # Surface the original exception to the caller for logging
            print(f"Failed to load model: {exc}")
            raise

    def run(self, person_img: Image.Image, cloth_img: Image.Image):
        """
        Run IDM-VTON try-on.
        """

        # Resize to model expected dimensions
        person_img = person_img.resize((512, 768))
        cloth_img = cloth_img.resize((512, 768))

        # IDM-VTON uses inpaint pipeline where garment is placed in a masked region
        result = self.pipe(
            image=person_img,
            mask_image=cloth_img,     # model uses garment as conditioning mask
            cloth=cloth_img           # custom argument used in some variants
        )

        output = result.images[0]
        return output
