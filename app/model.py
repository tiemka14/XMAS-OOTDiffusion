# app/model.py

import torch
from PIL import Image
from diffusers import StableDiffusionIDMInpaintPipeline


class IDMVTONModel:
    def __init__(self, device="cuda"):
        print("Loading IDM-VTON model...")

        self.pipe = StableDiffusionIDMInpaintPipeline.from_pretrained(
            "yisol/IDM-VTON",
            torch_dtype=torch.float16,
            safety_checker=None
        ).to(device)

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
