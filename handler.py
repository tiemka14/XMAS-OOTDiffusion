import base64
import runpod
from app.inference import run_tryon


def handler(event):
    """
    RunPod Serverless Handler.
    Expects:
    {
        "input": {
            "person_image": "<base64>",
            "cloth_image": "<base64>"
        }
    }
    """

    inp = event["input"]

    person_b64 = inp["person_image"]
    cloth_b64 = inp["cloth_image"]

    # decode images
    person_bytes = base64.b64decode(person_b64)
    cloth_bytes = base64.b64decode(cloth_b64)

    # run your OOTDiffusion inference
    output_bytes = run_tryon(
        person_bytes=person_bytes,
        cloth_bytes=cloth_bytes
    )

    # encode output back to base64
    output_b64 = base64.b64encode(output_bytes).decode("utf-8")

    return {
        "result": output_b64
    }


runpod.serverless.start({"handler": handler})
