import runpod
from app.inference import tryon   # your function

def handler(job):
    inp = job["input"]
    person = inp["person_image"]
    cloth = inp["cloth_image"]
    output = tryon(person, cloth)
    return {"result": output}

runpod.serverless.start({"handler": handler})
