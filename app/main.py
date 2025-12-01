# app/main.py

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import base64, io

from app.model import IDMVTONModel

app = FastAPI(title="IDM-VTON Try-On API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = IDMVTONModel(device="cuda")


@app.post("/tryon")
async def tryon(person: UploadFile = File(...), cloth: UploadFile = File(...)):
    person_img = Image.open(io.BytesIO(await person.read())).convert("RGB")
    cloth_img = Image.open(io.BytesIO(await cloth.read())).convert("RGB")

    output = model.run(person_img, cloth_img)

    buffer = io.BytesIO()
    output.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {"result": encoded}
