# app/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import base64, io

from app.model import IDMVTONModel
import os

app = FastAPI(title="IDM-VTON Try-On API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.model = None
app.state.model_loaded = False
app.state.model_error = None


@app.on_event("startup")
async def load_model_on_startup():
    # Load the heavy model in the background during startup so server binds quickly
    device = os.getenv("DEVICE", "cuda")
    try:
        model = IDMVTONModel(device=device)
        app.state.model = model
        app.state.model_loaded = True
    except Exception as exc:
        app.state.model_error = str(exc)
        # Keep server alive; readiness will be false until model loads correctly


@app.post("/tryon")
async def tryon(person: UploadFile = File(...), cloth: UploadFile = File(...)):
    if not app.state.model_loaded:
        raise HTTPException(status_code=503, detail={
            "error": "Model not loaded yet",
            "details": app.state.model_error,
        })

    person_img = Image.open(io.BytesIO(await person.read())).convert("RGB")
    cloth_img = Image.open(io.BytesIO(await cloth.read())).convert("RGB")

    output = model.run(person_img, cloth_img)

    buffer = io.BytesIO()
    output.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {"result": encoded}



@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": app.state.model_loaded,
        "model_error": app.state.model_error,
    }


@app.get("/ready")
async def ready():
    # Ready only when model is loaded
    if app.state.model_loaded:
        return {"status": "ready"}
    return {"status": "not ready", "model_error": app.state.model_error}
