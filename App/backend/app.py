from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import json

from inference import generate_report

app = FastAPI(title="X-ray Report Generation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_images")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def home():
    return {"message": "X-ray Report Generation API is running"}


@app.post("/generate-report")
async def generate_report_api(
    frontal: UploadFile = File(...),
    lateral: UploadFile = File(None),
    problems: str = Form("[]")
):
    frontal_path = os.path.join(UPLOAD_DIR, frontal.filename)

    with open(frontal_path, "wb") as buffer:
        shutil.copyfileobj(frontal.file, buffer)

    lateral_path = None
    if lateral is not None:
        lateral_path = os.path.join(UPLOAD_DIR, lateral.filename)

        with open(lateral_path, "wb") as buffer:
            shutil.copyfileobj(lateral.file, buffer)

    try:
        selected_problems = json.loads(problems)

        result = generate_report(
            frontal_path=frontal_path,
            lateral_path=lateral_path,
            problems=selected_problems
        )

        return {
            "filename": frontal.filename,
            "matched_id": result["matched_id"],
            "selected_problems": selected_problems,
            "generated_report": result["report"],
            "reference_report": result["reference"]
        }

    except Exception as e:
        return {
            "filename": frontal.filename,
            "error": str(e)
        }