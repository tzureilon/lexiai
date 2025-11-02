from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import httpx
from tempfile import NamedTemporaryFile
import os
from ocr import extract_text_from_file

app = FastAPI(title="LexiAI Backend")

import os

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
CLAUDE_MODEL = "claude-3-sonnet-20240229"

class ChatRequest(BaseModel):
    user_id: str
    message: str

class PredictionRequest(BaseModel):
    user_id: str
    case_details: str

class AnalyzeRequest(BaseModel):
    text: str

@app.post("/chat")
async def chat(req: ChatRequest):
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01"
    }
    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": req.message}
        ]
    }
    async with httpx.AsyncClient() as client:
        r = await client.post("https://api.anthropic.com/v1/messages", json=payload, headers=headers)
    return {"response": r.json()["content"][0]["text"]}

@app.post("/upload")
async def upload_doc(file: UploadFile = File(...)):
    try:
        with NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        text = extract_text_from_file(tmp_path)
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)

    return {"filename": file.filename, "extracted_text": text}

@app.post("/predict")
async def predict(req: PredictionRequest):
    """Return a stubbed prediction result. In future, integrate ML model."""
    # TODO: integrate with judicial prediction engine
    return {"probability": 0.5, "message": "Prediction stub"}

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    prompt = f"""Please analyze the following legal document and extract the following information:
- Document Type (e.g., Contract, Statement of Claim, etc.)
- Parties Involved (e.g., Plaintiff, Defendant, etc.)
- Key Dates

Here is the document text:
{req.text}

Format the output as a JSON object with the keys "document_type", "parties", and "dates".
"""

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01"
    }
    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    async with httpx.AsyncClient() as client:
        r = await client.post("https://api.anthropic.com/v1/messages", json=payload, headers=headers)

    return {"analysis": r.json()["content"][0]["text"]}
