from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import httpx

app = FastAPI(title="LexiAI Backend")

ANTHROPIC_API_KEY = "YOUR_ANTHROPIC_API_KEY"
CLAUDE_MODEL = "claude-sonnet-4-5"

class ChatRequest(BaseModel):
    user_id: str
    message: str

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
    content = await file.read()
    # TODO: parse document & send to vector DB
    return {"filename": file.filename, "size": len(content)}
