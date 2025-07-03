# api.py
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

@app.post("/summary")
async def summarize(file: UploadFile):
    from utils import extract_text, generate_summary
    text = extract_text(await file.read())
    summary = generate_summary(text)
    return {"summary": summary}

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
