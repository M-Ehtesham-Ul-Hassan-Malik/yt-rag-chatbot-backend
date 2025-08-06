from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from rag_pipeline import get_rag_answer
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatInput(BaseModel):
    question: str
    video_id: str

@app.post("/ask")
async def ask_question(body: ChatInput):
    answer = get_rag_answer(body.question, body.video_id)
    return {"answer": answer}
