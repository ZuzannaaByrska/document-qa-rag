from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.rag import answer_question
import os

app = FastAPI(
    title="Document Q&A RAG System",
    description="Ask questions about your PDF documents",
    version="1.0.0"
)

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: list

@app.get("/")
def root():
    return {"message": "Document Q&A RAG System is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    if not os.path.exists("chroma_db"):
        raise HTTPException(
            status_code=400,
            detail="No documents ingested yet. Run ingest.py first."
        )

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    result = answer_question(request.question)
    return result