from fastapi import FastAPI
from pydantic import BaseModel
from .rag_qa_system import RAGQueryProcessor
import logging

app = FastAPI()
query_processor = None
startup_error = None

try:
    query_processor = RAGQueryProcessor()
except Exception as e:
    startup_error = e
    logging.error(f"Failed to initialize RAGQueryProcessor: {e}")

class Question(BaseModel):
    text: str

class Answer(BaseModel):
    answer: str

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/ask", response_model=Answer)
def ask_question(question: Question):
    """
    Accepts a question and returns an answer by processing it 
    through the QA system.
    """
    if startup_error:
        return Answer(answer=f"Application failed to start: {startup_error}")
    answer_text = query_processor.answer_question(question.text)
    return Answer(answer=answer_text)
