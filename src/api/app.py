from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from src.model.inference import load_model
from src.evaluation.tracking import tracker
from src.config.logging_config import logger

bot = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bot
    logger.info("Starting API server...")
    bot = load_model()
    yield
    tracker.save()
    logger.info("API server shutdown")


app = FastAPI(
    title="Customer Support Chatbot API",
    version="1.0.0",
    lifespan=lifespan
)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    question: str
    response: str
    latency_ms: float = None


@app.get("/")
def root():
    return {"status": "ok", "model": "customer-support-chatbot"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": bot is not None,
        "total_requests": tracker.model_metrics.total_inferences,
        "avg_latency_ms": round(tracker.model_metrics.avg_latency_ms, 2),
        "error_rate": round(tracker.model_metrics.errors / max(1, tracker.model_metrics.total_inferences), 4)
    }


@app.get("/metrics")
def metrics():
    return tracker.get_summary()


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if bot is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    import time
    start = time.time()
    response = bot.chat(request.question)
    latency = (time.time() - start) * 1000
    
    return ChatResponse(
        question=request.question,
        response=response,
        latency_ms=round(latency, 2)
    )
