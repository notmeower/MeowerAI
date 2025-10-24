import os
import torch
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
from contextlib import asynccontextmanager
from collections import defaultdict
import time
from model import MeowerAI, load_model_safetensors
from main.config import api_config, model_config

model = None
tokenizer = None


# PRAISE BE


@app.exception_handler(424)
async def feelong_handler(request: Request, exc: HTTPException):
    return {"error": "feelong not found"}


# END OF PRAISE

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    try:
        metadata_path = f"{api_config.model_path}_metadata.json"
        
        if os.path.exists(metadata_path):
            print(f"Loading model from {api_config.model_path}")
            model = load_model_safetensors(MeowerAI, api_config.model_path)
        else:
            print(f"No model found at {metadata_path}, creating new model")
            from model import create_model
            model = create_model()
        
        if api_config.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = api_config.device
        
        model = model.to(device)
        model.eval()
        print(f"Model loaded on {device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        model = None
    
    yield
    
    model = None

app = FastAPI(
    title="MeowerAI API",
    description="API for the MeowerAI Language Model",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)

def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if api_config.api_key is None:
        return True
    if credentials is None:
        raise HTTPException(status_code=401, detail="API key required")
    if credentials.credentials != api_config.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

request_counts = defaultdict(list)

def check_rate_limit(request: Request):
    if api_config.max_requests_per_minute <= 0:
        return
    
    client_ip = request.client.host
    current_time = time.time()
    
    request_counts[client_ip] = [
        req_time for req_time in request_counts[client_ip]
        if current_time - req_time < 60
    ]
    
    if len(request_counts[client_ip]) >= api_config.max_requests_per_minute:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Commit suicide later."
        )
    
    request_counts[client_ip].append(current_time)

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.9
    do_sample: Optional[bool] = True

class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int

class ModelInfoResponse(BaseModel):
    model_name: str = "MeowerAI"
    parameters: int
    max_seq_length: int
    vocab_size: int
    device: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None
    )

@app.get("/info", response_model=ModelInfoResponse)
async def model_info(verified: bool = Depends(verify_api_key)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    from model import count_parameters
    return ModelInfoResponse(
        parameters=count_parameters(model),
        max_seq_length=model_config.max_seq_length,
        vocab_size=model_config.vocab_size,
        device=str(next(model.parameters()).device)
    )

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(
    request: GenerateRequest,
    verified: bool = Depends(verify_api_key),
    rate_limited: None = Depends(check_rate_limit)
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        print(f"Generating text for prompt: {request.prompt[:50]}...")
        
        prompt_tokens = [ord(c) for c in request.prompt]
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long)
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=request.do_sample
            )
        
        generated_tokens = generated_ids[0].tolist()
        generated_text = ''.join([chr(token) for token in generated_tokens if token < 128])
        
        if generated_text.startswith(request.prompt):
            generated_text = generated_text[len(request.prompt):]
        
        print(f"Generated {len(generated_tokens) - len(prompt_tokens)} tokens")
        
        return GenerateResponse(
            text=generated_text,
            tokens_generated=len(generated_tokens) - len(prompt_tokens)
        )
    except Exception as e:
        print(f"Generation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/generate/batch")
async def generate_batch(
    requests: List[GenerateRequest],
    verified: bool = Depends(verify_api_key),
    rate_limited: None = Depends(check_rate_limit)
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(requests) > 10:
        raise HTTPException(status_code=400, detail="Batch size too large (max 10)")
    
    print(f"Processing batch of {len(requests)} requests")
    
    responses = []
    for request in requests:
        try:
            response = await generate_text(request, verified=True, rate_limited=None)
            responses.append(response)
        except Exception as e:
            responses.append(GenerateResponse(
                text=f"Error: {str(e)}",
                tokens_generated=0
            ))
    
    return responses

@app.get("/stats")
async def get_stats(verified: bool = Depends(verify_api_key)):
    return {
        "total_requests": sum(len(requests) for requests in request_counts.values()),
        "active_clients": len(request_counts),
        "rate_limit": api_config.max_requests_per_minute
    }

@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return {"error": "Endpoint not found", "detail": str(exc)}

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: HTTPException):
    return {"error": "Internal server error", "detail": str(exc)}
  
def run_server():
    print(f"Starting MeowerAI API server on {api_config.host}:{api_config.port}")
    print(f"API Key required: {api_config.api_key is not None}")
    print(f"Rate limit: {api_config.max_requests_per_minute} requests/minute")
    print(f"god bless python interop")
    
    uvicorn.run(
        "api.server:app",
        host=api_config.host,
        port=api_config.port,
        workers=api_config.workers,
        reload=False
    )

if __name__ == "__main__":
    run_server()