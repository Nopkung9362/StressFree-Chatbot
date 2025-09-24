from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import os

from app.models.chat import ChatRequest, ChatResponse
from app.services.fact_checker import run_fact_checking_pipeline, model_instance

# --- Configuration ---
MODEL_PATH = "Pathfinder9362/Student-Mind-Mate-AI-v2"
OFFLOAD_DIR = "./offload_main"
os.makedirs(OFFLOAD_DIR, exist_ok=True)

def _load_model():
    """
    Loads the PEFT model using 4-bit quantization.
    This function is called once at startup.
    """
    print(f"Loading model from: {MODEL_PATH}...")
    try:
        config = PeftConfig.from_pretrained(MODEL_PATH)
        base_model_name = config.base_model_name_or_path
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_enable_fp32_cpu_offload=True
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            offload_folder=OFFLOAD_DIR,
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.pad_token = tokenizer.eos_token

        model = PeftModel.from_pretrained(base_model, MODEL_PATH)
        
        # Store the loaded model and tokenizer in the global placeholder
        model_instance["model"] = model
        model_instance["tokenizer"] = tokenizer
        
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        # In a real app, you might want to handle this more gracefully
        raise RuntimeError(f"Could not load model: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code runs on startup
    _load_model()
    yield
    # This code runs on shutdown (optional)
    print("Shutting down...")
    model_instance["model"] = None
    model_instance["tokenizer"] = None


app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"status": "Student's Mind Mate AI Backend is running!"}


@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    """
    Main chat endpoint that receives user prompts and returns a fact-checked answer.
    """
    if not model_instance["model"] or not model_instance["tokenizer"]:
        raise HTTPException(status_code=503, detail="Model is not available or still loading.")
    
    if not request.user_prompt:
        raise HTTPException(status_code=400, detail="User prompt cannot be empty.")
        
    try:
        # In a real-world scenario with vLLM, this would be an API call:
        # result = await call_vllm_service(request.user_prompt)
        
        # For now, we call our local pipeline directly
        result = run_fact_checking_pipeline(request.user_prompt)
        
        return ChatResponse(
            final_answer=result["final_answer"],
            fact_check_passed=result["fact_check_passed"],
            explanation=result["explanation"]
        )
    except Exception as e:
        print(f"An error occurred during chat processing: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the request.")

