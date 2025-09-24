from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import os

from app.models.chat import ChatRequest, ChatResponse
from app.services import fact_checker

# Create the FastAPI app instance
app = FastAPI(
    title="Student Mind Mate AI API",
    description="API for the AI-powered mental health support chatbot.",
    version="1.0.0"
)

# --- THE FIX for 404/Fetch Error ---
# Add CORS middleware to allow the frontend to communicate with this backend.
# This is a critical security feature in web applications.

# Define the list of origins that are allowed to make requests.
# In development, this is our React app's address.
origins = [
    "http://localhost:5173",
    "http://localhost:3000", # Often used by create-react-app
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # Allow specific origins
    allow_credentials=True,      # Allow cookies (if needed in the future)
    allow_methods=["*"],         # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],         # Allow all headers
)
# ------------------------------------

@app.on_event("startup")
def load_model():
    """
    This function runs when the server starts.
    It loads the large AI model into memory only once.
    """
    peft_model_id = os.getenv("MODEL_PATH", "Pathfinder9362/Student-Mind-Mate-AI-v2")
    print(f"Loading model from: {peft_model_id}")

    config = PeftConfig.from_pretrained(peft_model_id)
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
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(base_model, peft_model_id)

    # Store the loaded model and tokenizer in the service's global placeholder
    fact_checker.model_instance["model"] = model
    fact_checker.model_instance["tokenizer"] = tokenizer
    print("Model loaded successfully!")


@app.post("/chat", response_model=ChatResponse)
def handle_chat_request(request: ChatRequest):
    """
    This is the main endpoint that receives user messages.
    It uses the fact-checking pipeline to generate a safe response.
    """
    print(f"Received prompt: {request.prompt}")
    
    # Run the full pipeline from the service
    result = fact_checker.run_fact_checking_pipeline(request.prompt)
    
    return ChatResponse(
        answer=result["final_answer"],
        fact_check_passed=result["fact_check_passed"],
        explanation=result["explanation"]
    )