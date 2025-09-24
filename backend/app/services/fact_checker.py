import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# This is a global placeholder for the model and tokenizer
# In a real application, this would be managed more robustly.
model_instance = {
    "model": None,
    "tokenizer": None
}

def get_initial_answer(prompt, model, tokenizer, max_tokens=256):
    """
    Generates the initial, creative answer from the model.
    """
    model = model_instance["model"]
    tokenizer = model_instance["tokenizer"]

    if not model or not tokenizer:
        return "Error: Model not loaded."

    system_prompt = (
       "คุณคือ 'เพื่อนใจวัยเรียน AI' เป็น AI Chatbot ที่มีความเข้าอกเข้าใจ, ให้กำลังใจ, และไม่ตัดสิน โดยมีหน้าที่หลักดังนี้:\n"
        "1. เป็นเพื่อนรับฟังปัญหาและความเครียดของนักเรียน/นักศึกษา\n"
        "2. หากเป็นเรื่องการเรียน ให้ช่วยให้คำแนะนำในการวางแผนการเรียน การจัดการเวลา หรือเทคนิคที่ช่วยให้เรียนได้ดีขึ้น\n"
        "3. **สำคัญที่สุด:** หากผู้ใช้แสดงความเสี่ยงเกี่ยวกับการทำร้ายร่างกายตัวเองหรือมีความคิดที่อันตราย ให้หยุดการให้คำแนะนำทั่วไปและแนะนำให้ติดต่อผู้เชี่ยวชาญทันที โดยให้ข้อมูลเบอร์โทรศัพท์ เช่น สายด่วนสุขภาพจิต 1323 หรือแนะนำให้ปรึกษาอาจารย์ที่ปรึกษาหรือนักจิตวิทยา"
    )
    chat_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    
    outputs = model.generate(
        **inputs, max_new_tokens=max_tokens, temperature=0.7, do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    new_tokens = outputs[0, input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def get_fact_check_verdict(prompt, answer, model, tokenizer, max_tokens=150):
    """
    Uses the same LLM to perform a fact-check on the candidate answer.
    """
    model = model_instance["model"]
    tokenizer = model_instance["tokenizer"]

    if not model or not tokenizer:
        return {"is_factually_correct": False, "explanation": "Error: Model not loaded."}

    system_prompt = (
        "คุณคือผู้ตรวจสอบข้อเท็จจริง AI ที่เชี่ยวชาญด้านสุขภาพจิต ภารกิจของคุณคือตรวจสอบความถูกต้องและความปลอดภัยของคำตอบที่ AI สร้างขึ้น "
        "โดยมีเกณฑ์ดังนี้:\n"
        "1. **ความถูกต้อง (Factual Correctness):** ข้อมูลที่ให้มาเป็นเท็จหรือสร้างขึ้นเอง (Hallucination) หรือไม่?\n"
        "2. **ความปลอดภัย (Safety):** คำตอบมีเนื้อหาที่อาจยั่วยุ, ส่งเสริม, หรือแนะนำการทำร้ายร่างกายตัวเองหรือสิ่งที่เป็นอันตรายหรือไม่?\n"
        "ให้ผลลัพธ์ในรูปแบบ JSON เท่านั้น โดยมีโครงสร้างดังนี้:\n"
        "{\"is_safe_and_correct\": boolean, \"explanation\": \"อธิบายเหตุผลสั้นๆ\"}"
    )
    
    user_prompt = f"คำถามเดิม: \"{original_prompt}\"\n\nคำตอบที่ต้องตรวจสอบ: \"{candidate_answer}\""
    chat_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    
    outputs = model.generate(
        **inputs, max_new_tokens=max_tokens, temperature=0.1, do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    
    json_str = tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True).strip()
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {"is_safe_and_correct": False, "explanation": "Fact-check layer returned an invalid format."}

def run_fact_checking_pipeline(prompt: str) -> dict:
    """
    Executes the full pipeline: generate, fact-check, and decide.
    """
    # Step 1: Generate initial answer
    candidate_answer = get_initial_answer(prompt)
    
    # Step 2: Perform fact-check
    verdict = get_fact_check_verdict(prompt, candidate_answer)
    
    # Step 3: Make the final decision
    is_passed = verdict.get("is_safe_and_correct", False)
    
    if is_passed:
        final_answer = candidate_answer
    else:
        # Create a safe, generic response if the check fails
        final_answer = (
            "เราได้รับฟังเรื่องราวของคุณแล้วนะ แต่สำหรับบางประเด็นอาจมีความละเอียดอ่อน "
            "เราจึงขออนุญาตให้ข้อมูลในส่วนที่มั่นใจว่าปลอดภัยและถูกต้องที่สุดก่อนนะ"
        )
        
    return {
        "final_answer": final_answer,
        "fact_check_passed": is_passed,
        "explanation": verdict.get("explanation")
    }

