import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import gradio as gr

# ===== CONFIGURATION =====
MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"
LORA_WEIGHTS_PATH = "./results (1)/medical_model_final"  # Path to your saved LoRA weights

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"  # Will use CPU if no GPU available
)

print("Loading LoRA weights...")
model = PeftModel.from_pretrained(base_model, LORA_WEIGHTS_PATH)
model.eval()  # Set to evaluation mode

print("✅ Model loaded successfully!")

# ===== INFERENCE FUNCTION =====
def ask_medical_model(question, max_tokens=300, temperature=0.7):
    """
    Generate medical answer for a given question
    """
    prompt = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{question}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            repetition_penalty=1.2,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("assistant")[-1].strip()

# ===== TEST INFERENCE =====
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing Inference...")
    print("="*60)
    
    test_questions = [
        "What are the symptoms of diabetes?",
        "How is high blood pressure treated?",
        "What causes asthma?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Question: {question}")
        print("-" * 60)
        answer = ask_medical_model(question)
        print(f"Answer: {answer[:300]}...")  # Print first 300 chars
        print()