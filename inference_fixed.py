import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import warnings

# Suppress the warnings
warnings.filterwarnings('ignore')

# ===== CONFIGURATION =====
MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"
LORA_WEIGHTS_PATH = "./results (1)/medical_model_final/checkpoint-500"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

print("Loading LoRA configuration...")
peft_config = PeftConfig.from_pretrained(LORA_WEIGHTS_PATH)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,  # ✅ Fixed: Use 'dtype' instead of 'torch_dtype'
    device_map="auto",
    low_cpu_mem_usage=True
)

print("Loading and merging LoRA weights...")
# ✅ FIX: Load with is_trainable=False
model = PeftModel.from_pretrained(
    base_model, 
    LORA_WEIGHTS_PATH,
    is_trainable=False  # This prevents the KeyError
)
model.eval()

print("✅ Model loaded successfully!\n")

# ===== INFERENCE FUNCTION =====
def ask_medical_model(question, max_tokens=300, temperature=0.7):
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
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("assistant")[-1].strip()

# ===== TEST =====
if __name__ == "__main__":
    print("=" * 70)
    print("🧪 Testing Medical Q&A Model")
    print("=" * 70)
    
    test_questions = [
        "What are the symptoms of diabetes?",
        "How is high blood pressure treated?",
        "What causes asthma?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. ❓ Question: {question}")
        print("-" * 70)
        answer = ask_medical_model(question)
        print(f"💡 Answer:\n{answer}\n")
    
    # Interactive mode
    print("\n" + "=" * 70)
    print("💬 Interactive Mode (type 'quit' to exit)")
    print("=" * 70)
    
    while True:
        question = input("\n❓ Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not question:
            continue
        
        print("-" * 70)
        answer = ask_medical_model(model, tokenizer, question)
        print(f"💡 {answer}")