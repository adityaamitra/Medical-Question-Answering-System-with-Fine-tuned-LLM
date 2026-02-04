import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import pandas as pd

MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"

# Available models
AVAILABLE_MODELS = {
    "1": {
        "name": "Final Training Checkpoint (Step 500)",
        "path": "./results (1)/medical_model_final/checkpoint-500"
    },
    "2": {
        "name": "Best Hyperparameter Config 1 (Baseline - ROUGE-1: 0.3234)",
        "path": "./results (1)/best_model_config1"
    },
    "3": {
        "name": "Hyperparameter Config 2 (Higher LR - ROUGE-1: 0.3190)",
        "path": "./results (1)/best_model_config2"
    },
    "4": {
        "name": "Hyperparameter Config 3 (Lower LR - ROUGE-1: 0.3140)",
        "path": "./results (1)/best_model_config3"
    }
}

def load_model(model_path):
    """Load model and tokenizer"""
    print(f"\n📦 Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    print("✅ Model loaded!")
    return model, tokenizer

def ask_medical_model(model, tokenizer, question):
    """Generate answer"""
    prompt = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{question}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            repetition_penalty=1.2,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("assistant")[-1].strip()

# ===== MAIN =====
if __name__ == "__main__":
    print("=" * 70)
    print("🏥 Medical Q&A Model Selector")
    print("=" * 70)
    
    # Show available models
    print("\nAvailable models:")
    for key, info in AVAILABLE_MODELS.items():
        print(f"  {key}. {info['name']}")
    
    # Select model
    choice = input("\nSelect model (1-4) or press Enter for default (1): ").strip()
    if not choice:
        choice = "1"
    
    if choice not in AVAILABLE_MODELS:
        print("❌ Invalid choice. Using default (1).")
        choice = "1"
    
    selected = AVAILABLE_MODELS[choice]
    print(f"\n✅ Selected: {selected['name']}")
    
    # Load model
    model, tokenizer = load_model(selected['path'])
    
    # Test questions
    print("\n" + "=" * 70)
    print("🧪 Testing Model")
    print("=" * 70)
    
    test_questions = [
        "What are the symptoms of diabetes?",
        "How is high blood pressure treated?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. ❓ {question}")
        print("-" * 70)
        answer = ask_medical_model(model, tokenizer, question)
        print(f"💡 {answer}\n")
    
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