import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import gradio as gr
import warnings
import os

warnings.filterwarnings('ignore')
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"
LORA_WEIGHTS_PATH = "./results (1)/medical_model_final/checkpoint-500"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

peft_config = PeftConfig.from_pretrained(LORA_WEIGHTS_PATH)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True
)

model = PeftModel.from_pretrained(base_model, LORA_WEIGHTS_PATH, is_trainable=False)
model.eval()
print("✅ Model ready!\n")

def medical_chat(question):
    if not question or question.strip() == "":
        return "⚠️ Please enter a medical question."
    
    try:
        prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{question}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                repetition_penalty=1.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("assistant")[-1].strip()
    
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Simple interface without advanced features
demo = gr.Interface(
    fn=medical_chat,
    inputs=gr.Textbox(
        label="Enter Your Medical Question",
        placeholder="e.g., What are the symptoms of diabetes?",
        lines=3
    ),
    outputs=gr.Textbox(
        label="AI Response",
        lines=12
    ),
    title="⚕️ Med-Llama 3.2 Medical Q&A",
    description="Fine-tuned medical AI assistant | Educational use only",
    examples=[
        "What are the symptoms of diabetes?",
        "How is high blood pressure treated?",
        "What causes asthma?",
        "What are the side effects of chemotherapy?",
        "How can I prevent heart disease?"
    ]
)

if __name__ == "__main__":
    print("🌐 Launching at http://127.0.0.1:7860\n")
    demo.launch(server_name="127.0.0.1", server_port=7860)