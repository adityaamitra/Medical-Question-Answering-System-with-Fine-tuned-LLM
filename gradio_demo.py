import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import gradio as gr

# ===== CONFIGURATION =====
MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"
LORA_WEIGHTS_PATH = "./results (1)/medical_model_final"

print("Loading model components...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, LORA_WEIGHTS_PATH)
model.eval()
print("✅ Model loaded!")

# ===== INFERENCE FUNCTION =====
def medical_chat(question):
    if not question or question.strip() == "":
        return "⚠️ Please enter a medical question."
    
    try:
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
    
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ===== CREATE GRADIO INTERFACE =====
demo = gr.Interface(
    fn=medical_chat,
    inputs=gr.Textbox(
        label="Enter Your Medical Question",
        placeholder="e.g., What are the symptoms of diabetes?",
        lines=3
    ),
    outputs=gr.Textbox(
        label="AI Response",
        lines=10
    ),
    title="⚕️ Med-Llama 3.2 Professional Assistant",
    description=(
        "**Fine-tuned medical Q&A AI** | "
        "Trained on MedQuAD dataset | "
        "⚠️ **Educational use only - not a substitute for professional medical advice**"
    ),
    examples=[
        ["What are the symptoms of diabetes?"],
        ["How is high blood pressure treated?"],
        ["What causes asthma?"],
        ["What are the side effects of chemotherapy?"],
        ["How can I prevent heart disease?"]
    ],
    theme="soft"
)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Launching Gradio Interface...")
    print("="*60)
    demo.launch(
        share=False,  # Set to True if you want a public URL
        server_name="127.0.0.1",
        server_port=7860
    )