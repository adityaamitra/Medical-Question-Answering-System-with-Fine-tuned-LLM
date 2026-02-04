# ⚕️ Med-Llama 3.2: Medical Q&A Fine-Tuning

## 📋 Overview

Med-Llama 3.2 is a lightweight, fine-tuned Large Language Model (LLM) designed to answer medical questions with improved accuracy and safety compared to base models. Built upon the Llama-3.2-1B-Instruct architecture, this model utilizes LoRA (Low-Rank Adaptation) for efficient training on the MedQuAD dataset.

The project encompasses the entire pipeline: data preprocessing with safety injections, model fine-tuning, hyperparameter optimization, quantitative evaluation (ROUGE metrics), and a deployed interactive chat interface.

## 🚀 Key Features

- **Efficient Fine-Tuning**: Uses LoRA to train only a fraction of parameters (approx. 0.9%), enabling training on consumer GPUs (e.g., T4).
- **Safety First**: Automatically injects medical disclaimers into training data to ensure the model advises consulting professionals.
- **Robust Training**: Implements NEFTune (Noisy Embeddings Fine Tuning) to prevent overfitting and improve generalization.
- **Hyperparameter Optimization**: Includes a comparative study of Learning Rates and LoRA Ranks to find the optimal configuration.
- **Interactive UI**: Built-in Gradio web interface for real-time interaction.

## 🛠️ Tech Stack

- **Base Model**: unsloth/Llama-3.2-1B-Instruct
- **Libraries**: transformers, peft, datasets, trl, accelerate, evaluate
- **Optimization**: Quantization (Float16), LoRA, NEFTune
- **Interface**: Gradio

## 📊 Dataset

The model is trained on the **MedQuAD dataset** (keivalya/MedQuAD-MedicalQnADataset).

- **Structure**: Question-Answer pairs regarding diseases, drugs, and medical procedures.
- **Preprocessing**:
  - Formatted into Llama-3 specific chat templates (`<|start_header_id|>user...`).
  - **Safety Injection**: Appended "Disclaimer: Consult a healthcare professional for medical advice" to all training answers.
  - **Splitting**: 80% Train, 10% Validation, 10% Test.

## ⚙️ Methodology

### 1. Model Configuration

We utilized Parameter-Efficient Fine-Tuning (PEFT) with the following LoRA configuration:

- **Rank (r)**: 16
- **Alpha**: 32
- **Target Modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj (All linear layers)

### 2. Hyperparameter Optimization

Three configurations were tested to maximize ROUGE-1 scores:

| Config | Learning Rate | LoRA Rank | Result (ROUGE-1) |
|--------|---------------|-----------|------------------|
| 1 (Baseline) | 1e-4 | 16 | 0.3234 (Best) |
| 2 (Aggressive) | 2e-4 | 8 | 0.3190 |
| 3 (Conservative) | 5e-5 | 32 | 0.3140 |

## 📈 Performance Results

The fine-tuned model demonstrated significant improvement over the base Llama 3.2 1B model on the test set.

| Metric | Base Model | Fine-Tuned (Medical) | Improvement |
|--------|------------|----------------------|-------------|
| ROUGE-1 | 0.2547 | 0.3359 | 🟢 +8.12% |
| ROUGE-2 | 0.0455 | 0.1884 | 🟢 +14.29% |
| ROUGE-L | 0.1226 | 0.2367 | 🟢 +11.41% |

### Key Findings:
- The fine-tuned model allows for more concise, medically relevant answers.
- The model successfully learned to append safety disclaimers to high-risk responses.

## 💻 Installation & Usage

### Prerequisites

```bash
pip install transformers peft accelerate datasets trl bitsandbytes evaluate rouge_score huggingface_hub gradio
```

### Loading the Model

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load Base Model
base_model_id = "unsloth/Llama-3.2-1B-Instruct"
base_model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# Load Fine-Tuned Adapters (Path to your saved checkpoint)
model = PeftModel.from_pretrained(base_model, "./medical_model_final")
```

### Inference Example

```python
def ask_doctor(question):
    prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("assistant")[-1].strip()

print(ask_doctor("What are the symptoms of Diabetes?"))
```

## 🖥️ Interactive Demo

The project includes a Gradio interface for easy testing.

```python
# Run the last cell in the notebook to launch
demo.launch(share=True)
```

## ⚠️ Medical Disclaimer

**This model is for educational and research purposes only.**

It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

## 📜 License

This project utilizes the Llama 3.2 community license and the MedQuAD dataset license.