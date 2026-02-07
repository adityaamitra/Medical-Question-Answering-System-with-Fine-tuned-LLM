# 🏥 Fine-Tuning Llama-3 for Medical Question Answering (MedQuAD + LoRA)

Transforming a general Large Language Model into a **domain-specific medical assistant** using parameter-efficient fine-tuning on a Kaggle GPU.

This project fine-tunes **Llama-3 Instruct** using the **MedQuAD medical Q&A dataset** and deploys the model through a **Gradio web interface** for real-time medical question answering.

---

## 🚀 Project Highlights

- Real medical dataset (MedQuAD from Hugging Face)
- Parameter-Efficient Fine-Tuning using LoRA
- Trained on Kaggle Tesla T4 GPU
- Proper train / validation / test split
- ROUGE evaluation against baseline model
- Error analysis and hyperparameter comparison
- Ethical safeguards with medical disclaimer
- Working Gradio chatbot interface
- Separate inference scripts for CPU, merged model, and local testing

---

## 🧠 Problem Statement

General-purpose LLMs lack specialized knowledge required for sensitive domains like healthcare. This project adapts a pre-trained LLM to answer **medical questions** more accurately while remaining computationally efficient.

---

## 🗂️ Repository Structure


.
├── fine-tuning-a-llm.ipynb # Complete Kaggle training notebook
├── results/ # Saved LoRA adapters and checkpoints
├── gradio_demo_final.py # Final Gradio UI for inference
├── inference_*.py # Different inference scripts (CPU, local, merged)
├── merge_and_save.py # Script to merge LoRA with base model
├── select_and_test.py # Testing script for model outputs
├── README.md
└── requirements.txt


---

## 📚 Dataset

**MedQuAD – Medical Question Answering Dataset**  
Source: Hugging Face (`keivalya/MedQuAD-MedicalQnADataset`)

- Real medical questions from patients
- Expert-written answers from trusted medical sources
- Ideal for instruction fine-tuning

---

## 🏗️ Model & Training Approach

| Component | Choice | Reason |
|---|---|---|
| Base Model | `unsloth/Llama-3.2-1B-Instruct` | Instruction-tuned, small enough for Kaggle GPU |
| Fine-Tuning Method | LoRA (PEFT) | Memory-efficient, fast training |
| GPU | Kaggle Tesla T4 | Free 30-hour access |
| Framework | Hugging Face Trainer | Logging, checkpointing, evaluation |
| Evaluation Metric | ROUGE | Suitable for Q&A generation |
| Safety | Medical disclaimer added to outputs | Ethical safeguard |

---

## ⚙️ Training Details

- Train/Val/Test split: 80/10/10 (seed=42)
- Max token length: 512
- Batch size: 4 with gradient accumulation
- Learning rate: `1e-4` (best after comparison)
- Scheduler: Cosine with warmup
- Steps: 500
- Mixed precision FP16

---

## 📊 Evaluation

The fine-tuned model was compared against the base Llama-3 model using ROUGE scores on the test set.

✅ Fine-tuned model produced answers significantly closer to expert medical responses.  
❌ Failure cases analyzed for rare diseases, ambiguous symptoms, and long questions.

---

## 🔬 Error Analysis

Observed failure patterns:

- Rare medical conditions not in training data
- Very long/ambiguous queries
- Questions needing up-to-date medical knowledge

Suggested improvements:

- More training steps
- Larger base model (3B/7B)
- Retrieval-Augmented Generation (RAG)

---
## 🖥️ Running the Gradio Medical Assistant

### Step 1 — Install dependencies

pip install -r requirements.txt

graphql
Copy code

### Step 2 — Merge LoRA with base model (run once)

python merge_and_save.py

shell
Copy code

### Step 3 — Launch Gradio app

python gradio_demo_final.py

yaml
Copy code

Open the local URL in your browser and ask medical questions.

---

## 💬 Example Query

What are the symptoms of iron deficiency anemia?

The model returns a structured medical answer along with a safety disclaimer.

---

## ⚠️ Ethical Considerations

- Does not provide medical diagnosis
- Includes automatic medical disclaimer
- Trained on verified medical sources
- For educational and research purposes only

---

## 🧪 Reproducibility

To reproduce training:

1. Upload the notebook to Kaggle
2. Enable GPU (Tesla T4)
3. Run cells in order
4. Download the `results/` folder

---

## 📈 What This Project Demonstrates

- Domain adaptation of LLMs
- Efficient fine-tuning with limited hardware
- Proper evaluation and analysis
- Turning research into a usable application

---

## 🧾 Requirements

See `requirements.txt`

---

## 📌 Future Improvements

- Use larger Llama-3 models
- Add retrieval (RAG)
- Deploy as API
- Expand dataset

---

## 👨‍💻 Author

Aditya Mitra

---

## 📄 License

For educational and research purposes only.
