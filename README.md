🚀 Fine-Tuning GPT-2 on a Custom Text Dataset

This repository contains a complete workflow for fine-tuning the GPT-2 language model using HuggingFace Transformers, built directly from the included Jupyter Notebook:
📘 Foundational_model_gpt2.ipynb

The project demonstrates how to:

Prepare and tokenize custom datasets

Configure data loaders and training pipeline

Fine-tune GPT-2 using Trainer API

Save and reuse a domain-adapted model

Generate text using the fine-tuned model

📌 Features

✔️ Load GPT-2 tokenizer & model
✔️ Automatic dataset downloading (via wget)
✔️ Custom data preparation & chunking
✔️ Data collator for Language Modeling
✔️ Trainer-based fine-tuning
✔️ Model checkpoint saving
✔️ Inference script for text generation

📂 Project Structure
📦 gpt2-fine-tuning
 ┣ 📄 Foundational_model_gpt2.ipynb
 ┣ 📁 data/
 ┃ ┗ 📄 training_data.txt   ← auto-downloaded
 ┣ 📁 output/
 ┃ ┗ 📄 fine_tuned_model/   ← saved model
 ┣ 📄 requirements.txt
 ┗ 📄 README.md

🧰 Dependencies
transformers
torch
wget
datasets


Install using:

pip install -r requirements.txt

🧠 Workflow Overview
🔹 1. Load Tokenizer
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

🔹 2. Download Training Dataset

The notebook automatically downloads the dataset using:

import wget
wget.download("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
              "data/training_data.txt")

🔹 3. Prepare Dataset

A custom function splits long text into model-friendly chunks:

def load_dataset(file_path, tokenizer, block_size=512):
    ...

🔹 4. Load GPT-2 Model
from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("gpt2")

🔹 5. Data Collator

Used for next-token prediction training:

from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

🔹 6. Training Configuration
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./output/fine_tuned_model",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=50,
    save_steps=200,
)

🔹 7. Train the Model
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

trainer.train()

🔹 8. Save the Fine-Tuned Model
trainer.save_model("./output/fine_tuned_model")

🤖 Text Generation (Inference)

Once the model is trained, you can generate any text:

from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("./output/fine_tuned_model")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

prompt = "Once upon a time"
inputs = tokenizer.encode(prompt, return_tensors="pt")

outputs = model.generate(
    inputs,
    max_length=150,
    temperature=0.7,
    top_p=0.95,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

📈 Results

After fine-tuning:

✨ The model generates more coherent text
✨ Understands domain-specific patterns better
✨ Produces longer and smoother sequences
✨ Adapts to writing style of the dataset
