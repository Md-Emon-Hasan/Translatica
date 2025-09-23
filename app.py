import os
from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
import torch
from pathlib import Path

app = Flask(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dynamically get path relative to current file
BASE_DIR = Path(__file__).resolve().parent
tokenizer_path = BASE_DIR / "fine-tuned-model" / "fine-tuned-tokenizer"
model_path = BASE_DIR / "fine-tuned-model" / "fine-tuned-model"

# Load tokenizer and model
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
model.to(device)
model.eval()
print("Model & Tokenizer loaded.")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Empty input"}), 400

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

    # Generate translation
    translated_tokens = model.generate(
        inputs["input_ids"],
        max_length=256,
        num_beams=8,
        early_stopping=True,
    )

    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return jsonify({"translation": translated_text})

if __name__ == "__main__":
    app.run(debug=True)
