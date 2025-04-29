import os
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Name of the folder where the model and tokenizer are stored
MODEL_DIR_NAME = "fine-tuned-model"

# 3. Build the full path to the model/tokenizer folder
MODEL_DIR = os.path.join(BASE_DIR, MODEL_DIR_NAME)
TOKENIZER_PATH = os.path.join(MODEL_DIR, "fine-tuned-tokenizer")
MODEL_PATH = os.path.join(MODEL_DIR, "fine-tuned-model")

# 4. Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

# 5. Load the model
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

# ---------------- Streamlit UI ----------------

# Display the app title
st.title("Bilingual Bridge: Fine-tuned English to Spanish Translation")

# Show developer's name
st.markdown("Developed by **Md Emon Hasan**.")

# Add a separator line
st.markdown("---")

# Text area for user input
input_text = st.text_area("Enter text in English:", "Hello, how are you today?")

# Translation button
if st.button("Translate"):
    if input_text:
        # 6. Tokenize the input text
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        # 7. Generate translation from the model
        translated_tokens = model.generate(
            inputs["input_ids"],
            max_length=256,
            num_beams=8,
            early_stopping=False
        )

        # 8. Decode tokens back to Spanish text
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

        # 9. Display the translated text
        st.subheader("Translated text in Spanish:")
        st.write(translated_text)
    else:
        # If user input is empty
        st.error("Please enter text to translate.")

# Add another separator line
st.markdown("---")