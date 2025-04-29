# Bilingual Bridge: English to Spanish Translation
![Image](https://github.com/user-attachments/assets/0cc3d6e3-fda2-4037-8fad-f08c60442448)
## 📌 Overview
A production-grade neural machine translation (NMT) system that translates English text to Spanish using a fine-tuned sequence-to-sequence transformer model. Deployed with a responsive Streamlit UI, containerized with Docker, and integrated with GitHub CI/CD for continuous delivery.

---

## 🎯 Project Objectives
- Build and fine-tune a translation model using Hugging Face Transformers.
- Follow modular, scalable, and maintainable code architecture.
- Add CI/CD pipelines using GitHub Actions.
- Containerize with Docker for cross-platform deployment.
- Integrate logging and caching.

---

## 🧠 Model Training
- **Dataset:** `opus_books` (English-Spanish parallel corpus)
- **Model:** `Helsinki-NLP/opus-mt-en-es`
- **Tokenizer:** SentencePiece tokenizer from pretrained model
- **Fine-tuning:**
  - Batch size: 16
  - Epochs: 3
  - Learning rate: 2e-5
  - Weight decay: 0.01
  - Early stopping used
- **Saving:** Trained model and tokenizer saved for deployment

---

## 🧱 Modular Codebase
```
Bilingual-Bridge/
│
├── .github/                           # GitHub specific files
│   └── workflows/
│       └── main.yml                   # GitHub Actions CI/CD workflow file
│
├── fine-tuned-model/                          
│   └── fine-tuned-model/
│   └── fine-tuned-tokenizer/      
│
├── tests/
│   └── data.py
│
├── src/                     
│   └── data.py
│   └── logger.py
│   └── model.py
│
├── app.py
├── train.py
├── setup.py
├── logs/                   # Training and app logs
├── app.png                 # Utility functions
├── LICENSE  
├── Dockerfile              # For containerization
├── requirements.txt        # Dependencies
└── README.md               # Project summary
```

---

## 🐳 Docker Support
```Dockerfile
FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build & run:
```bash
docker build -t bilingual-bridge .
docker run -p 8501:8501 bilingual-bridge
```

---

## 🚀 CI/CD (GitHub Actions)
`.github/workflows/ci.yml`:
- Trigger on push/pull
- Install dependencies
- Lint and test modules
- Build Docker image
- Optional: deploy to Hugging Face Spaces or Docker Hub

---

## 🖥️ Streamlit Web App
Features:
- Input box for English text
- Button-triggered translation
- Spanish output display
- Logs activity in real-time

Example:
```python
translated_tokens = model.generate(
    inputs['input_ids'],
    max_length=256,
    num_beams=8,
    early_stopping=False
)
```

---

## 📜 Logging
Custom logger at `utils/logger.py`:
```python
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
```
Used across all major modules for traceability.

---

## 🔐 Environment Variables (Optional)
Use `.env` to store sensitive paths or API keys.

---

## ✍️ Prepared by  

**Md Emon Hasan**  
📧 **Email:** iconicemon01@gmail.com  
💬 **WhatsApp:** [+8801834363533](https://wa.me/8801834363533)  
🔗 **GitHub:** [Md-Emon-Hasan](https://github.com/Md-Emon-Hasan)  
🔗 **LinkedIn:** [Md Emon Hasan](https://www.linkedin.com/in/md-emon-hasan)  
🔗 **Facebook:** [Md Emon Hasan](https://www.facebook.com/mdemon.hasan2001/)

---

## 🔮 Future Enhancements
- Add multilingual support (more language pairs)
- Quantize and optimize for edge devices

---

## 📂 How to Run (Locally)
```bash
git clone https://github.com/Md-Emon-Hasan/Bilingual-Bridge
cd bilingual-bridge
pip install -r requirements.txt
streamlit run app.py
```

---

## 📌 License
MIT License. Free to use with credit.