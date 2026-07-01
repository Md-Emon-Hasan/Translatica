# Translatica: English to Spanish Translation

[![CI/CD](https://github.com/Md-Emon-Hasan/Translatica/actions/workflows/main.yml/badge.svg)](https://github.com/Md-Emon-Hasan/Translatica/actions) [![Python](https://img.shields.io/badge/python-3.11-blue)](https://python.org) ![TypeScript](https://img.shields.io/badge/typescript-%23007ACC.svg?style=flat&logo=typescript&logoColor=white) [![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com) ![React](https://img.shields.io/badge/react-%2320232a.svg?style=flat&logo=react&logoColor=%2361DAFB) ![Vite](https://img.shields.io/badge/vite-%23646CFF.svg?style=flat&logo=vite&logoColor=white) ![TailwindCSS](https://img.shields.io/badge/tailwindcss-%2338B2AC.svg?style=flat&logo=tailwind-css&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white) ![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow) ![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Translatica** is a production-ready AI-powered **English → Spanish literary translation system** designed to preserve tone, context, and narrative style. It leverages a **LoRA-fine-tuned transformer (PEFT)** to deliver high-quality translations with low inference cost, supported by a modular NLP pipeline, BLEU-based evaluation, and a clean full-stack web interface.

The system is **Dockerized and deployment-ready**, and can be scaled as a **SaaS product** for publishers and content platforms—demonstrating strong expertise in **model optimization, end-to-end system design, and business-oriented AI engineering**.

[![Project demo video](https://github.com/user-attachments/assets/704e7d2d-af01-46ee-998f-7342735db1b1)](https://github.com/user-attachments/assets/704e7d2d-af01-46ee-998f-7342735db1b1)

![Project Screenshot](https://github.com/user-attachments/assets/ac3c656e-319e-4f0a-beaa-8aba201b9ef1)

---

## Live Demo

[**Try Translatica Live**](https://bilingual-bridge.onrender.com/)

---

## Technical Stack

| Component        | Technology                                   |
| ---------------- | -------------------------------------------- |
| **Training**     | PyTorch, Transformers, PEFT, LoRA, Datasets  |
| **Inference**    | FastAPI, Uvicorn, Pydantic                   |
| **Model**        | `t5-small` (LoRA fine-tuned, PEFT)           |
| **Frontend**     | React, Vite, Tailwind CSS                    |
| **Testing**      | Pytest (66 tests, 96% coverage)              |
| **CI/CD**        | GitHub Actions (Lint → Test → Docker Build)  |
| **Deployment**   | Docker, Render                               |

---

## System Architecture

Translatica follows a **modular monolithic architecture** that clearly separates training, inference, API, and frontend layers while maintaining simple deployment and strong production readiness.

### High-Level Architecture

```
┌──────────────────────────┐
│        Frontend UI       │
│   HTML + CSS + JS (UI)   │
└─────────────┬────────────┘
              │ HTTP Requests
              ▼
┌──────────────────────────┐
│       FastAPI Server     │
│  Routing + Validation    │
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│   Translation Service    │
│  Preprocess → Inference  │
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│       Model Manager      │
│  Load LoRA + Tokenizer   │
└─────────────┬────────────┘
              │
              ▼
┌────────────────────────────────────┐
│  LoRA Fine-Tuned Transformer Model │
│      t5-small (PEFT / LoRA)        │
└────────────────────────────────────┘
```

---

## Prerequisites

Before you begin, ensure you have the following installed:
- **Python 3.11+**
- **Node.js 18+** & **npm**
- **Git**

---

## Quick Start

The easiest way to run the full application (Frontend + Backend) is using the unified runner.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Md-Emon-Hasan/Translatica.git
   cd Translatica
   ```

2. **Setup Backend:**
   ```bash
   # Create virtual environment
   python -m venv venv

   # Activate it
   # Windows:
   venv\Scripts\activate
   # Mac/Linux:
   # source venv/bin/activate

   # Install dependencies
   pip install -r backend/requirements.txt
   ```

3. **Setup Frontend:**
   ```bash
   cd frontend
   npm install
   cd ..
   ```

4. **Run the Application:**
   ```bash
   # Make sure venv is active
   python run.py
   ```
   - **Frontend UI:** [http://localhost:5173](http://localhost:5173)
   - **Backend API:** [http://localhost:8000](http://localhost:8000)

---

## Project Structure

```
Translatica/
│
├── .github/                             # GitHub Configuration
│   └── workflows/
│       └── main.yml                     # CI/CD Pipeline Configuration
│
├── backend/                             # Backend Service (FastAPI & Training)
│   ├── app/                                 # Main Application Package
│   │   ├── api/                             # API Request Handlers
│   │   │   ├── __init__.py
│   │   │   └── routes.py                    # Endpoint Definitions
│   │   ├── core/                            # Core Infrastructure
│   │   │   ├── __init__.py
│   │   │   ├── config.py                    # Application Settings
│   │   │   ├── database.py                  # Database Connection Logic
│   │   │   └── model.py                     # ML Model Loading & Management
│   │   ├── models/                          # Data Models
│   │   │   ├── __init__.py
│   │   │   └── translation.py               # Database Schema Models
│   │   ├── services/                        # Business Logic Layer
│   │   │   ├── __init__.py
│   │   │   └── translation.py               # Translation Processing Service
│   │   ├── utils/                           # Utility Functions
│   │   │   ├── __init__.py
│   │   │   └── logger.py                    # Logging Configuration
│   │   ├── __init__.py
│   │   └── main.py                          # FastAPI Application Entry Point
│   ├── data/                                # Persistent Data Storage
│   │   └── translations.db                  # SQLite Database File
│   ├── fine-tuned-model/                    # Trained Model Artifacts
│   │   ├── fine-tuned-model/                # Model Weights and Config
│   │   │   ├── adapter_config.json
│   │   │   ├── adapter_model.safetensors
│   │   │   └── README.md
│   │   └── fine-tuned-tokenizer/            # Tokenizer Assets (T5 SentencePiece)
│   │       ├── tokenizer.json
│   │       └── tokenizer_config.json
│   ├── logs/                                # Application Logs
│   │   └── app.log
│   ├── notebook/                            # Jupyter Notebooks
│   │   └── Translatica_colab_t5.ipynb       # Fine-tuning notebook (Colab, t5-small)
│   ├── tests/                               # Test Suite
│   │   ├── __init__.py
│   │   ├── conftest.py                      # Test Fixtures
│   │   ├── test_api.py                      # API Endpoint Tests
│   │   ├── test_config.py                   # Config Tests
│   │   ├── test_main.py                     # App Initialization Tests
│   │   ├── test_model.py                    # Model Manager Tests
│   │   ├── test_services.py                 # Service Layer Tests
│   │   ├── test_training_data.py            # Training Data Tests
│   │   ├── test_training_logger.py          # Training Logger Tests
│   │   ├── test_training_model.py           # Training Model Tests
│   │   ├── test_training_train.py           # Training Script Tests
│   │   └── test_training_trainer.py         # Trainer Tests
│   ├── training/                            # Model Training Source
│   │   ├── __init__.py
│   │   ├── data.py                          # Dataset Loading & Processing
│   │   ├── logger.py                        # Training Logger Config
│   │   ├── model.py                         # Training Model Configuration
│   │   ├── run_train.py                     # Training Execution Script
│   │   ├── train.py                         # Main Training Logic
│   │   └── trainer.py                       # Trainer Setup
│   ├── Dockerfile                           # Backend Docker Configuration
│   ├── pyproject.toml                       # Python Project Configuration
│   ├── requirements.txt                     # Python Dependencies
│   └── run.py                               # Backend-specific Runner
│
├── frontend/                            # Frontend Service (React + Vite)
│   ├── public/                              # Public Static Assets
│   │   └── vite.svg
│   ├── src/                                 # Frontend Source Code
│   │   ├── assets/                          # Assets
│   │   │   ├── css/
│   │   │   │   └── index.css                # Global Styles
│   │   │   ├── images/
│   │   │   └── react.svg
│   │   ├── components/                      # React Components
│   │   │   ├── layout/                      # Layout Components
│   │   │   │   ├── Footer.tsx
│   │   │   │   ├── Header.tsx
│   │   │   │   └── MainLayout.tsx
│   │   │   └── ui/                          # UI Components
│   │   │       └── Features.tsx
│   │   ├── features/                        # Feature Modules
│   │   │   └── translator/
│   │   │       └── TranslatorCard.tsx       # Main Translation Widget
│   │   ├── hooks/                           # Custom React Hooks
│   │   │   └── useParticles.tsx             # Background Animation Hook
│   │   ├── services/                        # API Services
│   │   │   └── api.ts                       # Backend API Client
│   │   ├── types/                           # TypeScript Types
│   │   ├── utils/                           # Frontend Utilities
│   │   ├── App.css                          # App-specific Styles
│   │   ├── App.tsx                          # Root Component
│   │   ├── main.tsx                         # Frontend Entry Point
│   │   └── vite-env.d.ts                    # Vite Type Definitions
│   ├── .gitignore
│   ├── Dockerfile                           # Frontend Docker 
│   ├── eslint.config.js
│   ├── index.html
│   ├── package-lock.json
│   ├── package.json
│   ├── postcss.config.js 
│   ├── tailwind.config.js                   # Tailwind CSS Configuration
│   ├── tsconfig.app.json
│   ├── tsconfig.json
│   ├── tsconfig.node.json
│   └── vite.config.ts
│
├── .gitignore                           # Git Ignore Rules
├── app.png                              # Application Screenshot
├── docker-compose.yml                   # Docker Compose Configuration
├── LICENSE                              # Project License
├── README.md                            # Project Documentation
├── render.yml                           # Render Deployment Configuration
└── run.py                               # Unified Application Launcher
```

---

## Development

If you prefer to run services individually for debugging:

### Backend (FastAPI)
```bash
cd backend
# Ensure venv is active
python -m uvicorn app.main:app --reload
```

### Frontend (React + Vite)
```bash
cd frontend
npm run dev
```

---

## API Documentation

### Endpoints

| Method | Endpoint     | Description            |
| ------ | ------------ | ---------------------- |
| GET    | `/`          | Web UI                 |
| POST   | `/translate` | Translate text         |
| GET    | `/health`    | Health check           |
| GET    | `/docs`      | Swagger UI             |

### Interactive Docs
Once running, access the automatic API docs:
- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## Model Training

To fine-tune the translation model:

### Train the Model

```bash
# Standard training
python -m backend.training.train

# Custom parameters
python -m backend.training.train \
    --model-checkpoint "t5-small" \
    --output-dir "./fine-tuned-model" \
    --num-epochs 3 \
    --batch-size 16
```

### Training Configuration

| Parameter        | Default                                |
| ---------------- | -------------------------------------- |
| Base Model       | `t5-small`                             |
| Dataset          | `opus_books` (en-es)                   |
| Task Prefix      | `translate English to Spanish: `       |
| Learning Rate    | `1e-3`                                 |
| LoRA Rank        | 8                                      |
| LoRA Alpha       | 32                                     |
| Target Modules   | `["q", "v"]` (T5 attention projections)|
| Trainable Params | ~294K of ~60.8M (~0.49%)               |

---

## Evaluation & Results

Fine-tuning quality is tracked with complementary metrics rather than a single number:

| Metric        | What it captures                                             |
| ------------- | ----------------------------------------------------------- |
| **BLEU**      | Word n-gram overlap with the reference (used to pick the best checkpoint) |
| **chrF**      | Character n-gram overlap — robust to Spanish morphology/inflection |
| **METEOR**    | Overlap with credit for synonyms and word order             |
| **BERTScore** | Semantic (meaning-based) similarity, beyond surface overlap |

During training, BLEU / chrF / METEOR are computed on the held-out split every epoch, and the checkpoint with the **best BLEU** is kept (`metric_for_best_model="bleu"`). Training ran for 3 epochs (final training loss ≈ **1.23**).

### Did fine-tuning improve the model?

**Yes — clearly for the task, modestly for the metrics.**

- **Task acquisition (qualitative):** Before fine-tuning, `t5-small` responds to the English→Spanish prompt in **German** (e.g. *"The book is on the table."* → *"Das Buch ist auf dem Tisch."*). After LoRA fine-tuning on `opus_books` (en-es), the same input produces **Spanish** (*"El bucho está en la mesa."*). So fine-tuning successfully taught the model the target language/task — the single most important outcome.
- **Semantic score (BERTScore F1, higher = closer in meaning):**

  | Model                 | BERTScore F1 |
  | --------------------- | ------------ |
  | Before (base t5-small)| 0.8105       |
  | After (LoRA fine-tuned)| **0.8249**  |

  A positive gain of **+0.014**, confirming the outputs moved closer to the reference meaning.

**Honest caveats.** The absolute quality is still limited — `t5-small` is tiny and LoRA trains only ~0.49% of its parameters, so BLEU stays low and the Spanish is not always fluent (e.g. *"El bucho"* instead of *"El libro"*). The BERTScore comparison above was measured on a small illustrative sample with example references, so treat it as directional evidence, not a full benchmark. The takeaway is that fine-tuning delivers a **clear, positive** improvement in the intended direction; a larger base (e.g. `google/mt5-small`) would raise the ceiling if higher fluency is needed.

---

## Testing

Run the full backend test suite:

```bash
cd backend
pytest tests/ -v --cov=app --cov=training --cov-report=term-missing
```

**Current Coverage:** ~96% (66 tests passed)

---

## Docker Deployment

Run the complete stack with Docker Compose:

```bash
# Build and start
docker-compose up --build

# Run in background
docker-compose up -d
```

---

## Logs

Logs are stored in `logs/` directory:
- `app.log` - Application logs
- `training.log` - Training logs

---

## Author

**Md Emon Hasan** 
**Email:** emon.mlengineer@gmail.com | [GitHub](https://github.com/Md-Emon-Hasan) | [Portfolio](https://emonlabs-ai.hitechparks.com) | [LinkedIn](https://www.linkedin.com/in/md-emon-hasan-695483237/) | [WhatsApp](https://wa.me/8801834363533)

---

## License

MIT License - see [LICENSE](LICENSE)