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
| **Model**        | Helsinki-NLP/opus-mt-en-es (LoRA fine-tuned) |
| **Frontend**     | React, Vite, Tailwind CSS                    |
| **Testing**      | Pytest (63 tests, 92% coverage)              |
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
│ Helsinki-NLP/opus-mt-en-es (PEFT)  │
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
│   │   └── fine-tuned-tokenizer/            # Tokenizer Assets
│   │       ├── source.spm
│   │       ├── special_tokens_map.json
│   │       ├── target.spm
│   │       ├── tokenizer_config.json
│   │       └── vocab.json
│   ├── logs/                                # Application Logs
│   │   └── app.log
│   ├── notebook/                            # Jupyter Notebooks
│   │   └── Experiment.ipynb                 # Training Experiments
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
    --model-checkpoint "Helsinki-NLP/opus-mt-en-es" \
    --output-dir "./fine-tuned-model" \
    --num-epochs 3 \
    --batch-size 16
```

### Training Configuration

| Parameter        | Default                      |
| ---------------- | ---------------------------- |
| Base Model       | `Helsinki-NLP/opus-mt-en-es` |
| Dataset          | `opus_books` (en-es)         |
| LoRA Rank        | 8                            |
| LoRA Alpha       | 32                           |
| Target Modules   | `["q_proj", "v_proj"]`       |
| Trainable Params | ~0.38%                       |

---

## Testing

Run the full backend test suite:

```bash
cd backend
pytest tests/ -v --cov=app --cov=training --cov-report=term-missing
```

**Current Coverage:** ~92% (63 tests passed)

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
**Email:** emon.mlengineer@gmail.com | [GitHub](https://github.com/Md-Emon-Hasan) | [LinkedIn](https://www.linkedin.com/in/md-emon-hasan-695483237/) | [WhatsApp](https://wa.me/8801834363533)

---

## License

MIT License - see [LICENSE](LICENSE)