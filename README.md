# Translatica: English to Spanish Translation

[![CI/CD](https://github.com/Md-Emon-Hasan/Translatica/actions/workflows/main.yml/badge.svg)](https://github.com/Md-Emon-Hasan/Translatica/actions) [![Python](https://img.shields.io/badge/python-3.11-blue)](https://python.org) ![TypeScript](https://img.shields.io/badge/typescript-%23007ACC.svg?style=flat&logo=typescript&logoColor=white) [![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com) ![React](https://img.shields.io/badge/react-%2320232a.svg?style=flat&logo=react&logoColor=%2361DAFB) ![Vite](https://img.shields.io/badge/vite-%23646CFF.svg?style=flat&logo=vite&logoColor=white) ![TailwindCSS](https://img.shields.io/badge/tailwindcss-%2338B2AC.svg?style=flat&logo=tailwind-css&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white) ![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow) ![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Translating a novel from English to Spanish by hand is slow, and machine-translation tools flatten the author's voice or cut long passages short. **Translatica** returns a Spanish translation that keeps the original's tone and style, answers consistently, keeps a record of every translation it makes, and warns when text looks like it's already Spanish instead of guessing silently. That combination is what turns it from a research demo into something a publisher could switch on for readers today.

Under the hood, Translatica is a modular monolith: a **FastAPI** backend serving a **LoRA-adapted (PEFT)** fine-tune of Helsinki-NLP's `opus-mt-en-es`, trained on the `opus_books` literary corpus and scored with BLEU. Requests pass through Pydantic validation, a thread-safe TTL cache keyed on text and decoding parameters, chunking that routes around `transformers`' silent truncation, and batched beam-search generation, then get logged to SQLite — plus a proxy-aware rate limiter, deterministic decoding, a **React 19 + TypeScript + Tailwind** frontend, a 193-test CI suite, and Docker images for both services on Render. That plumbing — caching, rate limits, CI — separates a **production-ready service** from a notebook that only works as a demo.

[![Project demo video](https://github.com/user-attachments/assets/704e7d2d-af01-46ee-998f-7342735db1b1)](https://github.com/user-attachments/assets/704e7d2d-af01-46ee-998f-7342735db1b1)

![Project Screenshot](https://github.com/user-attachments/assets/ac3c656e-319e-4f0a-beaa-8aba201b9ef1)

---

## Fine Tuned LLM

[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-omt--en--es--books-yellow)](https://huggingface.co/emonHasan01/omt-en-es-books)

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
| **Decoding**     | Beam search (configurable, `num_beams=4`)    |
| **Caching**      | cachetools TTLCache (thread-safe, in-memory) |
| **Rate Limiting**| slowapi (proxy-aware `X-Forwarded-For` keying) |
| **Frontend**     | React, Vite, Tailwind CSS                    |
| **Testing**      | Pytest (193 tests, 98% coverage)             |
| **CI/CD**        | GitHub Actions (Lint → Test → Docker Build)  |
| **Deployment**   | Docker, Render                               |

---

## System Architecture

Translatica follows a **modular monolithic architecture** that clearly separates training, inference, API, and frontend layers while maintaining simple deployment and strong production readiness.

### High-Level Architecture

```
┌──────────────────────────┐
│        Frontend UI       │
│ React 19 + TS + Vite (UI)│
└─────────────┬────────────┘
              │ HTTP Requests
              ▼
┌──────────────────────────┐
│       FastAPI Server     │
│  Routing + Rate Limiting │
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│    Validation Layer      │
│ Length + Alpha + Lang.   │
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│    TTL Response Cache    │
│  Hit → return, Miss ↓    │
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│   Translation Service    │
│ Chunking → Batching →    │
│ Chunk Cache → Reassemble │
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│       Model Manager      │
│ LoRA + Tokenizer + Beams │
└─────────────┬────────────┘
              │
              ▼
┌────────────────────────────────────┐
│  LoRA Fine-Tuned Transformer Model │
│ Helsinki-NLP/opus-mt-en-es (PEFT)  │
└────────────────────────────────────┘
```

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
│   │   │   ├── cache.py                     # Thread-Safe TTL Response Cache
│   │   │   ├── config.py                    # Application Settings
│   │   │   ├── database.py                  # Database Connection Logic
│   │   │   ├── model.py                     # ML Model Loading & Management
│   │   │   └── rate_limit.py                # Proxy-Aware Rate Limiting
│   │   ├── models/                          # Data Models
│   │   │   ├── __init__.py
│   │   │   └── translation.py               # Database Schema Models
│   │   ├── services/                        # Business Logic Layer
│   │   │   ├── __init__.py
│   │   │   └── translation.py               # Translation Processing Service
│   │   ├── utils/                           # Utility Functions
│   │   │   ├── __init__.py
│   │   │   ├── chunking.py                  # Sentence-Aware Text Chunking
│   │   │   ├── logger.py                    # Logging Configuration
│   │   │   └── validation.py                # Input Validation Rules
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
│   │   ├── app.log                          # Application Logs
│   │   └── training.log                     # Training Logs (created by a run)
│   ├── notebook/                            # Jupyter Notebooks
│   │   └── Experiment.ipynb                 # Training Experiments
│   ├── tests/                               # Test Suite
│   │   ├── __init__.py
│   │   ├── conftest.py                      # Test Fixtures
│   │   ├── test_api.py                      # API Endpoint Tests
│   │   ├── test_batch.py                    # Batch Endpoint Tests
│   │   ├── test_cache.py                    # Response Cache Tests
│   │   ├── test_chunking.py                 # Chunking Tests
│   │   ├── test_config.py                   # Config Tests
│   │   ├── test_generation_config.py        # Beam Search Config Tests
│   │   ├── test_history_stats.py            # History & Stats Tests
│   │   ├── test_main.py                     # App Initialization Tests
│   │   ├── test_model.py                    # Model Manager Tests
│   │   ├── test_model_info.py               # Model Info Endpoint Tests
│   │   ├── test_rate_limit.py               # Rate Limiting Tests
│   │   ├── test_services.py                 # Service Layer Tests
│   │   ├── test_validation.py               # Input Validation Tests
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
│   ├── compare_decoding.py                  # Greedy vs Beam Search Comparison
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
│   │   │   └── css/
│   │   │       └── index.css                # Global Styles
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
│   │   ├── App.tsx                          # Root Component
│   │   └── main.tsx                         # Frontend Entry Point
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
| LoRA Rank        | 16                            |
| LoRA Alpha       | 32                           |
| Target Modules   | `["q_proj", "v_proj"]`       |
| Trainable Params | ~0.38%                       |

---

## Results

Scored with **sacreBLEU** on 1,000 held-out `opus_books` en-es pairs, beam size 4.

| | sacreBLEU | vs base |
| --- | --- | --- |
| `Helsinki-NLP/opus-mt-en-es` (untouched) | 17.09 | — |
| **+ LoRA adapter** | **19.99** | **+2.90** |

Per epoch, against the base model's 17.09:

| Epoch | sacreBLEU |
| --- | --- |
| 1 | 19.23 |
| 2 | 19.89 |
| 3 | 19.99 |

The baseline row is the point. Without it, "19.99 BLEU" says nothing about whether the
fine-tune helped — `opus-mt-en-es` is already a strong translator, and any adapter has to
beat it rather than merely score well.

The gain is real but diffuse: on most sentences the base and adapted models produce
identical output, and the difference shows up as small shifts in word choice. Full
breakdown on the [model card](https://huggingface.co/emonHasan01/omt-en-es-books).

---

## API Documentation

### Endpoints

| Method | Endpoint            | Description                                                        | Rate Limit  | Cache        |
| ------ | ------------------- | ------------------------------------------------------------------ | ----------- | ------------ |
| GET    | `/`                 | Not served by the API (404) — the React UI is a separate service    | –           | –            |
| POST   | `/translate`        | Translate text                                                       | `20/minute` | 1 h TTL      |
| POST   | `/translate/batch`  | Translate a list of texts in one batched pass, order preserved       | `5/minute`  | 1 h TTL      |
| GET    | `/history`          | Paginated translation log (search + date filters)                    | `60/minute` | Uncached     |
| GET    | `/stats`            | Aggregate statistics + live cache statistics                         | `60/minute` | Uncached     |
| GET    | `/model-info`       | Base model, LoRA metadata, generation config, limits                 | Unlimited   | Uncached     |
| GET    | `/health`           | Health check                                                         | Unlimited   | Uncached     |
| GET    | `/docs`             | Swagger UI                                                           | Unlimited   | –            |

Cache TTL is the configured `CACHE_TTL_SECONDS` (default `3600`, i.e. 1 hour).

### Interactive Docs
Once running, access the automatic API docs:
- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## Inference Pipeline & Performance

Every translation request flows through **validation → cache → chunking → batched generation → reassembly**. This section explains why each layer exists.

### Sentence-aware chunking

`transformers` **silently truncates** input longer than the model's limit instead of raising, so long literary paragraphs were being cut off mid-translation with no error at all. The response carries an optional `chunk_count` so the split is observable.

### Beam search decoding

Generation parameters all come from configuration — nothing is hardcoded — and `do_sample` is always `False`, because caching is only correct if identical input reliably produces identical output. Setting `GEN_NUM_BEAMS=1` restores the plain greedy path, so the change is fully reversible.

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

## Testing

Run the full backend test suite:

```bash
cd backend
pytest tests/ -v --cov=app --cov=training --cov-report=term-missing
```

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

## Author

**Md Emon Hasan** 
**Email:** emon.mlengineer@gmail.com | [Portfolio](https://emonlabs-ai.hitechparks.com/) | [GitHub](https://github.com/Md-Emon-Hasan) | [LinkedIn](https://www.linkedin.com/in/md-emon-hasan-695483237/) | [WhatsApp](https://wa.me/8801834363533)

---

## License

MIT License - see [LICENSE](LICENSE)
