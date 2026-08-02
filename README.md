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

`transformers` **silently truncates** input longer than the model's limit instead of raising, so long literary paragraphs were being cut off mid-translation with no error at all. `app/utils/chunking.py` fixes this:

1. The document is split into paragraph blocks on blank lines.
2. Each block is split on sentence boundaries (`.`, `!`, `?`, plus the Spanish opening marks `¿` / `¡`).
3. Sentences are greedily packed into chunks that stay under `CHUNK_TOKEN_BUDGET`, measured in **real tokenizer tokens**, not characters. If the tokenizer is not reachable, a character-based estimate is used and a warning is logged.
4. A single sentence over budget is split further on clause boundaries (`;`, `:`, `,`) and, as a last resort, on a hard token window. **No text is ever dropped.**
5. Chunks are translated and rejoined with single spaces; the original `\n\n` paragraph breaks are restored.

The response carries an optional `chunk_count` so the split is observable.

### Beam search decoding

Generation parameters all come from configuration — nothing is hardcoded — and `do_sample` is always `False`, because caching is only correct if identical input reliably produces identical output. Setting `GEN_NUM_BEAMS=1` restores the plain greedy path, so the change is fully reversible.

Beam search is **noticeably slower on CPU**, which is what the Render deployment uses, so inference duration is logged per request through the existing logger. `backend/compare_decoding.py` prints greedy and beam-search output side by side for a fixed set of sample sentences — a qualitative eye-test only; it computes no metrics and makes no BLEU claims.

### Validation

| Rule | Behaviour |
| ---- | --------- |
| Empty / whitespace-only | `422` with a readable message |
| Longer than `MAX_INPUT_CHARS` | `422` |
| No alphabetic characters | `422` |
| Looks like Spanish already | **`200` with a `warning` field** |

Validation failures use FastAPI's standard `422`; the response body carries a human-readable `detail` string (plus a matching `error` field and a per-field `errors` list), which is exactly what the frontend renders.

The source-language check is a lightweight heuristic — Spanish-only characters plus a small set of common Spanish function words. It **warns instead of blocking** because the heuristic will produce false positives, and rejecting a valid request is worse than attaching a warning to a successful one.

### Caching

The cache is a `cachetools.TTLCache` guarded by a `threading.Lock` on every read and write (`TTLCache` is not thread-safe, and requests are served from FastAPI's threadpool). It is deliberately **in-memory rather than Redis**: the deployment is a single instance, it needs zero extra infrastructure, and `python run.py` stays runnable after a plain `pip install -r backend/requirements.txt`.

The key is a SHA-256 hash of the normalised text **plus the generation parameters**:

```
sha256("<normalised text>|<num_beams>|<max_new_tokens>|<length_penalty>|<no_repeat_ngram_size>")
```

Including the generation parameters means changing `num_beams` cannot serve stale results produced by the old decoding strategy. Caching happens at **both** the whole-request and the individual-chunk level, so a sentence repeated across paragraphs or across requests is free. Empty, `None` and error results are never cached.

### Batching

`POST /translate/batch` does **not** loop over single translations. Every item is validated, checked against the cache, and chunked; all remaining chunks from all items are tokenized together with `padding=True` and sent through **one** `model.generate()` call, then decoded and reassembled per item. Invalid items fail individually — each result carries `success` and either `translation` or `error` — so one bad entry never fails the batch, and the results always come back in the submitted order.

### Rate limiting

Limits are enforced with `slowapi`. Because the API runs behind Nginx and Render's router, `get_remote_address` would see only the proxy and bucket every user together, so the key function reads the first entry of `X-Forwarded-For` and falls back to the peer address. Exceeding a limit returns a clean JSON `429` with a `Retry-After` header and logs the event:

```json
{
  "detail": "Rate limit exceeded: 20 per 1 minute. Please retry in 48s.",
  "error": "Rate limit exceeded: 20 per 1 minute. Please retry in 48s.",
  "limit": "20 per 1 minute",
  "retry_after": 48
}
```

### Configuration

All of the following are `Settings` fields in `app/core/config.py` and can be overridden by environment variables or `.env`.

| Variable | Default | Purpose |
| -------- | ------- | ------- |
| `GEN_NUM_BEAMS` | `4` | Beam count (`1` = greedy fallback) |
| `GEN_MAX_NEW_TOKENS` | `256` | Maximum tokens generated per chunk |
| `GEN_LENGTH_PENALTY` | `1.0` | Beam search length penalty |
| `GEN_EARLY_STOPPING` | `True` | Stop when all beams finish |
| `GEN_NO_REPEAT_NGRAM_SIZE` | `3` | Blocks repetition loops |
| `MAX_INPUT_TOKENS` | `384` | Hard tokenizer truncation ceiling |
| `CHUNK_TOKEN_BUDGET` | `256` | Token budget per chunk |
| `CHARS_PER_TOKEN_ESTIMATE` | `4` | Fallback estimate when no tokenizer |
| `MIN_INPUT_CHARS` | `1` | Minimum accepted input length |
| `MAX_INPUT_CHARS` | `10000` | Maximum accepted input length |
| `LANG_CHECK_ENABLED` | `True` | Enable the Spanish-input heuristic |
| `CACHE_ENABLED` | `True` | Disable for a transparent pass-through |
| `CACHE_TTL_SECONDS` | `3600` | Entry lifetime |
| `CACHE_MAXSIZE` | `500` | Maximum cached entries |
| `RATE_LIMIT_ENABLED` | `True` | Master switch for all limits |
| `RATE_LIMIT_TRANSLATE` | `20/minute` | Limit for `POST /translate` |
| `RATE_LIMIT_BATCH` | `5/minute` | Limit for `POST /translate/batch` |
| `RATE_LIMIT_HISTORY` | `60/minute` | Limit for `GET /history` |
| `RATE_LIMIT_STATS` | `60/minute` | Limit for `GET /stats` |
| `MAX_BATCH_SIZE` | `20` | Maximum items per batch |
| `MAX_BATCH_CHARS` | `50000` | Maximum characters per batch |
| `HISTORY_DEFAULT_LIMIT` | `20` | Default `/history` page size |
| `HISTORY_MAX_LIMIT` | `100` | Maximum `/history` page size |
| `HISTORY_TEXT_PREVIEW_CHARS` | `200` | Truncation length in list responses |

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

**Current Coverage:** ~98% (193 tests passed)

Alongside the original API, config, model, service and training tests, the suite covers:

- `test_chunking.py` — sentence splitting, token-budget packing, over-long sentences, paragraph preservation
- `test_generation_config.py` — beam parameters from config, greedy fallback, config overrides
- `test_validation.py` — 422 rules, readable messages, Spanish input warning instead of rejection
- `test_cache.py` — hits/misses, key normalisation, generation params in the key, TTL expiry, disable switch
- `test_rate_limit.py` — limits, 429 body, `X-Forwarded-For` bucketing, unlimited endpoints
- `test_batch.py` — order preservation, partial success, cache reuse, a single `generate()` call
- `test_history_stats.py` — pagination, filters, truncation, empty-database behaviour
- `test_model_info.py` — every field present and sourced from config

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