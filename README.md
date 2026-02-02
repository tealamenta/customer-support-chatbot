# Customer Support Chatbot

Fine-tuned LLM for customer support automation using LoRA.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Model](https://img.shields.io/badge/Model-TinyLlama--1.1B-green.svg)
![Loss](https://img.shields.io/badge/Loss-0.538-brightgreen.svg)
![Tests](https://img.shields.io/badge/Tests-44%20passed-green.svg)
![Coverage](https://img.shields.io/badge/Coverage-94%25-green.svg)

---

## Table of Contents

1. [Results](#results)
2. [Sample Outputs](#sample-outputs)
3. [Architecture](#architecture)
4. [Installation](#installation)
5. [Usage](#usage)
6. [API](#api)
7. [Training](#training)
8. [MLOps](#mlops)
9. [Project Structure](#project-structure)
10. [Testing](#testing)

---

## Results

| Metric | Value |
|--------|-------|
| Base Model | TinyLlama-1.1B-Chat |
| Dataset | Bitext Customer Support (26,872 samples) |
| Technique | LoRA (r=16, alpha=32) |
| Training Loss | **0.538** |
| Training Time | 46 min (Colab T4) |
| Tests | 44 passed |
| Coverage | 94% |

### Loss Progression

| Step | Training Loss | Validation Loss |
|------|---------------|-----------------|
| 500 | 0.636 | 0.624 |
| 1000 | 0.571 | 0.572 |
| 1512 | **0.538** | - |

---

## Sample Outputs

**Customer:** I want to cancel my order

**Assistant:** I've come to understand that you need assistance with canceling your order, and I'm here to help! To proceed with the cancellation process, please follow these steps:
1. Access Your Account: Log in to our platform using your credentials.
2. Navigate to Order History: Once logged in, locate the section labeled "Order History"...

---

**Customer:** Where is my package?

**Assistant:** We've received your message regarding the location of your package. To assist you further, could you please provide me with the order number or any other relevant details associated with your purchase?

---

**Customer:** How do I get a refund?

**Assistant:** I've got that you would like to know how to obtain a refund, and I'm here to assist you with that. To initiate the process, please follow these steps...

---

## Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                         API Layer                           │
│                    FastAPI + Monitoring                     │
├─────────────────────────────────────────────────────────────┤
│   /chat          /health          /metrics                  │
└──────────┬──────────────────────────────┬───────────────────┘
           │                              │
           ▼                              ▼
┌─────────────────────┐      ┌─────────────────────────┐
│   Inference Engine  │      │   Metrics Tracker       │
│   - TinyLlama-1.1B  │      │   - Latency             │
│   - LoRA Adapter    │      │   - Requests            │
│   - Generation      │      │   - Error Rate          │
└─────────────────────┘      └─────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│                      Logging System                         │
│              Console + File (logs/*.log)                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites

- Python 3.11+
- 8GB RAM minimum

### Setup
```bash
git clone https://github.com/yourusername/customer-support-chatbot.git
cd customer-support-chatbot

# Install dependencies
pip install -e .

# Or with requirements
pip install -r requirements.txt
```

### Using Makefile
```bash
make install   # Install package
make test      # Run tests
make lint      # Run linter
```

---

## Usage

### Interactive Demo
```bash
python run.py demo
```
```
============================================================
CUSTOMER SUPPORT CHATBOT - DEMO
============================================================
Type 'quit' to exit

You: I want to cancel my order
Assistant: I've come to understand that you need assistance...

You: quit
Bye!
```

### Evaluation
```bash
python run.py eval
```

### API Server
```bash
python run.py api
```

Server runs at http://localhost:8000

---

## API

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Status check |
| GET | `/health` | Health + metrics summary |
| GET | `/metrics` | Full metrics JSON |
| POST | `/chat` | Generate response |

### POST /chat

**Request:**
```json
{
  "question": "I want to cancel my order"
}
```

**Response:**
```json
{
  "question": "I want to cancel my order",
  "response": "I've come to understand that you need assistance...",
  "latency_ms": 245.5
}
```

### GET /health
```json
{
  "status": "healthy",
  "model_loaded": true,
  "total_requests": 150,
  "avg_latency_ms": 230.5,
  "error_rate": 0.0
}
```

### Docker
```bash
# Build
docker build -t customer-support-chatbot .

# Run
docker-compose up
```

---

## Training

### Dataset

**Bitext Customer Support LLM Chatbot Training Dataset**

| Stat | Value |
|------|-------|
| Total samples | 26,872 |
| Train split | 24,184 (90%) |
| Val split | 2,688 (10%) |
| Intent categories | 27 |

### LoRA Configuration

| Parameter | Value |
|-----------|-------|
| r | 16 |
| lora_alpha | 32 |
| lora_dropout | 0.05 |
| target_modules | q_proj, k_proj, v_proj, o_proj |
| Trainable params | 4.2M (0.38%) |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 1 |
| Batch Size | 16 (4 x 4 grad accum) |
| Learning Rate | 2e-4 |
| Scheduler | Cosine |
| Warmup | 3% |
| Precision | FP16 |
| Hardware | Google Colab T4 |

### Run Training (Colab)

1. Upload `notebooks/03_colab_fine_tuning.ipynb` to Google Colab
2. Runtime > Change runtime type > T4 GPU
3. Run all cells (~45 min)

---

## MLOps

### Monitoring

| Feature | Description |
|---------|-------------|
| **Logging** | File + console logs in `logs/` |
| **Metrics** | Latency, requests, errors tracked |
| **Health Check** | `/health` endpoint with stats |
| **Metrics API** | `/metrics` for dashboards |

### CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`):
- Runs tests on push/PR
- Coverage report
- Linting with Ruff

### Model Card

See `models/MODEL_CARD.md` for:
- Model details
- Training data
- Intended use
- Limitations

---

## Project Structure
```
customer-support-chatbot/
├── .github/workflows/
│   └── ci.yml                    # CI/CD pipeline
├── data/
│   └── processed/                # Training data
├── logs/                         # Application logs
├── metrics/                      # Metrics JSON files
├── models/
│   ├── customer-support-model/   # Fine-tuned adapter
│   └── MODEL_CARD.md            # Model documentation
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_fine_tuning.ipynb      # Local (Mac)
│   └── 03_colab_fine_tuning.ipynb # Colab (recommended)
├── src/
│   ├── api/
│   │   └── app.py               # FastAPI endpoints
│   ├── config/
│   │   ├── settings.py          # Configuration
│   │   └── logging_config.py    # Logging setup
│   ├── evaluation/
│   │   ├── metrics.py           # Evaluation metrics
│   │   └── tracking.py          # Metrics tracking
│   └── model/
│       └── inference.py         # Model inference
├── tests/                       # 44 tests, 94% coverage
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── requirements.txt
├── run.py                       # CLI entry point
└── README.md
```

---

## Testing

### Run Tests
```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=term-missing
```

### Coverage Report

| Module | Coverage |
|--------|----------|
| config/settings.py | 100% |
| config/logging_config.py | 100% |
| evaluation/metrics.py | 100% |
| evaluation/tracking.py | 96% |
| model/inference.py | 91% |
| api/app.py | 82% |
| **TOTAL** | **94%** |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Base Model | TinyLlama-1.1B-Chat |
| Fine-tuning | LoRA (PEFT) |
| Framework | HuggingFace Transformers |
| Training | TRL SFTTrainer |
| API | FastAPI |
| Monitoring | Custom metrics + logging |
| CI/CD | GitHub Actions |
| Container | Docker |

---

## References

- [TinyLlama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- [Bitext Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
- [PEFT LoRA](https://huggingface.co/docs/peft)
- [TRL](https://huggingface.co/docs/trl)

---

## License

MIT License

---

## Author

tealamenta. - AI/ML Engineer
