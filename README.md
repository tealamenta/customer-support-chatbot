# Customer Support Chatbot

Fine-tuned LLM for customer support automation using LoRA with full MLOps pipeline.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Model](https://img.shields.io/badge/Model-TinyLlama--1.1B-green.svg)
![Loss](https://img.shields.io/badge/Loss-0.538-brightgreen.svg)
![Tests](https://img.shields.io/badge/Tests-44%20passed-green.svg)
![Coverage](https://img.shields.io/badge/Coverage-94%25-green.svg)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange.svg)
![Airflow](https://img.shields.io/badge/Airflow-Orchestration-red.svg)
![Kubernetes](https://img.shields.io/badge/Kubernetes-Deployment-blue.svg)

---

## Table of Contents

1. [Results](#results)
2. [Sample Outputs](#sample-outputs)
3. [Architecture](#architecture)
4. [MLOps Stack](#mlops-stack)
5. [Installation](#installation)
6. [Usage](#usage)
7. [API](#api)
8. [Training](#training)
9. [MLOps Commands](#mlops-commands)
10. [Project Structure](#project-structure)
11. [Testing](#testing)

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
└──────────┬──────────────────────────────────┬───────────────┘
           │                              │
           ▼                              ▼
┌─────────────────────┐      ┌─────────────────────────┐
│   Inference Engine  │      │   Metrics Tracker       │
│   - TinyLlama-1.1B  │      │   - Latency             │
│   - LoRA Adapter    │      │   - Requests            │
│   - Generation      │      │   - Error Rate          │
└─────────────────────┘      └─────────────────────────┘
```

---

## MLOps Stack

| Component | Technology | Port |
|-----------|------------|------|
| Experiment Tracking | MLflow | 5001 |
| Data Versioning | DVC | - |
| Orchestration | Airflow | 8081 |
| Containerization | Docker | 8000 |
| Deployment | Kubernetes | - |
| CI/CD | GitHub Actions | - |
| Monitoring | Prometheus + Grafana | 9090/3000 |
```
┌─────────────────────────────────────────────────────────────┐
│                    MLOps Pipeline                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Data          Training         Deployment     Monitoring  │
│   ┌───┐         ┌───┐           ┌───┐          ┌───┐       │
│   │DVC│ ──────► │MLflow│ ──────►│K8s│ ────────►│Prometheus│ │
│   └───┘         └───┘           └───┘          └───┘       │
│                   │                              │          │
│                   ▼                              ▼          │
│               ┌───────┐                     ┌───────┐       │
│               │Airflow│                     │Grafana│       │
│               └───────┘                     └───────┘       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites

- Python 3.11+
- Docker
- 8GB RAM minimum

### Setup
```bash
git clone https://github.com/tealamenta/customer-support-chatbot.git
cd customer-support-chatbot

# Install dependencies
pip install -e ".[dev]"

# Pull data (DVC)
dvc pull
```

### Using Makefile
```bash
make install   # Install package
make test      # Run tests
make lint      # Run linter
make run-api   # Start API
make mlflow    # Start MLflow UI
```

---

## Usage

### Interactive Demo
```bash
python run.py demo
```

### Evaluation
```bash
python run.py eval
```

### API Server
```bash
python run.py api
# http://localhost:8000
```

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

### Run Training (Colab)

1. Upload `notebooks/03_colab_fine_tuning.ipynb` to Google Colab
2. Runtime > Change runtime type > T4 GPU
3. Run all cells (~45 min)

---

## MLOps Commands

### MLflow (Experiment Tracking)
```bash
docker run -d -p 5001:5000 --name mlflow ghcr.io/mlflow/mlflow:v2.11.0 mlflow server --host 0.0.0.0
# http://localhost:5001
```

### DVC (Data Versioning)
```bash
dvc pull                    # Pull data
dvc repro                   # Reproduce pipeline
dvc push                    # Push to remote
```

### Airflow (Orchestration)
```bash
cd docker
docker-compose -f docker-compose.airflow.yml up -d
# http://localhost:8081 (admin/admin)
```

### Docker
```bash
# Standard build
docker build -t customer-support-chatbot .
docker run -p 8000:8000 customer-support-chatbot

# K8s build (with pre-loaded model)
docker build -f Dockerfile.k8s -t customer-support-chatbot:k8s .
```

### Kubernetes
```bash
minikube start
minikube image load customer-support-chatbot:k8s
kubectl apply -f k8s/deployment.yaml
minikube service chatbot-service --url
```

### Monitoring (Prometheus + Grafana)
```bash
cd monitoring
docker-compose up -d
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin)
```

---

## Project Structure
```
customer-support-chatbot/
├── .github/workflows/
│   └── ci-cd.yml                 # CI/CD pipeline
├── dags/
│   └── ml_pipeline.py            # Airflow DAG
├── data/
│   ├── processed/                # Training data (DVC tracked)
│   ├── models/                   # Model files (DVC tracked)
│   ├── processed.dvc
│   └── models.dvc
├── docker/
│   └── docker-compose.airflow.yml
├── k8s/
│   └── deployment.yaml           # Kubernetes manifests
├── logs/                         # Application logs
├── mlruns/                       # MLflow experiments
├── models/
│   ├── customer-support-model/   # Fine-tuned adapter
│   └── MODEL_CARD.md
├── monitoring/
│   ├── docker-compose.yml        # Prometheus + Grafana
│   └── prometheus.yml
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_fine_tuning.ipynb
│   └── 03_colab_fine_tuning.ipynb
├── src/
│   ├── api/
│   │   └── app.py                # FastAPI endpoints
│   ├── config/
│   │   ├── settings.py
│   │   └── logging_config.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── tracking.py
│   ├── model/
│   │   └── inference.py
│   └── training/
│       └── train.py              # Training with MLflow
├── tests/                        # 44 tests, 94% coverage
├── Dockerfile
├── Dockerfile.k8s                # K8s image with pre-loaded model
├── Makefile
├── pyproject.toml
├── dvc.yaml
├── run.py
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
| MLOps | MLflow, DVC, Airflow |
| Containers | Docker, Kubernetes |
| Monitoring | Prometheus, Grafana |
| CI/CD | GitHub Actions |

---

## References

- [TinyLlama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- [Bitext Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
- [PEFT LoRA](https://huggingface.co/docs/peft)
- [TRL](https://huggingface.co/docs/trl)
- [MLflow](https://mlflow.org/)
- [DVC](https://dvc.org/)
- [Airflow](https://airflow.apache.org/)

---

## License

MIT License

---

## Author

tealamenta - AI/ML Engineer
- GitHub: [@tealamenta](https://github.com/tealamenta)
