# Customer Support Chatbot - MLOps Pipeline

Fine-tuned LLM for customer support with end-to-end MLOps pipeline.

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

1. [Overview](#overview)
2. [MLOps Architecture](#mlops-architecture)
3. [Results](#results)
4. [Quick Start](#quick-start)
5. [Step-by-Step Guide](#step-by-step-guide)
6. [API Reference](#api-reference)
7. [Project Structure](#project-structure)
8. [Testing](#testing)

---

## Overview

Complete MLOps pipeline for fine-tuning and deploying an LLM-based customer support chatbot.

### Key Features

- Fine-tuned TinyLlama-1.1B with LoRA
- Full MLOps stack: MLflow, DVC, Airflow, Kubernetes
- Production-ready API with FastAPI
- Monitoring with Prometheus + Grafana
- 94% test coverage

---

## MLOps Architecture
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MLOps Pipeline                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌───────┐ │
│  │   Data   │    │ Training │    │  Model   │    │  Deploy  │    │Monitor│ │
│  │   DVC    │───►│  MLflow  │───►│ Evaluate │───►│   K8s    │───►│Grafana│ │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘    └───────┘ │
│                                                                             │
│                        ┌──────────────────┐                                │
│                        │     Airflow      │                                │
│                        │  Orchestration   │                                │
│                        └──────────────────┘                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Component | Technology | Port |
|-----------|------------|------|
| Data Versioning | DVC | - |
| Experiment Tracking | MLflow | 5001 |
| Orchestration | Airflow | 8081 |
| Containerization | Docker | 8000 |
| Deployment | Kubernetes | - |
| CI/CD | GitHub Actions | - |
| Monitoring | Prometheus + Grafana | 9090/3000 |

---

## Results

| Metric | Value |
|--------|-------|
| Base Model | TinyLlama-1.1B-Chat |
| Dataset | 26,872 samples |
| Final Loss | **0.538** |
| Training Time | 46 min (T4 GPU) |
| Trainable Params | 4.2M (0.38%) |

---

## Quick Start
```bash
# Clone and install
git clone https://github.com/tealamenta/customer-support-chatbot.git
cd customer-support-chatbot
pip install -e ".[dev]"

# Pull data
dvc pull

# Run demo
python run.py demo

# Run API
python run.py api
```

---

## Step-by-Step Guide

### Step 1: Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Verify installation
python -c "import torch; print(torch.__version__)"
```

---

### Step 2: Data Preparation

#### 2.1 Download Dataset
```bash
# Option A: Pull with DVC (if already tracked)
dvc pull

# Option B: Download from HuggingFace
python -c "
from datasets import load_dataset
dataset = load_dataset('bitext/Bitext-customer-support-llm-chatbot-training-dataset')
print(f'Total samples: {len(dataset[\"train\"])}')
"
```

#### 2.2 Explore Data
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

#### 2.3 Preprocess Data
```python
# Format for TinyLlama chat template
def format_chat(example):
    text = f"""<|system|>
You are a helpful customer support assistant.</s>
<|user|>
{example['instruction']}</s>
<|assistant|>
{example['response']}</s>"""
    return {"text": text}

# Apply and split
dataset = dataset["train"].map(format_chat)
dataset = dataset.train_test_split(test_size=0.1, seed=42)
```

#### 2.4 Version Data with DVC
```bash
# Track data
dvc add data/processed

# Commit
git add data/processed.dvc data/.gitignore
git commit -m "Track training data with DVC"

# Push to remote (optional)
dvc push
```

---

### Step 3: Training

#### 3.1 Start MLflow (for experiment tracking)
```bash
# Terminal 1: Start MLflow server
docker run -d -p 5001:5000 --name mlflow ghcr.io/mlflow/mlflow:v2.11.0 mlflow server --host 0.0.0.0

# Verify: open http://localhost:5001
```

#### 3.2 Option A: Train on Google Colab (Recommended)
```bash
# 1. Upload notebook to Colab
notebooks/03_colab_fine_tuning.ipynb

# 2. Select GPU runtime
Runtime > Change runtime type > T4 GPU

# 3. Run all cells (~46 minutes)

# 4. Download model
# Files will be saved to models/customer-support-model/
```

#### 3.3 Option B: Train Locally with MLflow
```bash
# Terminal 2: Run training
python -m src.training.train

# This will:
# - Load TinyLlama-1.1B
# - Apply LoRA configuration
# - Train for 1 epoch
# - Log metrics to MLflow
# - Save model to models/customer-support-model/
```

#### 3.4 Training Configuration

Edit `src/training/train.py` to modify:
```python
config = {
    "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "lora_r": 16,
    "lora_alpha": 32,
    "epochs": 1,
    "batch_size": 4,
    "learning_rate": 2e-4,
}
```

#### 3.5 Monitor Training
```bash
# View experiments in MLflow
open http://localhost:5001

# Check logged metrics:
# - loss (per step)
# - final_loss
# - training_time_minutes
```

---

### Step 4: Evaluation

#### 4.1 Run Evaluation
```bash
python run.py eval
```

Output:
```
============================================================
EVALUATION
============================================================
Evaluating on 50 samples...

RESULTS
============================================================
Coherence Rate: 92.0%
Avg Length Score: 0.85
Avg Keyword Score: 0.78

By Intent:
  cancel_order: 95% (10 samples)
  refund_request: 90% (8 samples)
  track_package: 94% (12 samples)
```

#### 4.2 Interactive Test
```bash
python run.py demo
```
```
You: I want to cancel my order
Assistant: I've come to understand that you need assistance...

You: quit
```

---

### Step 5: API Deployment

#### 5.1 Run Locally
```bash
python run.py api

# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

#### 5.2 Test API
```bash
# Health check
curl http://localhost:8000/health

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I cancel my order?"}'
```

---

### Step 6: Docker Deployment

#### 6.1 Build Image
```bash
# Standard image
docker build -t customer-support-chatbot .

# K8s image (with pre-loaded model)
docker build -f Dockerfile.k8s -t customer-support-chatbot:k8s .
```

#### 6.2 Run Container
```bash
docker run -d -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --name chatbot \
  customer-support-chatbot

# Test
curl http://localhost:8000/health
```

---

### Step 7: Kubernetes Deployment

#### 7.1 Start Minikube
```bash
minikube start
```

#### 7.2 Load Image
```bash
minikube image load customer-support-chatbot:k8s
```

#### 7.3 Deploy
```bash
kubectl apply -f k8s/deployment.yaml
```

#### 7.4 Get Service URL
```bash
# Terminal 1: Keep this running
minikube service chatbot-service --url

# Terminal 2: Test with the URL provided
curl http://127.0.0.1:<PORT>/health
```

#### 7.5 Check Status
```bash
# View pods
kubectl get pods

# View logs
kubectl logs -l app=chatbot --tail=20

# Scale
kubectl scale deployment customer-support-chatbot --replicas=3
```

---

### Step 8: Airflow Pipeline

#### 8.1 Start Airflow
```bash
cd docker
docker-compose -f docker-compose.airflow.yml up -d
```

#### 8.2 Access UI
```
URL: http://localhost:8081
Username: admin
Password: admin
```

#### 8.3 DAG Pipeline

The `customer_support_ml_pipeline` DAG runs:
```
start → validate_data → train_model → evaluate_model → deploy_model → end
```

---

### Step 9: Monitoring

#### 9.1 Start Prometheus + Grafana
```bash
cd monitoring
docker-compose up -d
```

#### 9.2 Access Dashboards
```
Prometheus: http://localhost:9090
Grafana: http://localhost:3000 (admin/admin)
```

#### 9.3 Available Metrics

| Metric | Endpoint |
|--------|----------|
| Health | GET /health |
| Metrics | GET /metrics |

---

### Step 10: CI/CD

Push to GitHub triggers:

1. Run tests (`pytest`)
2. Check coverage (>80%)
3. Build Docker image
4. Deploy (on main branch)
```bash
# Commit and push
git add .
git commit -m "Update model"
git push

# Check pipeline
open https://github.com/tealamenta/customer-support-chatbot/actions
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Status |
| GET | `/health` | Health check |
| GET | `/metrics` | Prometheus metrics |
| POST | `/chat` | Chat inference |

### Example
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I get a refund?"}'
```

Response:
```json
{
  "question": "How do I get a refund?",
  "response": "I understand you would like to request a refund...",
  "latency_ms": 245.5
}
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
│   ├── processed/                # DVC tracked
│   ├── models/                   # DVC tracked
│   ├── processed.dvc
│   └── models.dvc
├── docker/
│   └── docker-compose.airflow.yml
├── k8s/
│   └── deployment.yaml
├── mlruns/                       # MLflow experiments
├── models/
│   └── customer-support-model/
├── monitoring/
│   ├── docker-compose.yml
│   └── prometheus.yml
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_fine_tuning.ipynb
│   └── 03_colab_fine_tuning.ipynb
├── src/
│   ├── api/app.py
│   ├── config/
│   ├── evaluation/
│   ├── model/inference.py
│   └── training/train.py
├── tests/                        # 94% coverage
├── Dockerfile
├── Dockerfile.k8s
├── Makefile
├── pyproject.toml
├── dvc.yaml
└── run.py
```

---

## Testing
```bash
# Run tests
make test

# With coverage
pytest tests/ -v --cov=src --cov-report=term-missing
```

| Module | Coverage |
|--------|----------|
| Total | **94%** |

---

## Commands Reference

| Task | Command |
|------|---------|
| Install | `pip install -e ".[dev]"` |
| Pull data | `dvc pull` |
| Train | `python -m src.training.train` |
| Evaluate | `python run.py eval` |
| Demo | `python run.py demo` |
| API | `python run.py api` |
| Test | `make test` |
| MLflow | `docker run -d -p 5001:5000 ghcr.io/mlflow/mlflow:v2.11.0 mlflow server --host 0.0.0.0` |
| Airflow | `cd docker && docker-compose -f docker-compose.airflow.yml up -d` |
| K8s | `kubectl apply -f k8s/deployment.yaml` |
| Monitoring | `cd monitoring && docker-compose up -d` |

---

## License

MIT

## Author

tealamenta. - AI/ML Engineer
- GitHub: [@tealamenta](https://github.com/tealamenta)
