# Model Card: Customer Support Chatbot

## Model Details

- **Model Type:** TinyLlama-1.1B + LoRA Adapter
- **Base Model:** TinyLlama/TinyLlama-1.1B-Chat-v1.0
- **Fine-tuning:** LoRA (r=16, alpha=32)
- **Training Loss:** 0.538

## Training Data

- **Dataset:** Bitext Customer Support LLM Chatbot Training Dataset
- **Size:** 26,872 samples
- **Split:** 90% train, 10% validation
- **Languages:** English

## Training Procedure

| Parameter | Value |
|-----------|-------|
| Epochs | 1 |
| Batch Size | 16 |
| Learning Rate | 2e-4 |
| Scheduler | Cosine |
| Hardware | Google Colab T4 GPU |
| Training Time | 46 minutes |

## Intended Use

- Customer service automation
- FAQ answering
- Order management queries
- Refund/return assistance

## Limitations

- English only
- Limited to customer support domain
- May generate plausible but incorrect information
- Not suitable for sensitive/legal advice

## Metrics

| Step | Training Loss | Validation Loss |
|------|---------------|-----------------|
| 500 | 0.636 | 0.624 |
| 1000 | 0.571 | 0.572 |
| 1512 | 0.538 | - |
