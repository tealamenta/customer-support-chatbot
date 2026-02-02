import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.config.settings import CONFIG
from src.config.logging_config import logger
from src.evaluation.tracking import tracker


class CustomerSupportBot:
    def __init__(self, adapter_path: str = None):
        self.adapter_path = adapter_path or str(CONFIG.adapter_path)
        self.model = None
        self.tokenizer = None
        self.device = None
    
    def load(self):
        start_time = time.time()
        logger.info(f"Loading base model: {CONFIG.base_model}")
        
        base_model = AutoModelForCausalLM.from_pretrained(
            CONFIG.base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        logger.info(f"Loading adapter: {self.adapter_path}")
        self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.adapter_path)
        self.device = self.model.device
        
        load_time = time.time() - start_time
        tracker.log_model_load(self.adapter_path, load_time)
        logger.info(f"Model loaded in {load_time:.2f}s")
        
        return self
    
    def chat(self, question: str) -> str:
        start_time = time.time()
        
        try:
            prompt = f"""<|system|>
You are a helpful customer support assistant.</s>
<|user|>
{question}</s>
<|assistant|>
"""
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=CONFIG.max_new_tokens,
                    temperature=CONFIG.temperature,
                    top_p=CONFIG.top_p,
                    do_sample=True,
                    repetition_penalty=CONFIG.repetition_penalty,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "<|assistant|>" in response:
                response = response.split("<|assistant|>")[-1].strip()
            response = response.split("<")[0].strip()
            
            latency = time.time() - start_time
            tracker.log_inference(question, response, latency)
            logger.debug(f"Inference completed in {latency*1000:.0f}ms")
            
            return response
            
        except Exception as e:
            tracker.log_error()
            logger.error(f"Inference error: {e}")
            raise


def load_model(adapter_path: str = None) -> CustomerSupportBot:
    bot = CustomerSupportBot(adapter_path)
    return bot.load()
