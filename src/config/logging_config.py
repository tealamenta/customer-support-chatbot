import logging
import sys
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

def setup_logging(name: str = "chatbot") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console_format = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    console.setFormatter(console_format)
    
    # File handler
    log_file = LOG_DIR / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('[%(asctime)s] %(levelname)s [%(name)s] - %(message)s')
    file_handler.setFormatter(file_format)
    
    logger.addHandler(console)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logging()
