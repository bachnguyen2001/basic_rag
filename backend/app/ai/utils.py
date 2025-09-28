from datasets import load_dataset
from app.core.config import settings

def load_dataset_from_hf():
    ds = load_dataset(settings.DATASET_NAME, split="train")
    return ds