import torch
import numpy as np
from transformers import AutoTokenizer
RANDOM_SEED = 2020
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# Data configuration
TOP_K = 5

# Model configuration
BATCH_SIZE = 8
MAX_LEN = 16
EPOCHS = 5
PRE_TRAINED_MODEL_NAME = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./weights/distilbert.bin"