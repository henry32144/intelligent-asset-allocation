import torch
import numpy as np
from transformers import AutoTokenizer
RANDOM_SEED = 2020
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# Data configuration
TOP_K = 3
TRAIN_START_DATE = "2012-01-01"
TRAIN_END_DATE = "2015-12-31"
VALID_START_DATE = "2016-01-01"
VALID_END_DATE = "2016-12-31"
TEST_START_DATE = "2017-01-01"
TEST_END_DATE = "2020-07-01"

# Model configuration
BATCH_SIZE = 16
MAX_LEN = 32
EPOCHS = 100
PRE_TRAINED_MODEL_NAME = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./weights/distilbert.bin"

# Optimization
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.224