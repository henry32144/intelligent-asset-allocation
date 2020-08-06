import torch
import numpy as np
from transformers import AutoTokenizer
RANDOM_SEED = 224
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# Data configuration
TOP_K = 5
TRAIN_START_DATE = "2012-01-01"
TRAIN_END_DATE = "2015-12-31"
VALID_START_DATE = "2016-01-01"
VALID_END_DATE = "2016-12-31"
TEST_START_DATE = "2017-01-01"
TEST_END_DATE = "2020-07-01"

# Model configuration
BATCH_SIZE = 4
MAX_LEN = 32
EPOCHS = 4
DROPOUT_RATE = 0.1
PRE_TRAINED_MODEL_NAME = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./weights/2020-08-04_cnn_distilbert.bin"

# Optimization
# Best fine-tuning learning rate (among 5e-5, 4e-5, 3e-5, and 2e-5) from BERT paper
# OPTIMIZER: 'sgd', 'adam', 'adamw'
# SCHEDULER: 'hd', 'ed', 'cyclic', 'staircase', 'one_cycle', 'linear'
FIND_BEST_LR = False
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 5e-4
FC_WEIGHT = 0.2
CE_WEIGHT = 0.8
OPTIMIZER = "adamw"
SCHEDULER = "one_cycle"
