import sys
import time
import warnings
import logging
import config
import math
import torch
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from functools import reduce
from collections import defaultdict
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModel
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.optim.optimizer import Optimizer
from torch.optim.optimizer import required
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import (accuracy_score, roc_curve, auc)
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)


# =======================================
#               Utils
# =======================================
def print_time(func):
    def decorated_func(*args, **kwargs):
        s = time.time()
        ret = func(*args, **kwargs)
        e = time.time()

        print(f"Spend {e - s:.3f} s")
        return ret

    return decorated_func


def progressbar(iter, prefix="", size=60, file=sys.stdout):
    # Reference from https://stackoverflow.com/questions/3160699/python-progress-bar
    count = len(iter)
    def show(t):
        x = int(size*t/count)
        # file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), int(100*t/count), 100))
        file.write("{}[{}{}] {}%\r".format(prefix, "#"*x, "."*(size-x), int(100*t/count)))
        file.flush()
    show(0)
    for i, item in enumerate(iter):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()


# =======================================
#               Model
# =======================================
class ReutersDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, top_k):
        self.df = df
        self.targets = df.label.values
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.top_k = top_k
        self.len = len(self.df)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        total_top = self.df.iloc[item, 0:self.top_k].values
        target = self.targets[item]

        enc_list = []
        for k in total_top:
            enc = self.tokenizer.encode_plus(
                str(k),
                max_length=self.max_len,
                add_special_tokens=True,
                return_token_type_ids=False,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors="pt")
            enc["input_ids"] = enc["input_ids"].flatten()
            enc["attention_mask"] = enc["attention_mask"].flatten()
            enc_list.append(enc)

        return {
            "ids_and_mask": enc_list,
            "target": torch.tensor(target)
        }


def create_dataloader(df, tokenizer, max_len, top_k, batch_size):
    dataset = ReutersDataset(
        df=df,
        tokenizer=tokenizer,
        max_len=max_len,
        top_k=top_k)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0)


class CnnMaxPooling(nn.Module):

    def __init__(self, word_dim, window_size, out_channels):
        super(CnnMaxPooling, self).__init__()
        self.cnn = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(window_size, config.TOP_K))

    def forward(self, x):
        # x input: (batch, seq_len, word_dim)
        x_unsqueeze = x.unsqueeze(1)
        x_cnn = self.cnn(x_unsqueeze)
        x_cnn_result = x_cnn.squeeze(3)
        res, _ = x_cnn_result.max(dim=2)
        return res


class ReutersClassifier(nn.Module):

    def __init__(self, n_classes, top_k, p, window_size, out_channels):
        super(ReutersClassifier, self).__init__()
        self.PRE_TRAINED_MODEL_NAME = 'distilbert-base-uncased'
        self.distilbert_layer = AutoModel.from_pretrained(self.PRE_TRAINED_MODEL_NAME)
        self.dropout = nn.Dropout(p=p)
        self.cnn_max_pool = CnnMaxPooling(
            word_dim=self.distilbert_layer.config.dim, window_size=window_size, out_channels=out_channels)
        self.classifier = nn.Linear(in_features=out_channels, out_features=n_classes)

    def forward(self, ids_and_mask):
        pool_list = []
        for enc in ids_and_mask:
            pooled_output = self.distilbert_layer(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"])
            branch = self.dropout(pooled_output[0][:, 0, :])
            pool_list.append(branch.unsqueeze(2))
        concat = torch.cat([br for br in pool_list], 2)
        doc_embed = self.cnn_max_pool(concat)
        class_score = self.classifier(doc_embed)
        return class_score

    def freeze_bert_encoder(self):
        for param in self.distilbert_layer.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.distilbert_layer.parameters():
            param.requires_grad = True


# =======================================
#               Visualize
# =======================================
def evaluate_roc(probs, y_true):
    """
    - Print AUC and accuracy on the test set
    - Plot ROC
    @params    probs (np.array): an array of predicted probabilities with shape (len(y_true), 2)
    @params    y_true (np.array): an array of the true values with shape (len(y_true),)
    """
    preds = probs[:, 1]
    fpr, tpr, threshold = roc_curve(y_true, preds)
    roc_auc = auc(fpr, tpr)
    print(f'AUC: {roc_auc:.4f}')

    # Get accuracy over the test set
    y_pred = np.where(preds >= 0.5, 1, 0)
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # Plot ROC AUC
    plt.figure(figsize=(15, 5))
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.grid()
    plt.show()
