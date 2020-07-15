import warnings
import joblib
import torch
import pandas as pd
import numpy as np
from collections import defaultdict
from transformers import AutoTokenizer
from transformers import AutoModel
from torch import nn
from torch.nn import functional as F
from preprocessor import load_data

warnings.filterwarnings("ignore")


class ReutersClassifier(nn.Module):

    def __init__(self, n_classes, p=0.25):
        super(ReutersClassifier, self).__init__()
        self.PRE_TRAINED_MODEL_NAME = 'distilbert-base-uncased'
        self.distilbert_layer = AutoModel.from_pretrained(self.PRE_TRAINED_MODEL_NAME)
        self.dropout = nn.Dropout(p=p)
        self.classifier = nn.Linear(self.distilbert_layer.config.dim*3, n_classes)

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, input_ids_3, attention_mask_3):
        pooled_output_1 = self.distilbert_layer(
            input_ids=input_ids_1,
            attention_mask=attention_mask_1)
        pooled_output_2 = self.distilbert_layer(
            input_ids=input_ids_2,
            attention_mask=attention_mask_2)
        pooled_output_3 = self.distilbert_layer(
            input_ids=input_ids_3,
            attention_mask=attention_mask_3)
        branch_1 = self.dropout(pooled_output_1[0][:, 0, :])
        branch_2 = self.dropout(pooled_output_2[0][:, 0, :])
        branch_3 = self.dropout(pooled_output_3[0][:, 0, :])
        main = torch.cat([branch_1, branch_2, branch_3], 1)
        return F.softmax(self.classifier(main))

def main():
    PRE_TRAINED_MODEL_NAME = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    model = ReutersClassifier(n_classes=2)
    model.to(device)
    print("Model built!")

    df = load_data(file_name="./data/reuters_news.joblib")
    sample = tokenizer.encode_plus(
        df["Top 1 News"][0],
        max_length=32,
        add_special_tokens=True,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors="pt")

    with torch.no_grad():
        probabilty = model(sample["input_ids"].to(device), sample["attention_mask"].to(device))
    print(probabilty)

if __name__ == "__main__":
    main()