import sys
import warnings
import logging
import config
import torch
import joblib
import pandas as pd
import numpy as np
from collections import defaultdict
from model import ReutersClassifier
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from torch import nn
from preprocessor import load_data
from visualise import plot_history
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)


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

def matthews_correlation_coefficient(true_pos, true_neg, false_pos, false_neg):
    nominator = (true_pos*true_neg-false_pos*false_neg)
    denominator = np.sqrt((true_pos+false_pos)*(true_pos+false_neg)*(true_neg+false_pos)*(true_neg+false_neg)) + 1e-7
    return (nominator / denominator)

class ProgressBar(object):
    """Cheers @ajratner"""

    def __init__(self, n, length=40):
        # Protect against division by zero
        self.n      = max(1, n)
        self.nf     = float(n)
        self.length = length
        # Precalculate the i values that should trigger a write operation
        self.ticks = set([round(i/100.0 * n) for i in range(101)])
        self.ticks.add(n-1)
        self.bar(0)

    def bar(self, i, message=""):
        """Assumes i ranges through [0, n-1]"""
        if i in self.ticks:
            b = int(np.ceil(((i+1) / self.nf) * self.length))
            sys.stdout.write("\r[{0}{1}] {2}%\t{3}".format(
                "="*b, " "*(self.length-b), int(100*((i+1) / self.nf)), message
            ))
            sys.stdout.flush()

    def close(self, message=""):
        # Move the bar to 100% before closing
        self.bar(self.n-1)
        sys.stdout.write("{0}\n\n".format(message))
        sys.stdout.flush()

def train_distilbert(model, data_loader, loss_function, optimizer, device, scheduler, n_examples):
    model = model.train()

    losses = []
    correct_predictions = 0
    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0

    for i, data in enumerate(data_loader):
        for d in data["ids_and_mask"]:
            d["input_ids"] = d["input_ids"].to(device)
            d["attention_mask"] = d["attention_mask"].to(device)
        targets = data["target"].to(device)

        outputs = model(data["ids_and_mask"])
        _, preds = torch.max(outputs, dim=1)
        loss = loss_function(outputs, targets)

        for p, t in zip(preds, targets):
            if p == 1 and t == 1:
                true_pos += 1
            if p == 0 and t == 0:
                true_neg += 1
            if p == 1 and t == 0:
                false_pos += 1
            if p == 0 and t == 1:
                false_neg += 1

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        for d in data["ids_and_mask"]:
            d["input_ids"] = d["input_ids"].to("cpu")
            d["attention_mask"] = d["attention_mask"].to("cpu")
        targets = data["target"].to("cpu")

    recall = float(true_pos) / float(true_pos + false_neg + 1e-7)
    precision = float(true_pos) / float(true_pos + false_pos + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    accuracy = (true_pos + true_neg) / float(n_examples)
    mcc = matthews_correlation_coefficient(true_pos, true_neg, false_pos, false_neg)

    return accuracy, f1, mcc, np.mean(losses)


def eval_distilbert(model, data_loader, loss_function, device, n_examples):
    model = model.eval()

    losses = []
    correct_predictions = 0
    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0

    with torch.no_grad():
        for data in data_loader:
            for d in data["ids_and_mask"]:
                d["input_ids"] = d["input_ids"].to(device)
                d["attention_mask"] = d["attention_mask"].to(device)
            targets = data["target"].to(device)

            outputs = model(data["ids_and_mask"])
            _, preds = torch.max(outputs, dim=1)
            loss = loss_function(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

            for p, t in zip(preds, targets):
                if p == 1 and t == 1:
                    true_pos += 1
                if p == 0 and t == 0:
                    true_neg += 1
                if p == 1 and t == 0:
                    false_pos += 1
                if p == 0 and t == 1:
                    false_neg += 1

            for d in data["ids_and_mask"]:
                d["input_ids"] = d["input_ids"].to("cpu")
                d["attention_mask"] = d["attention_mask"].to("cpu")
            targets = data["target"].to("cpu")

    recall = float(true_pos) / float(true_pos + false_neg + 1e-7)
    precision = float(true_pos) / float(true_pos + false_pos + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    accuracy = (true_pos + true_neg) / float(n_examples)
    mcc = matthews_correlation_coefficient(true_pos, true_neg, false_pos, false_neg)

    return accuracy, f1, mcc, np.mean(losses)

def test_distilbert(model, data_loader, device):
    model = model.eval()
    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
    with torch.no_grad():
        for data in data_loader:
            for d in data["ids_and_mask"]:
                d["input_ids"] = d["input_ids"].to(device)
                d["attention_mask"] = d["attention_mask"].to(device)
            targets = data["target"].to(device)

            outputs = model(data["ids_and_mask"])
            _, preds = torch.max(outputs, dim=1)

            for p, t in zip(preds, targets):
                if p == 1 and t == 1:
                    true_pos += 1
                if p == 0 and t == 0:
                    true_neg += 1
                if p == 1 and t == 0:
                    false_pos += 1
                if p == 0 and t == 1:
                    false_neg += 1

            for d in data["ids_and_mask"]:
                d["input_ids"] = d["input_ids"].to("cpu")
                d["attention_mask"] = d["attention_mask"].to("cpu")
            targets = data["target"].to("cpu")

    recall = float(true_pos) / float(true_pos + false_neg + 1e-7)
    precision = float(true_pos) / float(true_pos + false_pos + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    mcc = matthews_correlation_coefficient(true_pos, true_neg, false_pos, false_neg)
    print("F1: {:.4f}\nACC: {:.4f}\nMCC: {:.4f}".format(f1, accuracy, mcc))

def main():
    # df = load_data(
    #     ticker_name="GOOG",
    #     news_filename="./data/reuters_news_google_v1.joblib",
    #     labels=["forex", "stock"],
    #     sort_by="stock",
    #     top_k=config.TOP_K)
    # train = df.loc[pd.to_datetime(config.TRAIN_START_DATE).date():pd.to_datetime(config.TRAIN_END_DATE).date()]
    # valid = df.loc[pd.to_datetime(config.VALID_START_DATE).date():pd.to_datetime(config.VALID_END_DATE).date()]
    # test = df.loc[pd.to_datetime(config.TEST_START_DATE).date():pd.to_datetime(config.TEST_END_DATE).date()]
    # joblib.dump(train, "train.bin", compress=3)
    # joblib.dump(valid, "valid.bin", compress=3)

    print("Loading data...")
    train = joblib.load("./data/train.bin")
    valid = joblib.load("./data/valid.bin")
    test = joblib.load("./data/test.bin")
    print("Load data successfully!")

    train_dataloader = create_dataloader(train, config.tokenizer, config.MAX_LEN, config.TOP_K, config.BATCH_SIZE)
    val_dataloader = create_dataloader(valid, config.tokenizer, config.MAX_LEN, config.TOP_K, config.BATCH_SIZE)
    test_dataloader = create_dataloader(test, config.tokenizer, config.MAX_LEN, config.TOP_K, config.BATCH_SIZE)

    torch.cuda.empty_cache()
    model = ReutersClassifier(n_classes=2, top_k=config.TOP_K)
    model.to(config.device)

    optim = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY, correct_bias=False)
    total_steps = len(train_dataloader) * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=0,
        num_training_steps=total_steps)
    loss_function = nn.CrossEntropyLoss().to(config.device)

    history = defaultdict(list)
    best_loss = np.inf

    for epoch in range(config.EPOCHS):
        print("=" * 20)
        print("Epoch {}/{}".format(epoch + 1, config.EPOCHS))

        train_acc, train_f1, train_mcc, train_loss = train_distilbert(
            model, train_dataloader, loss_function, optim, config.device, scheduler, len(train))

        print("Train | Loss: {:.4f} | Accuracy: {:.4f} | F1: {:.4f} | MCC: {:.4f}".format(
            train_loss, train_acc, train_f1, train_mcc))

        val_acc, val_f1, val_mcc, val_loss = eval_distilbert(
            model, val_dataloader, loss_function, config.device, len(valid))

        print("Valid | Loss: {:.4f} | Accuracy: {:.4f} | F1: {:.4f} | MCC: {:.4f}".format(
            val_loss, val_acc, val_f1, val_mcc))

        history["train_f1"].append(train_f1)
        history["train_mcc"].append(train_mcc)
        history["train_acc"].append(train_acc)
        history["train_loss"].append(train_loss)
        history["val_f1"].append(val_f1)
        history["val_mcc"].append(val_mcc)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)

        if val_loss < best_loss:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_loss = val_loss

    test_distilbert(model, test_dataloader, config.device)
    plot_history(history)

if __name__ == "__main__":
    main()