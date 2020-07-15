import warnings
import config
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from model import ReutersClassifier
from transformers import AutoTokenizer
from transformers import AutoModel
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from torch import nn
from torch.nn import functional as F
from preprocessor import load_data
warnings.filterwarnings("ignore")


class ReutersDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.targets = df.label.values
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.len = len(self.df)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        top_1, top_2, top_3 = self.df.iloc[item, 1:4].values
        target = self.targets[item]

        encoding_1 = self.tokenizer.encode_plus(
            str(top_1),
            max_length=self.max_len,
            add_special_tokens=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt")

        encoding_2 = self.tokenizer.encode_plus(
            str(top_2),
            max_length=self.max_len,
            add_special_tokens=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt")

        encoding_3 = self.tokenizer.encode_plus(
            str(top_3),
            max_length=self.max_len,
            add_special_tokens=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt")

        return {
            "tweet_text_1": str(top_1),
            "tweet_text_2": str(top_2),
            "tweet_text_3": str(top_3),
            "input_ids_1": encoding_1["input_ids"].flatten(),
            "input_ids_2": encoding_2["input_ids"].flatten(),
            "input_ids_3": encoding_3["input_ids"].flatten(),
            "attention_mask_1": encoding_1["attention_mask"].flatten(),
            "attention_mask_2": encoding_2["attention_mask"].flatten(),
            "attention_mask_3": encoding_3["attention_mask"].flatten(),
            "target": torch.tensor(target, dtype=torch.long)
        }

def create_dataloader(df, tokenizer, max_len, batch_size):
    dataset = ReutersDataset(
        df=df,
        tokenizer=tokenizer,
        max_len=max_len)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0)

def matthews_correlation_coefficient(true_pos, true_neg, false_pos, false_neg):
    nominator = (true_pos*true_neg-false_pos*false_neg)
    denominator = np.sqrt((true_pos+false_pos)*(true_pos+false_neg)*(true_neg+false_pos)*(true_neg+false_neg)) + 1e-7
    return (nominator / denominator)

def train_distilbert(model, data_loader, loss_function, optimizer, device, scheduler, n_examples):
    model = model.train()

    losses = []
    correct_predictions = 0
    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0

    for data in data_loader:
        input_ids_1 = data["input_ids_1"].to(device)
        input_ids_2 = data["input_ids_2"].to(device)
        input_ids_3 = data["input_ids_3"].to(device)
        attention_mask_1 = data["attention_mask_1"].to(device)
        attention_mask_2 = data["attention_mask_2"].to(device)
        attention_mask_3 = data["attention_mask_3"].to(device)
        targets = data["target"].to(device)

        outputs = model(input_ids_1, input_ids_2, input_ids_3,
                        attention_mask_1, attention_mask_2, attention_mask_3)
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
            input_ids_1 = data["input_ids_1"].to(device)
            input_ids_2 = data["input_ids_2"].to(device)
            input_ids_3 = data["input_ids_3"].to(device)
            attention_mask_1 = data["attention_mask_1"].to(device)
            attention_mask_2 = data["attention_mask_2"].to(device)
            attention_mask_3 = data["attention_mask_3"].to(device)
            targets = data["target"].to(device)

            outputs = model(input_ids_1, input_ids_2, input_ids_3,
                            attention_mask_1, attention_mask_2, attention_mask_3)
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

    recall = float(true_pos) / float(true_pos + false_neg + 1e-7)
    precision = float(true_pos) / float(true_pos + false_pos + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    accuracy = (true_pos + true_neg) / float(n_examples)
    mcc = matthews_correlation_coefficient(true_pos, true_neg, false_pos, false_neg)

    return accuracy, f1, mcc, np.mean(losses)

def plot_history(history):
    plt.figure(figsize=(15, 6))

    plt.subplot(2, 2, 1)
    plt.plot([int(x + 1) for x in range(config.EPOCHS)], history["train_mcc"], label="train mcc")
    plt.plot([int(x + 1) for x in range(config.EPOCHS)], history["val_mcc"], label="validation mcc")
    plt.title("Training MCC History")
    plt.ylabel("MCC")
    plt.xlabel("Epoch")
    plt.ylim([-1, 1])
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot([int(x + 1) for x in range(config.EPOCHS)], history["train_f1"], label="train f1")
    plt.plot([int(x + 1) for x in range(config.EPOCHS)], history["val_f1"], label="validation f1")
    plt.title("Training F1 History")
    plt.ylabel("F1")
    plt.xlabel("Epoch")
    plt.ylim([0, 1])
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.plot([int(x + 1) for x in range(config.EPOCHS)], history["train_acc"], label="train acc")
    plt.plot([int(x + 1) for x in range(config.EPOCHS)], history["val_acc"], label="validation acc")
    plt.title("Training Accuracy History")
    plt.ylabel("F1")
    plt.xlabel("Epoch")
    plt.ylim([0, 1])
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot([int(x + 1) for x in range(config.EPOCHS)], history["train_loss"], label="train loss")
    plt.plot([int(x + 1) for x in range(config.EPOCHS)], history["val_loss"], label="validation loss")
    plt.title("Training Loss History")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.ylim([0, 1])
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

def main():
    df = load_data()
    train = df.loc[:pd.to_datetime("2017-01-01").date()]
    valid = df.loc[pd.to_datetime("2017-01-01").date():]

    train_dataloader = create_dataloader(train, config.tokenizer, config.MAX_LEN, config.BATCH_SIZE)
    val_dataloader = create_dataloader(valid, config.tokenizer, config.MAX_LEN, config.BATCH_SIZE)

    torch.cuda.empty_cache()
    model = ReutersClassifier(2)
    model.to(config.device)

    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.417, correct_bias=False)
    total_steps = len(train_dataloader) * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps)
    loss_function = nn.CrossEntropyLoss().to(config.device)

    history = defaultdict(list)
    best_f1 = 0

    for epoch in range(config.EPOCHS):
        print("=" * 20)
        print("Epoch {}/{}".format(epoch + 1, config.EPOCHS))

        train_acc, train_f1, train_mcc, train_loss = train_distilbert(
            model, train_dataloader, loss_function, optimizer, config.device, scheduler, len(train))

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

        if val_f1 > best_f1:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_f1 = val_f1

    plot_history(history)

if __name__ == "__main__":
    main()