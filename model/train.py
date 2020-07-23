import sys
import warnings
import logging
import config
import math
import torch
import joblib
import pandas as pd
import numpy as np
import torch.optim as optim
from collections import OrderedDict
from functools import reduce
from collections import defaultdict
from model import ReutersClassifier
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.optim.optimizer import required
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
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


def add_metrics_to_log(log, metrics, results, prefix=''):
    for metric, result in metrics, results:
        log[prefix + metric] = str(result)


def log_to_message(log, precision=4):
    fmt = "{0}: {1:." + str(precision) + "f}"
    return "    ".join(fmt.format(k, v) for k, v in log.items())


def progressbar(iter, prefix="", size=60, file=sys.stdout):
    # Reference from https://stackoverflow.com/questions/3160699/python-progress-bar
    count = len(iter)
    def show(t):
        x = int(size*t/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), int(100*t/count), 100))
        file.flush()
    show(0)
    for i, item in enumerate(iter):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()


class ProgressBar(object):
    """Cheers @ajratner"""
    def __init__(self, n, length=40):
        # Protect against division by zero
        self.n = max(1, n)
        self.nf = float(n)
        self.length = length

        # Pre-calculate the i values that should trigger a write operation
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


def train_baseline(
        train_data, valid_data, model, optim_name, lr_scheduler_type, verbose=1, momentum=0.0,
        weight_decay=5e-4, lr_decay=1.0, hyper_lr=1e-8, step_size=30, t_0=10, t_mult=2):
    """
    Reference from https://github.com/awslabs/adatune/blob/master/bin/baselines.py
    Args:
        train_data: pandas dataframe
        valid_data: pandas dataframe
        model: pytorch class
        optim_name: str ('sgd', 'adam', 'adamw)
        lr_scheduler_type: str ('hd', 'ed', 'cyclic', 'staircase')
        verbose: bool (0, 1)
        momentum: float
        weight_decay: float
        lr_decay: float
        hyper_lr: float
        step_size: int
        t_0: int
        t_mult: int

    Returns:
        history plot
    """
    train_dataloader = create_dataloader(train_data, config.tokenizer, config.MAX_LEN, config.TOP_K, config.BATCH_SIZE)
    val_dataloader = create_dataloader(valid_data, config.tokenizer, config.MAX_LEN, config.TOP_K, config.BATCH_SIZE)

    model.to(config.device)
    cur_lr = config.LEARNING_RATE
    scheduler = None
    loss_function = nn.CrossEntropyLoss().to(config.device)
    history = defaultdict(list)
    best_loss = np.inf

    if lr_scheduler_type == 'hd':
        if optim_name == 'adam':
            optimizer = AdamHD(
                model.parameters(), lr=config.LEARNING_RATE, weight_decay=weight_decay,
                eps=1e-4, hypergrad_lr=hyper_lr)
        elif optim_name == 'sgd':
            optimizer = SGDHD(
                model.parameters(), lr=config.LEARNING_RATE, momentum=momentum,
                weight_decay=weight_decay, hypergrad_lr=hyper_lr)
        else:
            print("Only can choose either adam or sgd so far...")
    else:
        if optim_name == 'adam':
            optimizer = optim.Adam(
                model.parameters(), lr=config.LEARNING_RATE, weight_decay=weight_decay, eps=1e-4)
        elif optim_name == 'sgd':
            optimizer = optim.SGD(
                model.parameters(), lr=config.LEARNING_RATE, momentum=momentum, weight_decay=weight_decay)
        else:
            optimizer = AdamW(
                model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY, correct_bias=False)

        if lr_scheduler_type == 'ed':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
        elif lr_scheduler_type == 'staircase':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=lr_decay)
        elif lr_scheduler_type == 'cyclic':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=t_0, T_mult=t_mult, eta_min=config.LEARNING_RATE * 1e-4)

    logs = []
    for epoch in range(config.EPOCHS):
        model = model.train()
        print("\n")
        print("Epoch {}/{}".format(epoch + 1, config.EPOCHS))

        if verbose:
            pb = ProgressBar(len(train_data))
        log = OrderedDict()

        losses = []
        correct_predictions = 0
        true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0

        for i, data in enumerate(progressbar(train_dataloader)):
            for d in data["ids_and_mask"]:
                d["input_ids"] = d["input_ids"].to(config.device)
                d["attention_mask"] = d["attention_mask"].to(config.device)
            targets = data["target"].to(config.device)

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
            optimizer.zero_grad()
            if scheduler and lr_scheduler_type == 'cyclic':
                scheduler.step(epoch + (i / len(train_data)))

            for d in data["ids_and_mask"]:
                d["input_ids"] = d["input_ids"].to("cpu")
                d["attention_mask"] = d["attention_mask"].to("cpu")
            targets = data["target"].to("cpu")

            log['loss'] = np.mean(losses)
            if verbose:
                pb.bar(i, log_to_message(log))

        train_recall = float(true_pos) / float(true_pos + false_neg + 1e-7)
        train_precision = float(true_pos) / float(true_pos + false_pos + 1e-7)
        train_f1 = 2 * train_precision * train_recall / (train_precision + train_recall + 1e-7)
        train_accuracy = (true_pos + true_neg) / float(true_pos + true_neg + false_pos + false_neg)
        train_mcc = matthews_correlation_coefficient(true_pos, true_neg, false_pos, false_neg)

        print("Train | Loss: {:.4f} | Accuracy: {:.4f} | F1: {:.4f} | MCC: {:.4f}".format(
            np.mean(losses), train_accuracy, train_f1, train_mcc))

        val_acc, val_f1, val_mcc, val_loss = eval_distilbert(
            model, val_dataloader, loss_function, config.device, len(valid_data))
        log['val_loss'] = val_loss

        print("Valid | Loss: {:.4f} | Accuracy: {:.4f} | F1: {:.4f} | MCC: {:.4f}".format(
            val_loss, val_acc, val_f1, val_mcc))

        history["train_f1"].append(train_f1)
        history["train_mcc"].append(train_mcc)
        history["train_acc"].append(train_accuracy)
        history["train_loss"].append(np.mean(losses))
        history["val_f1"].append(val_f1)
        history["val_mcc"].append(val_mcc)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)

        if val_loss < best_loss:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_loss = val_loss

        logs.append(log)
        if verbose:
            pb.close(log_to_message(log))

        cur_lr = 0.0
        for param_group in optimizer.param_groups:
            cur_lr = param_group['lr']
        print('Learning rate after epoch {} is: {:.6f}'.format(epoch+1, cur_lr))

    plot_history(history)


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


class AdamHD(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, hypergrad_lr=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, hypergrad_lr=hypergrad_lr)
        super(AdamHD, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                if state['step'] > 1:
                    prev_bias_correction1 = 1 - beta1 ** (state['step'] - 1)
                    prev_bias_correction2 = 1 - beta2 ** (state['step'] - 1)
                    # Hypergradient for Adam:
                    h = torch.dot(grad.view(-1),
                                  torch.div(exp_avg, exp_avg_sq.sqrt().add_(group['eps'])).view(-1)) * math.sqrt(
                        prev_bias_correction2) / prev_bias_correction1
                    # Hypergradient descent of the learning rate:
                    group['lr'] += group['hypergrad_lr'] * h

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss


class SGDHD(Optimizer):

    def __init__(self, params, lr=required, momentum=0.0, dampening=0,
                 weight_decay=0.0, nesterov=False, hypergrad_lr=1e-6):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, hypergrad_lr=hypergrad_lr)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDHD, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("SGDHD doesn't support per-parameter options (parameter groups)")

        self._params = self.param_groups[0]['params']
        self._params_numel = reduce(lambda total, p: total + p.numel(), self._params, 0)

    def _gather_flat_grad_with_weight_decay(self, weight_decay=0):
        views = []
        for p in self._params:
            if p.grad is None:
                view = torch.zeros_like(p.data)
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            if weight_decay != 0:
                view.add_(weight_decay, p.data.view(-1))
            views.append(view)
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.data.add_(step_size, update[offset:offset + numel].view_as(p.data))
            offset += numel
        assert offset == self._params_numel

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        loss = None
        if closure is not None:
            loss = closure()

        group = self.param_groups[0]
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']

        grad = self._gather_flat_grad_with_weight_decay(weight_decay)

        # NOTE: SGDHD has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict
        state = self.state[self._params[0]]
        # State initialization
        if len(state) == 0:
            state['grad_prev'] = torch.zeros_like(grad)

        grad_prev = state['grad_prev']
        # Hypergradient for SGD
        h = torch.dot(grad, grad_prev)
        # Hypergradient descent of the learning rate:
        group['lr'] += group['hypergrad_lr'] * h

        if momentum != 0:
            if 'momentum_buffer' not in state:
                buf = state['momentum_buffer'] = torch.zeros_like(grad)
                buf.mul_(momentum).add_(grad)
            else:
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(1 - dampening, grad)
            if nesterov:
                grad.add_(momentum, buf)
            else:
                grad = buf

        state['grad_prev'] = grad

        self._add_grad(-group['lr'], grad)

        return loss


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


def main2():
    print("Loading data...")
    train_data = joblib.load("./data/train.bin")
    valid_data = joblib.load("./data/valid.bin")
    print("Load data successfully!")

    model = ReutersClassifier(n_classes=2, top_k=config.TOP_K)
    model.to(config.device)
    print("Load model successfully!")

    train_baseline(train_data, valid_data, model, optim_name="adam", lr_scheduler_type="hd", verbose=0)


if __name__ == "__main__":
    main2()