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
from model import ReutersClassifier
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.optim.optimizer import required
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import (accuracy_score, roc_curve, auc)
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
    return nominator/denominator


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


def train_baseline(
        train_data, valid_data, model, optim_name, lr_scheduler_type, momentum=0.0,
        weight_decay=5e-4, lr_decay=1.0, hyper_lr=1e-8, step_size=30, t_0=10, t_mult=2):
    """
    Reference from https://github.com/awslabs/adatune/blob/master/bin/baselines.py
    Args:
        train_data: pandas dataframe
        valid_data: pandas dataframe
        model: pytorch class
        optim_name: str ('sgd', 'adam', 'adamw)
        lr_scheduler_type: str ('hd', 'ed', 'cyclic', 'staircase', 'one_cycle', 'linear')
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

    # Hypergradient Descent: https://arxiv.org/pdf/1703.04782.pdf
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
            print("In HD, only can choose either ADAM or SGD so far...")
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

        # Exponential Decay
        if lr_scheduler_type == 'ed':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
        # Staircase Decay
        elif lr_scheduler_type == 'staircase':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=lr_decay)
        # Cosine Annealing with Restarts
        elif lr_scheduler_type == 'cyclic':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=t_0, T_mult=t_mult, eta_min=config.LEARNING_RATE * 1e-4)
        # One Cycle Policy Learning Rate Scheduler
        elif lr_scheduler_type == 'one_cycle':
            total_steps = len(train_dataloader) * config.EPOCHS
            scheduler = OneCycleLR(optimizer, num_steps=total_steps, lr_range=(1e-8, 1e-3))
            # One Cycle Policy Learning Rate Scheduler
        elif lr_scheduler_type == 'linear':
            total_steps = len(train_dataloader) * config.EPOCHS
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    print("\nTrain on {} samples, validate on {} samples".format(train_data.shape[0], valid_data.shape[0]))
    print("Optimizer using {} | Scheduler using {}".format(optim_name, lr_scheduler_type))
    print("Total Epoch: {} | Batch Size {}".format(config.EPOCHS, config.BATCH_SIZE))
    for epoch in range(config.EPOCHS):
        t0_epoch, t0_batch = time.time(), time.time()
        model = model.train()
        print("Epoch {}/{}".format(epoch + 1, config.EPOCHS))

        losses = []
        batch_time_elapsed = []
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
                scheduler.step(int(epoch+(i/len(train_data))))

            if scheduler and lr_scheduler_type != 'cyclic':
                scheduler.step()

            for d in data["ids_and_mask"]:
                d["input_ids"] = d["input_ids"].to("cpu")
                d["attention_mask"] = d["attention_mask"].to("cpu")
            targets = data["target"].to("cpu")

            time_elapsed = time.time() - t0_batch
            batch_time_elapsed.append(time_elapsed)

        train_recall = float(true_pos) / float(true_pos + false_neg + 1e-7)
        train_precision = float(true_pos) / float(true_pos + false_pos + 1e-7)
        train_f1 = 2 * train_precision * train_recall / (train_precision + train_recall + 1e-7)
        train_accuracy = (true_pos + true_neg) / float(true_pos + true_neg + false_pos + false_neg)
        train_mcc = matthews_correlation_coefficient(true_pos, true_neg, false_pos, false_neg)

        print("Train | Loss: {:.4f} | Accuracy: {:.4f} | F1: {:.4f} | MCC: {:.4f}".format(
            np.mean(losses), train_accuracy, train_f1, train_mcc))

        val_acc, val_f1, val_mcc, val_loss = eval_distilbert(
            model, val_dataloader, loss_function, config.device, len(valid_data))
        time_elapsed = time.time() - t0_epoch

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

        cur_lr = 0.0
        for param_group in optimizer.param_groups:
            cur_lr = param_group['lr']
        print('Learning rate after epoch {} is: {:.6f}\n'.format(epoch+1, cur_lr))
        print("Elapsed time per batch: {}\nTotal time elapsed: {}".format(np.mean(batch_time_elapsed), time_elapsed))

    # plot_history(history)
    return history


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


class OneCycleLR:
    """ Sets the learing rate of each parameter group by the one cycle learning rate policy
    proposed in https://arxiv.org/pdf/1708.07120.pdf.
    It is recommended that you set the max_lr to be the learning rate that achieves
    the lowest loss in the learning rate range test, and set min_lr to be 1/10 th of max_lr.
    So, the learning rate changes like min_lr -> max_lr -> min_lr -> final_lr,
    where final_lr = min_lr * reduce_factor.
    Note: Currently only supports one parameter group.
    Args:
        optimizer:             (Optimizer) against which we apply this scheduler
        num_steps:             (int) of total number of steps/iterations
        lr_range:              (tuple) of min and max values of learning rate
        momentum_range:        (tuple) of min and max values of momentum
        annihilation_frac:     (float), fracion of steps to annihilate the learning rate
        reduce_factor:         (float), denotes the factor by which we annihilate the learning rate at the end
        last_step:             (int), denotes the last step. Set to -1 to start training from the beginning

    Useful resources:
        https://towardsdatascience.com/finding-good-learning-rate-and-the-one-cycle-policy-7159fe1db5d6
        https://medium.com/vitalify-asia/whats-up-with-deep-learning-optimizers-since-adam-5c1d862b9db0
    """

    def __init__(self,
                 optimizer: Optimizer,
                 num_steps: int,
                 lr_range: tuple = (0.1, 1.),
                 momentum_range: tuple = (0.85, 0.95),
                 annihilation_frac: float = 0.1,
                 reduce_factor: float = 0.01,
                 last_step: int = -1):
        # Sanity check
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(type(optimizer).__name__))
        self.optimizer = optimizer

        self.num_steps = num_steps

        self.min_lr, self.max_lr = lr_range[0], lr_range[1]
        assert self.min_lr < self.max_lr, \
            "Argument lr_range must be (min_lr, max_lr), where min_lr < max_lr"

        self.min_momentum, self.max_momentum = momentum_range[0], momentum_range[1]
        assert self.min_momentum < self.max_momentum, \
            "Argument momentum_range must be (min_momentum, max_momentum), where min_momentum < max_momentum"

        self.num_cycle_steps = int(num_steps * (1. - annihilation_frac))  # Total number of steps in the cycle
        self.final_lr = self.min_lr * reduce_factor

        self.last_step = last_step

        if self.last_step == -1:
            self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer. (Borrowed from _LRScheduler class in torch.optim.lr_scheduler.py)
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state. (Borrowed from _LRScheduler class in torch.optim.lr_scheduler.py)
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def get_momentum(self):
        return self.optimizer.param_groups[0]['momentum']

    def step(self):
        """
        Conducts one step of learning rate and momentum update
        """
        current_step = self.last_step + 1
        self.last_step = current_step

        if current_step <= self.num_cycle_steps // 2:
            # Scale up phase
            scale = current_step / (self.num_cycle_steps // 2)
            lr = self.min_lr + (self.max_lr - self.min_lr) * scale
            momentum = self.max_momentum - (self.max_momentum - self.min_momentum) * scale
        elif current_step <= self.num_cycle_steps:
            # Scale down phase
            scale = (current_step - self.num_cycle_steps // 2) / (self.num_cycle_steps - self.num_cycle_steps // 2)
            lr = self.max_lr - (self.max_lr - self.min_lr) * scale
            momentum = self.min_momentum + (self.max_momentum - self.min_momentum) * scale
        elif current_step <= self.num_steps:
            # Annihilation phase: only change lr
            scale = (current_step - self.num_cycle_steps) / (self.num_steps - self.num_cycle_steps)
            lr = self.min_lr - (self.min_lr - self.final_lr) * scale
            momentum = None
        else:
            # Exceeded given num_steps: do nothing
            return

        self.optimizer.param_groups[0]['lr'] = lr
        if momentum:
            self.optimizer.param_groups[0]['momentum'] = momentum


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


def plot_find_lr(train_data, net, optimizer, criterion, init_value=1e-8, final_value=10., beta=0.98):
    # Reference from https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html?utm_source=hacpai.com
    train_dataloader = create_dataloader(train_data, config.tokenizer, config.MAX_LEN, config.TOP_K, config.BATCH_SIZE)
    num = len(train_dataloader)-1
    mult = (final_value / init_value) ** (1/num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []

    for data in progressbar(train_dataloader):
        batch_num += 1
        # As before, get the loss for this mini-batch of inputs/outputs
        for d in data["ids_and_mask"]:
            d["input_ids"] = d["input_ids"].to(config.device)
            d["attention_mask"] = d["attention_mask"].to(config.device)
        labels = data["target"].to(config.device)
        optimizer.zero_grad()
        outputs = net(data["ids_and_mask"])
        loss = criterion(outputs, labels)
        # Compute the smoothed loss
        avg_loss = beta * avg_loss + (1-beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        # Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        # Record the best loss
        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss
        # Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        # Do the SGD step
        loss.backward()
        optimizer.step()
        # Update the lr for the next step
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr

    plt.figure(figsize=(15, 5))
    # The skip of the first 10 values and the last 5 is another thing that the fastai library does by default, to
    # remove the initial and final high losses and focus on the interesting parts of the graph.
    plt.plot(log_lrs[10:-5], losses[10:-5])
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()


# First version without progress bar
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


# Second version with progress bar, different optimizer, and different scheduler
def main2():
    print("Loading data...")
    train_data = joblib.load("./data/train_top10.bin")
    valid_data = joblib.load("./data/valid_top10.bin")
    print("Load data successfully!")

    model = ReutersClassifier(n_classes=2, top_k=config.TOP_K)
    model.unfreeze_bert_encoder()
    model.to(config.device)
    print("Load model successfully!")

    history = train_baseline(train_data, valid_data, model, optim_name="adam", lr_scheduler_type="one_cycle")
    plot_history(history)


# Find best learning rate
def main3():
    print("Loading data...")
    train_data = joblib.load("./data/train_top10.bin")
    valid_data = joblib.load("./data/valid_top10.bin")
    print("Load data successfully!")

    model = ReutersClassifier(n_classes=2, top_k=config.TOP_K)
    model.unfreeze_bert_encoder()
    model.to(config.device)
    print("Load model successfully!")

    criterion = nn.CrossEntropyLoss().to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=1e-1)
    plot_find_lr(train_data, model, optimizer, criterion, init_value=1e-8, final_value=10., beta=0.98)


if __name__ == "__main__":
    main2()