import torch
import torch.autograd as ag

def clear_grad(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.detach()
            p.grad.zero_()


def compute_loss_accuracy(net, data_loader, criterion, device):
    net.eval()
    correct = 0
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            total_loss += criterion(outputs, labels).item()
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()

    return total_loss, 100.0 * (correct / len(data_loader.dataset))


class ValidationGradient(object):
    def __init__(self, val_data, criterion, device):
        self.val_data = val_data
        self.val_iter = iter(self.val_data)
        self.device = device
        self.criterion = criterion.to(self.device)

    def val_grad(self, net):
        try:
            val_inputs, val_labels = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_data)
            val_inputs, val_labels = next(self.val_iter)

        val_inputs, val_labels = val_inputs.to(self.device), val_labels.to(self.device)

        val_outputs = net(val_inputs)
        val_loss_node = self.criterion(val_outputs, val_labels)

        return ag.grad(val_loss_node, net.parameters())

