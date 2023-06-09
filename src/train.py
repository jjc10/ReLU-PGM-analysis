import os
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch
from torch.utils.tensorboard import SummaryWriter

from src.config import set_up_paths


def count_gradient_norm(parameters):
    total_norm = 0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def train_model(network, train_loader, test_loader, config_dict, run_path):
    log_dir = os.path.join(run_path, 'logs')
    set_up_paths([log_dir])
    writer = SummaryWriter(log_dir=log_dir)
    optimizer = optim.SGD(network.parameters(), lr=config_dict['learning_rate'],
                          momentum=config_dict['momentum'])

    test_losses = []

    def train(epoch):
        train_losses = []
        train_counter = []
        gradient = []
        network.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = network(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            gradient.append(count_gradient_norm(network.parameters()))
            optimizer.step()
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))

        total_loss = np.mean(train_losses)
        total_gradient = np.mean(gradient)
        writer.add_scalar('Loss/train', total_loss, epoch)
        writer.add_scalar('gradient', total_gradient, epoch)

    def test():
        network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = network(data)
                test_loss += F.nll_loss(output, target).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
                test_loss /= len(test_loader.dataset)
                test_losses.append(test_loss)
        print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            100*test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    test()
    for epoch in range(1, config_dict['n_epochs'] + 1):
        train(epoch)
        test()
    torch.save(network.state_dict(),
               os.path.join(run_path, 'model.pth'))
    torch.save(optimizer.state_dict(),
               os.path.join(run_path, 'optimizer.pth'))
