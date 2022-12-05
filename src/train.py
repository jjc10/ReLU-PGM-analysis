import os
import torch.nn.functional as F
import torch.optim as optim
import torch


def train_model(network, train_loader, test_loader, config_dict, run_path):
    optimizer = optim.SGD(network.parameters(), lr=config_dict['learning_rate'],
                          momentum=config_dict['momentum'])

    train_losses = []
    train_counter = []
    test_losses = []

    def train(epoch):
        network.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = network(data)
            loss = F.nll_loss(output, target, reduction='sum')
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            torch.save(network.state_dict(),
                       os.path.join(run_path, 'model.pth'))
            torch.save(optimizer.state_dict(),
                       os.path.join(run_path, 'optimizer.pth'))

    def test():
        network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = network(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
                test_loss /= len(test_loader.dataset)
                test_losses.append(test_loss)
        print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        # train_loss = 0
        # correct = 0
        # with torch.no_grad():
        #     for data, target in train_loader:
        #         output = network(data)
        #         train_loss += F.nll_loss(output, target,
        #                                  reduction='sum').item()
        #         pred = output.data.max(1, keepdim=True)[1]
        #         correct += pred.eq(target.data.view_as(pred)).sum()
        #         train_loss /= len(train_loader.dataset)
        #         train_losses.append(train_loss)
        # print('\nTrain set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #     train_loss, correct, len(train_loader.dataset),
        #     100. * correct / len(train_loader.dataset)))

    test()
    for epoch in range(1, config_dict['n_epochs'] + 1):
        train(epoch)
        test()
