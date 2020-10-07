import argparse
import numpy as np
import os
import torch
import torch.nn.modules as nn
import torch.utils.tensorboard as tensorboard
import dataloader
from models import main_models


def step1(args):
    print(args)
    epochs = args.e1
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    tfwriter = tensorboard.SummaryWriter(os.path.join(args.s, 'record'))
    train_loader = dataloader.mnist_loader(batch_size=args.b)
    test_loader = dataloader.mnist_loader(batch_size=args.b, train=False)
    classifier = main_models.Classifier().cuda()
    encoder = main_models.Encoder().cuda()
    loss_cls = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(params=list(encoder.parameters()) + list(classifier.parameters()), lr=args.lr)
    input('==> step 1 ready')
    for cur_epoch in range(1, epochs + 1):
        for cur_step, (imgs, labels) in enumerate(train_loader, 1):
            imgs, labels = imgs.cuda(), labels.cuda()
            optimizer.zero_grad()
            y_pred = classifier(encoder(imgs))
            loss = loss_cls(y_pred, labels)
            loss.backward()
            optimizer.step()
            tfwriter.add_scalar('Training Loss', loss.item(), cur_step + (cur_epoch - 1) * len(train_loader))
            if cur_step % 10 == 0:
                print('Epoch: %d/%d\tStep: %d/%d\tLoss: %.6f' % (cur_epoch, epochs, cur_step, len(train_loader), loss.item()))
        total, correct = 0, 0
        for imgs, labels in test_loader:
            imgs, labels = imgs.cuda(), labels.cuda()
            out = classifier(encoder(imgs))
            _, pred = torch.max(out, dim=1)
            total += len(labels)
            correct += (pred == labels).sum().item()
        print('Validation Accuracy: %.6f' % (correct / total))
        tfwriter.add_scalar('Validation Accuracy', correct / total, cur_epoch)
    print('Train finished')
    encoder.save(os.path.join(args.s, 'encoder.ckpt'))
    classifier.save(os.path.join(args.s, 'classifier.ckpt'))
    print('Model saved')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e1', type=int, default=10, help='train epochs for step 1, default: 10')
    parser.add_argument('-e2', type=int, default=100, help='train epochs for step 2, default: 100')
    parser.add_argument('-e3', type=int, default=100, help='train epochs for step 3, default: 100')
    parser.add_argument('-t', type=int, default=7, help='target samples, default: 7')
    parser.add_argument('-b', type=int, default=64, help='train batch size, default: 64')
    parser.add_argument('-g', type=int, default=0, help='which gpu to use, default: 0')
    parser.add_argument('-lr', type=float, default=1e-3, help='learning rate, default: 1e-3')
    parser.add_argument('-s', type=str, default='result', help='path to save result and models, default: result')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.g)
    a1 = input('Train Step 1 ? (Y/N)')
    if a1 == 'Y':
        step1(args)
