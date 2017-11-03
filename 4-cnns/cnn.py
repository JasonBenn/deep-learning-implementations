# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import glob
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models
from torchvision import transforms as t
import matplotlib.pyplot as plt
import time
import os

plt.ion()   # interactive mode


DATA_PATH = "/home/jbenn/data/hymenoptera/"
PHASES = ['train', 'val']
MEAN_TRANSFORM = np.array([0.485, 0.456, 0.406])
STD_TRANSFORM = np.array([0.229, 0.224, 0.225])
BATCH_SIZE = 4

transforms = {
    'train': t.Compose([
        t.RandomSizedCrop(224),
        t.RandomHorizontalFlip(),
        t.ToTensor(),
#         t.Normalize(MEAN_TRANSFORM, STD_TRANSFORM)
    ]),
    'val': t.Compose([
        t.Scale(256),
        t.CenterCrop(224),
        t.ToTensor(),
#         t.Normalize(MEAN_TRANSFORM, STD_TRANSFORM)
    ])
}

image_folders = {
    phase: datasets.ImageFolder(DATA_PATH + phase, transforms[phase])
    for phase in PHASES
}

class_names = image_folders['val'].classes

dataloaders = { phase: torch.utils.data.DataLoader(
        dataset=image_folders[phase],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    ) for phase in PHASES }


def imshow(inp):
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
    plt.imshow((inp * 255).numpy().transpose(1, 2, 0).astype("uint8"))
#     print(inp * 255)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)  # pause a bit so that plots are updated

# inputs, classes = next(iter(dataloaders['train']))
# grid = torchvision.utils.make_grid(inputs)
# ValueError: Floating point image RGB values must be in the 0..1 range.
# imshow(grid, title=[class_names[x] for x in classes])
imshow(inputs[0])


dataset_sizes = { phase: len(image_folders[phase]) for phase in PHASES }

def train(model, criterion, optimizer, num_epochs):
    losses = {}
    loss_history = { 'train': [], 'val': [] }

    for epoch in range(num_epochs):
        for phase in PHASES:
            losses[phase] = 0
            if phase == 'train':
                model.train()
            elif phase == 'val':
                model.eval()

            for inputs, classes in dataloaders[phase]:
                inputs = Variable(inputs.cuda(1))
                classes = Variable(classes.cuda(1))

                optimizer.zero_grad()
                outputs = model(inputs)
                predictions, prediction_indexes = torch.max(outputs.data, 1)

                loss = criterion(outputs, classes)

                losses[phase] += loss.data[0] / dataset_sizes[phase]
                loss_history[phase].append(loss.data[0])

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

        print("epoch {}\t train: {:.4f}\t val: {:.4f}".format(epoch, losses['train'], losses['val']))

    torch.save(model.state_dict(), "last_weights")
    return loss_history


CONV_STRIDE = 3

class VGGish(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, CONV_STRIDE, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, CONV_STRIDE, padding=1),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, CONV_STRIDE, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, CONV_STRIDE, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, CONV_STRIDE, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, CONV_STRIDE, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.fc = nn.Linear(in_features=256*56*56, out_features=2)
        self.softmax = nn.Softmax()

    def forward(self, inp):
        out = self.layer1(inp)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return self.softmax(out)

model = VGGish().cuda(1)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=.01, momentum=0.9)
loss_history = train(model, criterion, optimizer, num_epochs=5)


# visualizer
# enumerate val
# wrap vals in var, cudafy
# get predictions
# plt.subplot
# imshow
plt.plot(loss_history['train'])
plt.plot(loss_history['val'])




