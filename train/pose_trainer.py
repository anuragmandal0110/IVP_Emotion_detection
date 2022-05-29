#from dataset import EmotionDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms
from torch.optim import lr_scheduler
import torch
from model import get_model
#import requests
import random
import os
from pose_dataset import PoseDataset
from pose_classifier import PoseClassification


if(not os.path.exists("../../../scratch/am12064/ivp/dataset/pose")):
    raise "DATASET NOT FOUND"


TRAIN_DATASET_PATH = "../../../scratch/am12064/ivp/dataset/pose/images/train"
TEST_DATA_PATH = "../../../scratch/am12064/ivp/dataset/pose/images/validation"
EPOCH = 100

model = PoseClassification()

trainDataset = PoseDataset(TRAIN_DATASET_PATH)
testDataset = PoseDataset(TEST_DATA_PATH)

train_data_loader = DataLoader(
    trainDataset, batch_size=50, shuffle=True, num_workers=1)
test_data_loader = DataLoader(
    testDataset, batch_size=50, shuffle=True, num_workers=1)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.6)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


dataset_sizes = {"train": trainDataset.__len__(), "val": testDataset.__len__()}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Device :- {device}")

if(torch.cuda.is_available()):
    model.cuda()

print(dataset_sizes)
# Updated model
print("Updated model")
print(model.eval())

best_acc = 0

model.aux_logits=False

for epoch in range(EPOCH):
    for phase in ['train', 'val']:
        if phase == 'train':
            loader = train_data_loader
            model.train()  # Set model to training mode
        else:
            loader = test_data_loader
            model.eval()
        running_loss = 0.0
        running_corrects = 0
        for i, (inputs, classes) in enumerate(loader):
            print(f"Epoch {epoch} Minibatch {i}/{len(loader)}")
            inputs = inputs.to(device)
            classes = classes.to(device)

            optimizer.zero_grad()
            output = model(inputs)
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(output, classes)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
            running_loss += loss.item()
            running_corrects += torch.sum(preds == classes.data)

        if phase == 'train':
            exp_lr_scheduler.step()
        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        # print statisticsx`x`
        print(
            f'Epoch: {epoch} {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        if(epoch_acc > best_acc):
            best_acc = epoch_acc
            torch.save(model.state_dict(), "./saved.pt")
