import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from torchsummary import summary
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR
from torch_lr_finder import LRFinder
from models import *


import argparse
from utils import *
from CIFAR10_dataloader import *

def get_args():
    parser = argparse.ArgumentParser(description='CIFAR10 Training using ResNet')
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--model_name', default="resnet18", type=str)
    # parser.add_argument('--optimizer', default="SGD", type=str)
    # parser.add_argument('--scheduler', default=None, type=str)
    # parser.add_argument('--loss_type', default='cross_entropy', type=str)
    args = parser.parse_args()
    return args

def get_model(model_name, device):
    if model_name=='resnet18':
        return ResNet18().to(device)
    elif model_name=='resnet34':
        return ResNet34().to(device)
        

def train(model, device, train_loader, optimizer, criterion, train_acc, train_losses):
    model.train()
    pbar = tqdm(train_loader)

    train_loss = 0
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        # Get a batch of samples
        data, target = data.to(device) , target.to(device)
        # Initilizes the gradients to zero
        optimizer.zero_grad()

        # Predict
        pred = model(data)

        # Calculate Loss
        loss = criterion(pred, target)
        train_loss+=loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

        correct += GetCorrectPredCount(pred, target)
        processed += len(data)

        pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

    train_acc.append(100*correct/processed)
    train_losses.append(train_loss/len(train_loader))
    return train_acc, train_losses

def test(model, device, test_loader, criterion, test_acc, test_losses):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            test_loss += criterion(output, target, reduction='sum').item()  # Sum up batch loss
            correct += GetCorrectPredCount(output, target)

    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_acc, test_losses

def main():
    # Get arguments
    args = get_args()
    
    # View images in dataset
    train_data, test_data = get_CIFAR10_dataset()
    
    show_sample_images(train_data, "Train Data")
    show_sample_images(test_data, "Test Data")
    
    # Analyse dataset to get mean and std dev
    analyse_dataset()

    # Set the seed for reproducible results
    SEED = 1
    set_manualSeed(SEED)
    
    # Dataloader
    dataloader_args = dict(shuffle=True, batch_size=args.batch_size, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
    train_loader, test_loader = get_CFAR10_data_loaders(train_data, test_data)


    # Show images after augmentation
    show_augmented_images(train_loader)

    # Initialize model
    device = selectDevice()
    model = get_model("resnet18",device)
    summary(model, input_size=(3, 32, 32))

    num_epoch = 20
    pref_start_LR = 3e-2
    pref_weight_decay = 1e-5

    # Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=pref_start_LR, weight_decay=pref_weight_decay)

    # LR Finder
    lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    lr_finder.range_test(train_loader, end_lr=100, num_iter=200, step_mode="exp")
    lr_finder.plot()
    lr_finder.reset()

    suggested_lr = 8.63E-02
    
    # Train and Test model
    # Data to plot accuracy and loss graphs
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []

    test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=suggested_lr, epochs=num_epoch, steps_per_epoch=len(train_loader), \
                                            pct_start=5/num_epoch, div_factor=100)


    for epoch in range(num_epoch):
        print('Epoch : ',epoch)
        train_acc, train_losses = train(model, device, train_loader, optimizer, criterion, train_acc, train_losses)
        test_acc, test_losses = test(model, device, test_loader, criterion, test_acc, test_losses)
        scheduler.step()
    
    # Analyse results
    viewAnalysis(train_losses, train_acc, test_losses, test_acc)
    show_misclassified_images(device, model, test_loader)
    show_gradcam_images(device, model, test_loader, [model.layer3[-1]])




if __name__ == "__main__":
    main()