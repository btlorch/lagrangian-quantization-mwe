# Based on CIFAR-10 tutorial: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from model import Net
from data import load_stl10_dataset
import torch.optim as optim
import numpy as np
import argparse
from tqdm import tqdm
import os


def train(data_dir, batch_size, num_epochs, output_dir):

    # Transform images to [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset, testset = load_stl10_dataset(data_dir=data_dir, transform=transform)
    num_batches_per_epoch = int(np.ceil(len(trainset) / batch_size))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = Net(num_classes=2)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in tqdm(range(num_epochs), desc="Training epoch"):  # loop over the dataset multiple times
        epoch_loss = 0

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"[Epoch {epoch:03d}] Loss = {epoch_loss / num_batches_per_epoch}")

    print('Finished Training')

    ########################################################################
    # Let's quickly save our trained model:
    model_filepath = os.path.join(output_dir, "stl10_net.pth")
    torch.save(net.state_dict(), model_filepath)

    correct = 0
    total = 0
    net.eval()

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the {total} test images: {100 * correct / total:3.2f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./stl10", help="Path to STL10 directory")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--output_dir", type=str, default=".", help="Where to store trained model")
    args = vars(parser.parse_args())

    train(**args)
