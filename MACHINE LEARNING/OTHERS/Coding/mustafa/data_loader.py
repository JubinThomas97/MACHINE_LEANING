#importing the necessary library files
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#Defining function for training loader
def get_train_loader(batch_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],
                            [0.5,0.5,0.5])
    ])

    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader

#Defining function for testing loader
def get_test_loader(batch_size):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],
                            [0.5,0.5,0.5])
    ])

    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader
