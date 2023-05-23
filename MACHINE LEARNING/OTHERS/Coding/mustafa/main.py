# Importing all the necessary library files
import datetime
import torch
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from data_loader import get_train_loader, get_test_loader
from model import CIFAR10CNN, CIFAR10LNN
from trainer import train_test
from visulaizer import plot_confusion_matrix_CNN, plot_confusion_matrix_LNN, plot_train_test_accuracy_CNN, plot_train_test_accuracy_LNN, plot_loss_CNN, plot_loss_LNN

# Defining hyperparameters
learning_rate = 0.001
num_epochs = 20
batch_size = 64

# The variables with '_l' means that those variables are used to perfom calculations specifically for LNN parts


# Setting of respective dataloaders
train_loader = get_train_loader(batch_size=batch_size)
test_loader = get_test_loader(batch_size=batch_size)

# Defining CNN and lNN networks
model = CIFAR10CNN()
model_l = CIFAR10LNN()

# Training anfd Testing
train_acc, train_loss, test_acc, test_loss, true_labels, predicted_labels, train_acc_l, train_loss_l, test_acc_l, test_loss_l, predicted_labels_l = train_test(
    model, model_l, train_loader, test_loader, learning_rate, num_epochs, batch_size)

# Classes present in the dataset
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

labelling = ('plane-1', 'car-2', 'bird-3', 'cat-4', 'deer-5',
             'dog-6', 'frog-7', 'horse-8', 'ship-9', 'truck-10')

print(labelling)

# Saving results in folder
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_folder = os.path.join("00_Results", f"results_{now}")
os.makedirs(results_folder)

# Confusion matrix and other plots for both CNN and LNN
true_labels = list(map(int, true_labels))
predicted_labels = list(map(int, predicted_labels))
predicted_labels_l = list(map(int, predicted_labels_l))
cm = confusion_matrix(true_labels, predicted_labels)
cm_l = confusion_matrix(true_labels, predicted_labels_l)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_l = cm_l.astype('float') / cm_l.sum(axis=1)[:, np.newaxis]

plot_confusion_matrix_CNN(true_labels, predicted_labels, cm, classes,
                          save_path=os.path.join(results_folder, "Confusion_Matrix_CNN.png"))

plot_confusion_matrix_LNN(true_labels, predicted_labels_l, cm_l, classes,
                          save_path=os.path.join(results_folder, "Confusion_Matrix_LNN.png"))

plot_train_test_accuracy_CNN(train_acc, test_acc, save_path=os.path.join(
    results_folder, "training_vs_testing_accuracy_CNN.png"))

plot_train_test_accuracy_LNN(train_acc_l, test_acc_l, save_path=os.path.join(
    results_folder, "training_vs_testing_accuracy_LNN.png"))

plot_loss_CNN(train_loss, test_loss, save_path=os.path.join(
    results_folder, "training_vs_testing_loss_CNN.png"))

plot_loss_LNN(train_loss_l, test_loss_l, save_path=os.path.join(
    results_folder, "training_vs_testing_loss_LNN.png"))

torch.save(model.state_dict(), os.path.join(results_folder, "CNN_model.pth"))
torch.save(model_l.state_dict(), os.path.join(results_folder, "LNN_model.pth"))
