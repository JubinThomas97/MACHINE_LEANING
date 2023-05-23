#Importing all the necessary library files
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

#Defining a function for Confusion matrix for CNN
def plot_confusion_matrix_CNN(true_lables, predicted_labels, cm, classes, save_path=None):
    sns.heatmap(cm, annot=True, fmt='.0%',
                cbar_kws={'format': mtick.PercentFormatter(
                    xmax=1.0, decimals=0)},
                xticklabels=[i+1 for i in range(10)],
                yticklabels=[i+1 for i in range(10)])
    plt.title('Confusion matrix CNN')
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

#Defining a function for Confusion matrix for LNN
def plot_confusion_matrix_LNN(true_lables, predicted_labels_l, cm_l, classes, save_path=None):
    sns.heatmap(cm_l, annot=True, fmt='.0%',
                cbar_kws={'format': mtick.PercentFormatter(
                    xmax=1.0, decimals=0)},
                xticklabels=[i+1 for i in range(10)],
                yticklabels=[i+1 for i in range(10)])
    plt.title('Confusion matrix LNN')
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


#Defining a function for plotting of graphs of training and testing accuracies for CNN
def plot_train_test_accuracy_CNN(train_acc, test_acc, save_path=None):
    plt.figure(figsize=(10, 7))
    plt.plot(train_acc, label='Training accuracy CNN')
    plt.plot(test_acc, label='Test accuracy CNN')
    plt.title('Training and test accuracy over epochs CNN')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

#Defining a function for plotting of graphs of trainig and testing accuracies for LNN
def plot_train_test_accuracy_LNN(train_acc_l, test_acc_l, save_path=None):
    plt.figure(figsize=(10, 7))
    plt.plot(train_acc_l, label='Training accuracy LNN')
    plt.plot(test_acc_l, label='Test accuracy LNN')
    plt.title('Training and test accuracy over epochs LNN')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

#Defining a function for plotting of graphs of loss values for CNN
def plot_loss_CNN(train_loss, test_loss, save_path=None):
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, label='Training loss')
    plt.plot(test_loss, label='Test loss')
    plt.title('Training and test loss over epochs CNN')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

#Defining a function for plotting of graphs of loss values for LNN
def plot_loss_LNN(train_loss_l, test_loss_l, save_path=None):
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss_l, label='Training loss LNN')
    plt.plot(test_loss_l, label='Test loss LNN')
    plt.title('Training and test loss over epochs LNN')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
