#Importing all the necessary library files
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

#Defining a function for both training and testing of the model
def train_test(model, model_l, train_loader, test_loader, learning_rate, num_epochs, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model_l = model_l.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer_l = optim.Adam(model_l.parameters(), lr=learning_rate)
    train_loss_list, train_acc_list = [], []
    train_loss_list_l, train_acc_list_l = [], []
    test_loss_list, test_acc_list = [], []
    test_loss_list_l, test_acc_list_l = [], []
    
    # Lists for confusion matrix
    true_labels = []
    predicted_labels = []
    predicted_labels_l = []

    for epoch in range(num_epochs):
        model.train()
        model_l.train()
        train_loss, train_correct = 0, 0
        train_loss_l, train_correct_l = 0, 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            optimizer_l.zero_grad()
            output = model(data)
            output_l = model_l(data)
            loss = criterion(output, target)
            loss_l = criterion(output_l, target)
            loss.backward()
            loss_l.backward()
            optimizer.step()
            optimizer_l.step()
            train_loss += loss.item()
            train_loss_l += loss_l.item()
            pred = output.argmax(dim=1, keepdim=True)
            pred_l = output_l.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()
            train_correct_l += pred_l.eq(target.view_as(pred_l)).sum().item()

        train_loss /= len(train_loader.dataset)
        train_loss_l /= len(train_loader.dataset)
        train_acc = 100. * train_correct / len(train_loader.dataset)
        train_acc_l = 100. * train_correct_l / len(train_loader.dataset)
        train_loss_list.append(train_loss)
        train_loss_list_l.append(train_loss_l)
        train_acc_list.append(train_acc)
        train_acc_list_l.append(train_acc_l)

        model.eval()
        model_l.eval()
        test_loss, test_correct = 0, 0
        test_loss_l, test_correct_l = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                output_l = model_l(data)
                test_loss += criterion(output, target).item()
                test_loss_l += criterion(output_l, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                pred_l = output_l.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(target.view_as(pred)).sum().item()
                test_correct_l += pred_l.eq(target.view_as(pred_l)).sum().item()

            # Store the true labels and predicted labels for the last epoch
            if epoch == num_epochs - 1:
                # Iterate over the test data
                for images, labels in test_loader:
                    if torch.cuda.is_available():
                        images = Variable(images.cuda())
                        labels = Variable(labels.cuda())
                    # Get the predicted labels for the images
                    outputs = model(images)
                    outputs_l = model_l(images)
                    _, predicted = torch.max(outputs.data, 1)
                    _, predicted_l = torch.max(outputs_l.data, 1)

                    # Convert the labels and predicted labels to numpy arrays
                    true_labels.extend(labels.cpu().numpy())
                    predicted_labels.extend(predicted.cpu().numpy())
                    predicted_labels_l.extend(predicted_l.cpu().numpy())

        test_loss /= len(test_loader.dataset)
        test_loss_l /= len(test_loader.dataset)
        test_acc = 100. * test_correct / len(test_loader.dataset)
        test_acc_l = 100. * test_correct_l / len(test_loader.dataset)
        test_loss_list.append(test_loss)
        test_loss_list_l.append(test_loss_l)
        test_acc_list.append(test_acc)
        test_acc_list_l.append(test_acc_l)

        print('Epoch [{}/{}], Train Loss CNN: {:.4f}, Train Acc CNN: {:.2f}%, Test Loss CNN: {:.4f}, Test Acc CNN: {:.2f}%'
              .format(epoch+1, num_epochs, train_loss, train_acc, test_loss, test_acc))
        
        print('Epoch [{}/{}], Train Loss LNN: {:.4f}, Train Acc LNN: {:.2f}%, Test Loss LNN: {:.4f}, Test Acc LNN: {:.2f}%'
              .format(epoch+1, num_epochs, train_loss_l, train_acc_l, test_loss_l, test_acc_l))

    return train_acc_list, train_loss_list, test_acc_list, test_loss_list, true_labels, predicted_labels, train_acc_list_l, train_loss_list_l, test_acc_list_l, test_loss_list_l, predicted_labels_l
