import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
try:
    from tqdm import tqdm
except ImportError:
    print('tqdm could not be imported. If you want to use progress bar during training,'
          'install tqdm from https://github.com/tqdm/tqdm.')


class ChineseHandWriteDataset(Dataset):
    def __init__(self, root="", label_dic={}, transform=None, resize=True, resize_size=128):
        self.img_file = []
        self.labelfile = []
        self.transform = transform
        self.root = root
        self.resize = resize
        self.resize_size = resize_size
        self.label_dic = label_dic
        for _, file in enumerate(os.listdir(root)):
            self.img_file.append(root + '/' + file)

    def __getitem__(self, index):
        img_path = self.img_file[index]
        img = Image.open(img_path).convert('L')
        label_chinese = img_path[-5:-4]
        if label_chinese in self.label_dic:
            label = self.label_dic[label_chinese]
        if self.resize:
            img = img.resize((self.resize_size, self.resize_size))

        return self.transform(img), label

    def __len__(self):
        return len(self.img_file)


# Create CNN Model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=48, kernel_size=(3,3), padding=(1,1))
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=(3,3), padding=(1,1))
        self.conv3 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=(3,3), padding=(1,1))
        self.conv4 = nn.Conv2d(in_channels=192, out_channels=256, kernel_size=(3,3), padding=(1,1))
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(in_features=16*16*256, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=64)
        self.Dropout = nn.Dropout(0.25)
        self.fc3 = nn.Linear(in_features=64, out_features=50)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.Dropout(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.Dropout(x)
        x = x.view(-1, 16*16*256)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.Dropout(x)
        x = self.fc3(x)
        return x


def fit_model(model, loss_func, optimizer, num_epochs, train_loader, test_loader, device):
    # Traning the Model
    # history-like list for store loss & acc value
    training_loss = []
    training_accuracy = []
    validation_loss = []
    validation_accuracy = []
    best_acc = 0
    for epoch in range(num_epochs):
        # training model & store loss & acc / epoch
        correct_train = 0
        total_train = 0
        train_bar = tqdm(train_loader)
        model.train()
        for images, labels in train_bar:
            # 1.Define variables
            images = images.to(device)
            labels = labels.to(device)
            # 2.Clear gradients
            optimizer.zero_grad()
            # 3.Forward propagation
            outputs = model(images)
            # 4.Calculate softmax and cross entropy loss
            train_loss = loss_func(outputs, labels)
            # 5.Calculate gradients
            train_loss.backward()
            # 6.Update parameters
            optimizer.step()
            # 7.Get predictions from the maximum value
            predicted = torch.max(outputs.data, 1)[1]
            # 8.Total number of labels
            total_train += len(labels)
            # 9.Total correct predictions
            correct_train += (predicted == labels).float().sum()
            train_bar.set_description(desc='[%d/%d] | Train Loss:%.4f' %
                                           (epoch + 1, num_epochs, train_loss.item()))
        # 10.store val_acc / epoch
        train_accuracy = 100 * correct_train / float(total_train)
        training_accuracy.append(train_accuracy)
        # 11.store loss / epoch
        training_loss.append(train_loss.data)

        # evaluate model & store loss & acc / epoch
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            model.eval()
            val_bar = tqdm(test_loader)
            for images, labels in val_bar:
                # 1.Define variables
                images = images.to(device)
                labels = labels.to(device)
                # 2.Forward propagation
                outputs = model(images)
                # 3.Calculate softmax and cross entropy loss
                val_loss = loss_func(outputs, labels)
                # 4.Get predictions from the maximum value
                predicted = torch.max(outputs.data, 1)[1]
                val_bar.set_description(desc='[%d/%d] | Validation Loss:%.4f' % (epoch + 1, num_epochs, val_loss.item()))
                # 5.Total number of labels
                total_test += len(labels)
                # 6.Total correct predictions
                correct_test += (predicted == labels).float().sum()
        # 6.store val_acc / epoch
        val_accuracy = 100 * correct_test / float(total_test)
        best_acc = max(best_acc, val_accuracy)
        validation_accuracy.append(val_accuracy)
        # 11.store val_loss / epoch
        validation_loss.append(val_loss.data)
        print('Train Epoch: {}/{} Traing_Loss: {:.4f} Traing_acc: {:.2f}% Val_Loss: {:.4f} Val_accuracy: {:.2f} '
              'Best Val_accuracy: {:.2f}%'.format(epoch+1, num_epochs, train_loss.data, train_accuracy,
                                                val_loss.data, val_accuracy, best_acc))

    return training_loss, training_accuracy, validation_loss, validation_accuracy


def main():
    root = './k_mean_data/'

    label_list = {}
    f = open('training data dic.txt', 'r', encoding="utf-8")
    for idx, line in enumerate(f.readlines()):
        if idx == 50:
            break
        label_list[line[0]] = idx

    train_dataset = []
    valid_dataset = []
    transform = transforms.ToTensor()
    for idx, dir_ in enumerate(os.listdir(root)):
        dataset = ChineseHandWriteDataset(root=root + dir_, label_dic=label_list, transform=transform,
                                          resize=True,
                                          resize_size=64)
        train_set_size = int(len(dataset) * 0.8)
        valid_set_size = len(dataset) - train_set_size
        train_set, valid_set = random_split(dataset, [train_set_size, valid_set_size], torch.Generator().manual_seed(0))
        train_dataset.append(train_set)
        valid_dataset.append(valid_set)

    train_dataset = ConcatDataset(train_dataset)
    valid_dataset = ConcatDataset(valid_dataset)

    # Hyper Parameters
    # batch_size, epoch and iteration
    LR = 0.001
    batch_size = 8
    valid_batch_size = 8
    n_iters = 10000
    epochs = n_iters / (len(train_dataset) / batch_size)
    epochs = int(epochs)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1)

    valid_dataloader = DataLoader(
        valid_dataset, batch_size=valid_batch_size, pin_memory=True, num_workers=1)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
        print('Warning! Using CPU.')

    cnn = ConvNet()
    cnn.to(device)
    print(cnn)
    summary(cnn, (1, 64, 64))
    opt = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
    ce = nn.CrossEntropyLoss()  # the target label is not one-hotted

    training_loss, training_accuracy, validation_loss, validation_accuracy = fit_model(cnn, ce, opt,
                                                                                       epochs, train_dataloader,
                                                                                       valid_dataloader, device)

    # CNN Classifier report and analysis
    # visualization
    plt.plot(range(epochs), training_loss, 'b-', label='Training_loss')
    plt.plot(range(epochs), validation_loss, 'g-', label='validation_loss')
    plt.title('Training & Validation loss')
    plt.xlabel('Number of epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.plot(range(epochs), training_accuracy, 'b-', label='Training_accuracy')
    plt.plot(range(epochs), validation_accuracy, 'g-', label='Validation_accuracy')
    plt.title('Training & Validation accuracy')
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
