import torch
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from cnn import CNN, CNN_1d
import numpy as np


def evaluation(model, device, dataloader, len_data):
    predictions = []
    correct_test = 0
    with torch.no_grad():
        model.eval()
        for img, label in dataloader:
            img = img.to(device)
            label = label.to(device)
            outputs = model(img)
            predicted = torch.max(outputs.data, 1)[1]
            predictions.append(predicted.item())
            correct_test += (predicted == label).float().sum()

    return predictions, correct_test / len_data


def image_show_test_data(root, y_t, y_pred):
    fig = plt.figure()
    axes = []
    row = int(len(y_t) / 5)
    col = 5
    # fig.set_size_inches(12, 14)
    for i, file in enumerate(os.listdir(root)):
        img_path = root + file
        img = Image.open(img_path)
        img = img.resize((64, 64))
        axes.append(fig.add_subplot(row, col, i+1))
        if y_t[i] == y_pred[i]:
            axes[-1].set_title('label= %d, predict= %d' % (y_t[i], y_pred[i]), fontsize=10)
        else:
            axes[-1].set_title('label= %d, predict= %d' % (y_t[i], y_pred[i]), fontsize=10, color='red')
        a = np.asarray(img)
        plt.imshow(a)
        plt.axis('off')
    plt.show()


def main():
    root = './test_data/'
    is_1d = True
    if is_1d:
        model_path = 'best_model_CNN1D.pth'
        # model_path = 'final_model_CNN1D.pth'
    else:
        model_path = 'best_model_CNN.pth'
        # model_path = 'final_model.pth'
    label_list = {}
    f = open('training data dic.txt', 'r', encoding="utf-8")
    for idx, line in enumerate(f.readlines()):
        if idx == 50:
            break
        label_list[line[0]] = idx

    X_test = []
    y_test = []
    transform = transforms.ToTensor()
    for _, file in enumerate(os.listdir(root)):
        img_path = root + file
        img = Image.open(img_path).convert('L')
        img = img.resize((64, 64))
        if is_1d:
            trans_img = transform(img).view(1, 64*64)
        else:
            trans_img = transform(img)
        X_test.append(trans_img)
        y_test.append(label_list[file[-5:-4]])
    X_test_tensor = torch.stack(X_test)
    y_test_tensor = torch.tensor(y_test)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=1)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
        print('Warning! Using CPU.')

    model = None
    try:
        model = torch.load(model_path)

    except:
        print('Can not find the model file!')
        exit()

    model.to(device)

    predictions, accuracy = evaluation(model, device, test_dataloader, len(y_test))
    chinese_label = list(label_list.keys())
    for i in range(len(predictions)):
        print('Cnn predict label:', chinese_label[predictions[i]])
        print('Ground truth label:', chinese_label[y_test[i]])
        print()

    print('Test accuracy: %.2f' % accuracy)

    image_show_test_data(root, y_test, predictions)


if __name__ == "__main__":
    main()
