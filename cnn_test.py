import torch
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from cnn import CNN


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


def main():
    root = './test_data/'
    model_path = 'best_model.pth'

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
        trans_img = transform(img)
        X_test.append(trans_img)
        y_test.append(label_list[file[-5:-4]])
    X_test = torch.stack(X_test)
    y_test = torch.tensor(y_test)
    test_dataset = TensorDataset(X_test, y_test)
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
    print('Cnn predict label:', predictions)
    print('Ground truth label:', y_test.data.tolist())
    print('Test accuracy: %.2f' % accuracy)


if __name__ == "__main__":
    main()
