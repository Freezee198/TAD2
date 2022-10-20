import numpy as np
import cv2
import torch
from monai.apps import download_and_extract
from monai.config import print_config
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121, EfficientNetBN
from monai.transforms import *
from monai.data import Dataset, DataLoader
from monai.utils import set_determinism
from sklearn.metrics import classification_report
import os.path

train_dir = '../data/train'
class_names = os.listdir(train_dir)


def main():
    trainX = np.load("../data/processed_data/trainX.npy")
    trainY = np.load("../data/processed_data/trainY.npy")
    valX = np.load("../data/processed_data/valX.npy")
    valY = np.load("../data/processed_data/valY.npy")
    testX = np.load("../data/processed_data/testX.npy")
    testY = np.load("../data/processed_data/testY.npy")

    num_class = 75

    train_transforms = Compose([
        LoadImage(image_only=True),
        Resize((-1, 1)),
        SumDimension(2),
        MyResize(),
        AddChannel(),
        ToTensor(),
    ])

    val_transforms = Compose([
        LoadImage(image_only=True),
        Resize((-1, 1)),
        SumDimension(2),
        MyResize(),
        AddChannel(),
        ToTensor(),
    ])

    train_ds = MedNISTDataset(trainX, trainY, train_transforms)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
    val_ds = MedNISTDataset(valX, valY, val_transforms)
    val_loader = DataLoader(val_ds, batch_size=16, num_workers=2)
    test_ds = MedNISTDataset(testX, testY, val_transforms)
    test_loader = DataLoader(test_ds, batch_size=16, num_workers=2)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # EfficientNetBN
    model = EfficientNetBN(
        "efficientnet-b7",
        spatial_dims=2,
        in_channels=1,
        num_classes=num_class
    ).to(device)

    if os.path.isfile('../model/best_metric_model.pth'):
        model.load_state_dict(torch.load('best_metric_model.pth'))

    # train(model, train_ds, train_loader, val_loader, num_class, device)
    test(model, test_loader, device)


def test(model, test_loader, device):
    model.eval()
    y_true = list()
    y_pred = list()

    with torch.no_grad():
        for test_data in test_loader:
            test_images, test_labels = test_data[0].to(device), test_data[1].to(device)
            pred = model(test_images).argmax(dim=1)
            for i in range(len(pred)):
                y_true.append(test_labels[i].item())
                y_pred.append(pred[i].item())

    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))


def train(model, train_ds, train_loader, val_loader, num_class, device):
    act = Activations(softmax=True)
    to_onehot = AsDiscrete(to_onehot=num_class)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)
    epoch_num = 10
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    auc_metric = ROCAUCMetric()
    metric_values = list()
    for epoch in range(epoch_num):
        print('-' * 10)
        print(f"epoch {epoch + 1}/{epoch_num}")
        model.train()
        epoch_loss = 0
        step = 0

        for batch_data in train_loader:
            step += 1
            batch_data_long = batch_data[1].type(torch.LongTensor)
            inputs, labels = batch_data[0].to(device), batch_data_long.to(device)
            optimizer.zero_grad()
            outputs = model(inputs.float())  ##### .float()
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"{step}/{len(train_ds) // train_loader.batch_size}, train_loss: {loss.item():.4f}")
            epoch_len = len(train_ds) // train_loader.batch_size

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                for val_data in val_loader:
                    val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                    y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                    y = torch.cat([y, val_labels], dim=0)

                y_onehot = [to_onehot(i) for i in y]
                y_pred_act = [act(i) for i in y_pred]
                auc_metric(y_pred_act, y_onehot)
                auc_result = auc_metric.aggregate()
                auc_metric.reset()
                del y_pred_act, y_onehot
                metric_values.append(auc_result)
                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)

                if acc_metric > best_metric:
                    best_metric = acc_metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), '../model/best_metric_model.pth')
                    print('saved new best metric model')

                print(f"current epoch: {epoch + 1} current AUC: {auc_result:.4f}"
                      f" current accuracy: {acc_metric:.4f} best AUC: {best_metric:.4f}"
                      f" at epoch: {best_metric_epoch}")


class SumDimension(Transform):
    def __init__(self, dim=1):
        self.dim = dim

    def __call__(self, inputs):
        return inputs.sum(self.dim)


class MyResize(Transform):
    def __init__(self, size=(120, 120)):
        self.size = size

    def __call__(self, inputs):
        image = cv2.resize(np.array(inputs), dsize=(self.size[1], self.size[0]), interpolation=cv2.INTER_CUBIC)
        image2 = image[20:100, 20:100]
        return image2


class MedNISTDataset(Dataset):

    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]


if __name__ == "__main__":
    main()
