import numpy as np
import torch
from monai.apps import download_and_extract
from monai.config import print_config
from monai.metrics import ROCAUCMetric
from monai.networks.nets import EfficientNetBN
from monai.transforms import *
from monai.data import DataLoader
from monai.utils import set_determinism
import os.path
import MyDataLoader

train_dir = '../data/train'
class_names = os.listdir(train_dir)


def main():
    trainX = np.load("../data/processed_data/trainX.npy")
    trainY = np.load("../data/processed_data/trainY.npy")
    valX = np.load("../data/processed_data/valX.npy")
    valY = np.load("../data/processed_data/valY.npy")

    num_class = len(class_names)

    train_transforms = Compose([
        LoadImage(image_only=True),
        Resize((-1, 1)),
        MyDataLoader.SumDimension(2),
        MyDataLoader.MyResize(),
        AddChannel(),
        ToTensor(),
    ])

    val_transforms = Compose([
        LoadImage(image_only=True),
        Resize((-1, 1)),
        MyDataLoader.SumDimension(2),
        MyDataLoader.MyResize(),
        AddChannel(),
        ToTensor(),
    ])

    train_ds = MyDataLoader.MedNISTDataset(trainX, trainY, train_transforms)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
    val_ds = MyDataLoader.MedNISTDataset(valX, valY, val_transforms)
    val_loader = DataLoader(val_ds, batch_size=16, num_workers=2)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # EfficientNetBN
    model = EfficientNetBN(
        "efficientnet-b7",
        spatial_dims=2,
        in_channels=1,
        num_classes=num_class
    ).to(device)

    if os.path.isfile('../models/best_metric_model.pth'):
        model.load_state_dict(torch.load('../models/best_metric_model.pth'))

    train(model, train_ds, train_loader, val_loader, num_class, device)


def train(model, train_ds, train_loader, val_loader, num_class, device):
    act = Activations(softmax=True)
    to_onehot = AsDiscrete(to_onehot=num_class)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)
    epoch_num = 20
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


if __name__ == "__main__":
    main()
