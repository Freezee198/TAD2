import numpy as np
import MyDataLoader
import torch
from monai.transforms import *
from monai.data import DataLoader
from monai.networks.nets import EfficientNetBN
from monai.utils import set_determinism
from sklearn.metrics import classification_report
import os.path

test_dir = '../data/test'
class_names = os.listdir(test_dir)


def main():
    testX = np.load("../data/processed_data/testX.npy")
    testY = np.load("../data/processed_data/testY.npy")

    num_class = len(class_names)

    val_transforms = Compose([
        LoadImage(image_only=True),
        Resize((-1, 1)),
        MyDataLoader.SumDimension(2),
        MyDataLoader.MyResize(),
        AddChannel(),
        ToTensor(),
    ])

    test_ds = MyDataLoader.MedNISTDataset(testX, testY, val_transforms)
    test_loader = DataLoader(test_ds, batch_size=16, num_workers=2)

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


if __name__ == "__main__":
    main()
