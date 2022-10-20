import numpy as np
import MyData
import torch
from monai.transforms import *
from monai.data import DataLoader
from monai.networks.nets import EfficientNetBN
from monai.utils import set_determinism
from sklearn.metrics import classification_report
import os.path
import sys
import json
import yaml
import pickle


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython test.py data model report\n")
        sys.exit(1)

    model_name = params['model']
    data_folder = sys.argv[1]
    model_path = "models/" + sys.argv[2]
    testX = np.load(data_folder + "/testX.npy")
    testY = np.load(data_folder + "/testY.npy")

    num_class = params['class_number']

    val_transforms = Compose([
        LoadImage(image_only=True),
        Resize((-1, 1)),
        MyData.SumDimension(2),
        MyData.MyResize(),
        AddChannel(),
        ToTensor(),
    ])

    test_ds = MyData.MedNISTDataset(testX, testY, val_transforms)
    test_loader = DataLoader(test_ds, batch_size=16, num_workers=2)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # EfficientNetBN
    model = EfficientNetBN(
        model_name,
        spatial_dims=2,
        in_channels=1,
        num_classes=num_class
    ).to(device)

    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path))

    test(model, test_loader, device)


def test(model, test_loader, device):
    model.eval()
    y_true = list()
    y_pred = list()
    with open(sys.argv[1] + "/CLASS NAMES", "rb") as fp:
        class_names = pickle.load(fp)

    with torch.no_grad():
        for test_data in test_loader:
            test_images, test_labels = test_data[0].to(device), test_data[1].to(device)
            pred = model(test_images).argmax(dim=1)
            for i in range(len(pred)):
                y_true.append(test_labels[i].item())
                y_pred.append(pred[i].item())

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, output_dict=True)
    report_file = os.path.join("classification_report.json")
    with open(report_file, "w") as fd:
        json.dump(report, fd)
    score_file = os.path.join("scores.json")
    with open(score_file, "w") as fd:
        json.dump({"accuracy": report['accuracy']}, fd)


if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))["test"]
    main()
