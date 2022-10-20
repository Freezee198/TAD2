import numpy as np
import cv2
from monai.transforms import *
from monai.data import Dataset


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
