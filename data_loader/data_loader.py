import torch
import torch.nn as nn
import pandas as pd

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from omegaconf import OmegaConf


class UltraMnist(Dataset):
    def __init__(self, data, image_dir: str, transforms=None):
        super(UltraMnist, self).__init__()

        self.csv_path = csv_path
        self.image_dir = image_dir

        self.data = data
        self.transforms = transforms

    def __len__(self):
        # ToDo: iterate on full data
        return len(self.data) // 10

    def __getitem__(self, id):
        image_id, label = self.data.loc[id]
        # ToDo: remove conversion to RGB, modify model
        image = Image.open(Path(f'{self.image_dir}/{image_id}.jpeg')) #.convert('RGB')
        if self.transforms:
            image = self.transforms(image)
        return image, torch.tensor(label)


if __name__ == '__main__':
    conf = OmegaConf.load('../conf/config.yaml')
    _transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    ultramnist_dataset = UltraMnist(conf.train_csv_path, conf.train_image_dir, transforms=_transforms)
    n = len(ultramnist_dataset)
    for i in tqdm(range(n), total=n):
        data_point = ultramnist_dataset[i]