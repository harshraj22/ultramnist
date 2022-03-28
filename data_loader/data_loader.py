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
    def __init__(self, data, image_dir: str, transforms=None, div_factor=1):
        super(UltraMnist, self).__init__()

        self.image_dir = image_dir

        self.data = data
        self.transforms = transforms
        self.div_factor = div_factor

    def __len__(self):
        return len(self.data) // self.div_factor

    def __getitem__(self, id):
        image_id, label = self.data.loc[id]
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