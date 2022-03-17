import torch
import torch.nn as nn
import pandas as pd

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from itertools import product
import random

from typing import List, Any

from omegaconf import OmegaConf


def random_flip(images: List[Any]):
    if random.randint(0, 1):
        images = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in images]
        label = 1
    else:
        label = 0
    return images, label


def random_rotate(images: List[Any]):
    rotations = [(-90, 0), (0, 1), (90, 2)]
    degree, label = random.choice(rotations)

    rotated_images = [img.rotate(degree) for img in images]
    return rotated_images, label


class PreTrainingDataset(Dataset):
    def __init__(self, data, image_dir: str, transforms=None):
        super(PreTrainingDataset, self).__init__()

        self.image_dir = image_dir

        self.data = data
        self.transforms = transforms

    def __len__(self):
        # ToDo: iterate on full data
        return len(self.data) // 3000

    def __getitem__(self, id):
        image_id, label = self.data.loc[id]
        # ToDo: remove conversion to RGB, modify model
        image = Image.open(Path(f'{self.image_dir}/{image_id}.jpeg')) #.convert('RGB')
        w, h = image.size
        d = 2000 # num of splits per width and height
        cropped_images = []

        # Crop image into 4 equal parts
        grid = product(range(0, h-h%d, d), range(0, w-w%d, d))
        for i, j in grid:
            box = (j, i, j+d, i+d)
            cropped_images.append(image.crop(box))

        rotate_label, flip_label = 0, 0
        # Randomly crop or rotate
        if random.randint(0, 10) <= 6:
            # 60% of the time, rotate
            cropped_images, rotate_label = random_rotate(cropped_images)
        else:
            cropped_images, flip_label = random_flip(cropped_images)

        if self.transforms:
            cropped_images = [self.transforms(image) for image in cropped_images]
        return torch.stack(cropped_images), torch.tensor(rotate_label), torch.tensor(flip_label)


if __name__ == '__main__':
    conf = OmegaConf.load('../conf/config.yaml')
    _transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    ultramnist_dataset = PreTrainingDataset(pd.read_csv(conf.train_csv_path), conf.train_image_dir, transforms=_transforms)
    n = len(ultramnist_dataset)
    for i in tqdm(range(n), total=n):
        data_point = ultramnist_dataset[i]
        tqdm.write(f'{data_point[0].shape}, {data_point[1].shape}, {data_point[2].shape}, {data_point[1].item()},  {data_point[2].item()}')