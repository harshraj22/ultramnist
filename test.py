import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.optim import AdamW

from PIL import Image 
from tqdm import tqdm
import random
from omegaconf import OmegaConf
import logging
import sys
from pathlib import Path
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split


from models.mobilenetv3 import mobilenetv3_small
from data_loader.data_loader import UltraMnist

from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("[%(name)s] [%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False


conf = OmegaConf.load('/kaggle/working/ultramnist/conf/config.yaml')
device = 'cuda' if torch.cuda.is_available() else 'cpu'


FINAL_IMG_SIZE = 500
val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.1307,), std = (0.3081,)),
    transforms.Resize((FINAL_IMG_SIZE, FINAL_IMG_SIZE))
])


df = pd.read_csv(conf.test_csv_path) #.drop("digit_sum", 1) # .sample(frac=0.002)
# df.set_index("digit_sum", inplace = True)
df['digit_sum'] = df.index
df = df.reindex(['id', 'digit_sum'], axis=1)

# df = df.sample(frac=0.002).reset_index(drop=True)

# ToDo: get weights of the model
model = mobilenetv3_small().to(device)
if Path(conf.model_weights_load).exists():
    logger.info(f'Loaded weights from: {conf.model_weights_load}')
    model.load_state_dict(torch.load(conf.model_weights_load, map_location=device), strict=False)
    model = model.to(device)
    
swa_model = AveragedModel(model)
if Path(conf.swa_model_weights).exists():
    # swa_model.load_state_dict(torch.load(conf.swa_model_weights, map_location=device))
    swa_model = swa_model.to(device)

print(df.head())

model.eval()
swa_model.eval()

dataset = UltraMnist(df.copy(), conf.test_image_dir, transforms=val_transforms, div_factor=1)
dl = DataLoader(dataset, batch_size=94, num_workers=2)

with torch.no_grad():
    for batch in tqdm(dl, total=len(dl)):
        img, index = batch
        # img = val_transforms(img).to(device)
        out = model(img.to(device))
        pred = torch.argmax(out, dim=1)

        df.at[index.numpy(), 'digit_sum'] = pred.cpu().numpy()

print(df.head())
df.to_csv('preds.csv', index=False)
    