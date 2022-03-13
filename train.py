import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import AdamW

from PIL import Image 
from tqdm import tqdm
import random
from omegaconf import OmegaConf
import logging
import sys
from pathlib import Path

from models.mobilenetv3 import mobilenetv3_small
from data_loader.data_loader import UltraMnist


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


# ToDo: Add augmentations
_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((2000, 2000))
])

dataset = UltraMnist(conf.train_csv_path, conf.train_image_dir, transforms=_transforms)
num_datapoints = len(dataset)
# split into train-test
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(0.8 * num_datapoints), num_datapoints - int(0.8 * num_datapoints)])
train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers)
val_dataloader = DataLoader(val_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers)


# ToDo: get weights of the model
model = mobilenetv3_small().to(device)
if Path(conf.model_weights_load).exists():
    logger.info(f'Loaded weights from: {conf.model_weights_load}')
    model.load_state_dict(torch.load(conf.model_weights_load))
    model = model.to(device)
optimizer = AdamW(model.parameters())
# ToDo: Add an LR Schedular
criterian = nn.CrossEntropyLoss()

# print(f'Type of num_epochs: {type(conf.num_epochs)}')
best_val_accuracy = 0.0

for epoch in tqdm(range(conf.num_epochs), total=conf.num_epochs):
    for phase, ds in [('train', train_dataloader), ('val', val_dataloader)]:
        running_loss, running_corrects = 0, 0
        if phase == 'train':
            model.train()
        else:
            model.eval()

        with torch.set_grad_enabled(phase=='train'):
            for batch in tqdm(ds, desc=f'Phase: {phase}'):
                optimizer.zero_grad()
                
                imgs, labels = batch
                imgs, labels = imgs.to(device), labels.to(device)
                outs = model(imgs)
                preds = torch.argmax(outs, dim=1)
                loss = criterian(outs, labels)

                # tqdm.write(f'shapes: Input: {imgs.shape}, labels: {labels.shape}, outs: {outs.shape}, preds: {preds.shape}')
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # tqdm.write(f'preds: {preds}, labels: {labels}, matches: {(preds == labels).sum()}')

                running_loss += loss.item() * len(batch)
                running_corrects += (preds == labels).sum()
        # Add code for LR Schedular
        # Log running loss, accuracy
        tqdm.write(f'phase: [{phase}] | Loss: {running_loss:.3f} | Acc: {running_corrects / len(ds) :.3f}')


    # Currently Saving on each epoch, only the best model
    if running_corrects / len(ds) > best_val_accuracy:
        best_val_accuracy = running_corrects / len(ds)
        torch.save(model.state_dict(), conf.model_weights_save)
        tqdm.write(f'Saved weights to: {conf.model_weights_save}')
