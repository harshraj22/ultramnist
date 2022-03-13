# following tutorial from captum: https://captum.ai/tutorials/Resnet_TorchVision_Interpret

import torch
import torch.nn as nn
from torchvision import transforms

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

from omegaconf import OmegaConf
from PIL import Image 
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from pathlib import Path

import os
import sys
sys.path.append('..')
from models.mobilenetv3 import mobilenetv3_small


def get_exp_dir() -> str:
    """Returns the directory where to save the current atrifacts.
    Tries to create a new directory each time it is called.

    Returns:
        Name of new directory created to be used.
    """
    dirs = os.listdir()
    dirs = list(filter(lambda x: x.startswith('exp'), dirs))
    dirs = [int(dir[3:]) for dir in dirs] + [0]
    dir_name = f'exp{max(dirs)+1}'
    Path(dir_name).mkdir(parents=True, exist_ok=True)
    return dir_name


def visualize_integrated_gradients(image_path: str, _transforms: transforms.Compose, model: nn.Module, n_steps:int = 10):
    """Runs the integrated gradients algorithm on the given image using the passed model.

    Parameters
    ----------
    image_path : str
        The path of input image to run the integrated gradients algorithm on.
    _transforms : transforms.Compose
        Transforms to be applied to the image before passing into the model.
    model : nn.Module
        The model on which to run the integrated gradients algorithm.
    n_steps : int, optional
        The number of steps to be used in the integrated gradients algorithm, by default 10
    """

    model.eval()
    pil_img = Image.open(image_path)    
    img = _transforms(pil_img)

    out = model(img.unsqueeze(0))
    pred = torch.argmax(out.squeeze(0)).item()
    print(f'Model predicted: {pred}')

    integrated_grads = IntegratedGradients(model)

    # add extra dimention to image
    attributions_ig = integrated_grads.attribute(img.unsqueeze(0), target=pred, n_steps=n_steps)

    default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                    [(0, '#ffffff'),
                                                    (0.25, '#000000'),
                                                    (1, '#000000')], N=256)

    result = viz.visualize_image_attr(np.transpose(attributions_ig.squeeze(0).cpu().detach().numpy(), (1,2,0)),
                                np.transpose(img.cpu().detach().numpy(), (1,2,0)),
                                method='heat_map',
                                cmap=default_cmap,
                                show_colorbar=True,
                                sign='positive',
                                outlier_perc=1)

    dir_name = get_exp_dir()

    result[1].set_title(f'Integrated Grad | Model Pred: {pred}')

    result[0].savefig(Path(f'{dir_name}/result1.jpg'))
    result[1].figure.savefig(Path(f'{dir_name}/result2.jpg'))
    pil_img.save(Path(f'{dir_name}/input.jpg'))

    # print(type(result[0]), type(result[1]))
    print(f'Artifacts saved to {dir_name}')


if __name__ == '__main__':

    cfgs = OmegaConf.load('/kaggle/working/ultramnist/conf/config.yaml')

    IMAGE_PATH = '/kaggle/input/ultra-mnist/train/zzvjljhlij.jpeg'
    model = mobilenetv3_small()
    model.load_state_dict(torch.load(cfgs.model_weights_load, map_location=torch.device('cpu')))

    _transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((2000, 2000))
    ])

    visualize_integrated_gradients(IMAGE_PATH, _transforms, model)


