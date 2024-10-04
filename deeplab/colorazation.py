import torch
# from .model import MainModel as model
import torch
import torchvision
import os
import shutil
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import time
import glob
from torch.utils.data import Dataset, DataLoader
from skimage.color import rgb2lab, lab2rgb
from torch import nn, optim
from torchvision import transforms
import torch.nn.functional as F
ImageFile.LOAD_TRUNCATED_IMAGES = True

INPUT_SHAPE = 256


def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """

    L = (L + 1.0) * 50.0
    ab = ab * 110.0
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)

def load_model(model_class, file_path):
    """
    Load PyTorch model from file.

    Args:
        model_class (torch.nn.Module): PyTorch model class to load.
        file_path (str): File path to load the model from.

    Returns:
        model (torch.nn.Module): Loaded PyTorch model.
    """
    model = model_class()
    model.load_state_dict(torch.load(file_path))
    return model



def create_lab_tensors(image):
    """
    This function receives an image path or a direct image input and creates a dictionary of L and ab tensors.
    Args:
    - image: either a path to the image file or a direct image input.
    Returns:
    - lab_dict: dictionary containing the L and ab tensors.
    """
    if isinstance(image, str):
        # Open the image and convert it to RGB format
        img = Image.open(image).convert("RGB")
    else:
        img = image.convert("RGB")

    custom_transforms = transforms.Compose(
        [
            transforms.Resize((INPUT_SHAPE, INPUT_SHAPE), Image.BICUBIC),
            transforms.RandomHorizontalFlip(),  # A little data augmentation!
        ]
    )
    img = custom_transforms(img)
    img = np.array(img)
    img_lab = rgb2lab(img).astype("float32")  # Converting RGB to L*a*b
    img_lab = transforms.ToTensor()(img_lab)
    L = img_lab[[0], ...] / 50.0 - 1.0  # Between -1 and 1
    L = L.unsqueeze(0)
    ab = img_lab[[1, 2], ...] / 110.0  # Between -1 and 1
    return {"L": L, "ab": ab}



def predict_color(model, image, save=False):
    """
    This function receives an image path or a direct image input and creates a dictionary of L and ab tensors.
    Args:
    - model : Pytorch Gray Scale to Colorization Model
    - image: either a path to the image file or a direct image input.
    """
    data = create_lab_tensors(image)
    # predict_and_visualize_single_image(model, data, save)
    genimg = predict_and_return_image(model, data)
    return genimg


def predict_and_return_image(model, data):
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    fake_color = model.fake_color.detach()
    L = model.L
    fake_imgs = lab_to_rgb(L, fake_color)
    return fake_imgs[0]