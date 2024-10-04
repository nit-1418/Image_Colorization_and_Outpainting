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
from torchvision.utils import make_grid
import torch.nn.functional as F
ImageFile.LOAD_TRUNCATED_IMAGES = True
# import deeplabmodel as deeplabmodel
from unet_scratch import UNETmodel
# import UNETmodel as UNETmodel

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


def init_weights(net, init="norm", gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and "Conv" in classname:
            if init == "norm":
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")

            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif "BatchNorm2d" in classname:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)
    print(f"model initialized with {init} initialization")
    return net


def init_model(model, device):
    model = model.to(device)
    model = init_weights(model)
    return model

class GANLoss(nn.Module):
    def __init__(self, gan_mode="vanilla", real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer("real_label", torch.tensor(real_label))
        self.register_buffer("fake_label", torch.tensor(fake_label))
        if gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == "lsgan":
            self.loss = nn.MSELoss()

    def get_labels(self, preds, target_is_real):
        if target_is_real:
            labels = self.real_label
        else:
            labels = self.fake_label
        return labels.expand_as(preds)

    def __call__(self, preds, target_is_real):
        labels = self.get_labels(preds, target_is_real)
        loss = self.loss(preds, labels)
        return loss
    
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


def predict_and_visualize_single_image(model, data, save=True):
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    fake_color = model.fake_color.detach()
    L = model.L
    fake_imgs = lab_to_rgb(L, fake_color)
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(L[0][0].cpu(), cmap="gray")
    axs[0].set_title("Grey Image")
    axs[0].axis("off")

    axs[1].imshow(fake_imgs[0])
    axs[1].set_title("Colored Image")
    axs[1].axis("off")
    plt.show()
    if save:
        fig.savefig(f"colorization_{time.time()}.png")


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


# def predict_and_return_image(image):
#     data = create_lab_tensors(image)
#     deeplabmodel.net_G.eval()
#     with torch.no_grad():
#         deeplabmodel.setup_input(data)
#         deeplabmodel.forward()
#     fake_color = deeplabmodel.fake_color.detach()
#     L = deeplabmodel.L
#     fake_imgs = lab_to_rgb(L, fake_color)
#     return fake_imgs[0]

def predict_and_return_image(model, data):
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    fake_color = model.fake_color.detach()
    L = model.L
    fake_imgs = lab_to_rgb(L, fake_color)
    return fake_imgs[0]