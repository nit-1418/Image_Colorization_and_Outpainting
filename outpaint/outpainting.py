import os
import torch
import torch.nn as nn
from collections import OrderedDict
from scipy.ndimage.morphology import distance_transform_edt
import numpy as np
import skimage.transform
import cv2
# from inference import output_size, input_size, expand_size


# Set the input size and output size
input_size = 128
output_size = 192
expand_size = (output_size - input_size) // 2

# Define the generator class
class CEGenerator(nn.Module):
    def __init__(self, channels=3, extra_upsample=False):
        super(CEGenerator, self).__init__()

        def downsample(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        def upsample(in_feat, out_feat, normalize=True):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.ReLU())
            return layers

        if not(extra_upsample):
            self.model = nn.Sequential(
                *downsample(channels, 64, normalize=False),
                *downsample(64, 64),
                *downsample(64, 128),
                *downsample(128, 256),
                *downsample(256, 512),
                nn.Conv2d(512, 4000, 1),
                *upsample(4000, 512),
                *upsample(512, 256),
                *upsample(256, 128),
                *upsample(128, 64),
                nn.Conv2d(64, channels, 3, 1, 1),
                nn.Tanh()
            )
        else:
            self.model = nn.Sequential(
                *downsample(channels, 64, normalize=False),
                *downsample(64, 64),
                *downsample(64, 128),
                *downsample(128, 256),
                *downsample(256, 512),
                nn.Conv2d(512, 4000, 1),
                *upsample(4000, 512),
                *upsample(512, 256),
                *upsample(256, 128),
                *upsample(128, 64),
                *upsample(64, 64),
                nn.Conv2d(64, channels, 3, 1, 1),
                nn.Tanh()
            )

    def forward(self, x):
        return self.model(x)

# Function to load the model
def load_model(model_path):
    model = CEGenerator(extra_upsample=True)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Remove 'module' if present
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            name = k[7:] # remove 'module'
        else:
            name = k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.cpu()
    model.eval()
    return model

# Function to construct masked input
def construct_masked(input_img):
    resized = skimage.transform.resize(input_img, (input_size, input_size), anti_aliasing=True)
    result = np.ones((output_size, output_size, 3))
    result[expand_size:-expand_size, expand_size:-expand_size, :] = resized
    return result

# Function to perform outpainting
def perform_outpaint(gen_model, input_img, blend_width=1):
    gen_model.eval()
    torch.set_grad_enabled(False)

    resized = skimage.transform.resize(input_img, (input_size, input_size), anti_aliasing=True)
    masked_img = np.ones((output_size, output_size, 3))
    masked_img[expand_size:-expand_size, expand_size:-expand_size, :] = resized

    masked_img = masked_img.transpose(2, 0, 1)
    masked_img = torch.tensor(masked_img[np.newaxis], dtype=torch.float)

    output_img = gen_model(masked_img)

    output_img = output_img.cpu().numpy()
    output_img = output_img.squeeze().transpose(1, 2, 0)
    output_img = np.clip(output_img, 0, 1)

    norm_input_img = input_img.copy().astype('float')
    if np.max(norm_input_img) > 1:
        norm_input_img /= 255
    blended_img, src_mask = blend_result(output_img, norm_input_img, blend_width)
    blended_img = np.clip(blended_img, 0, 1)

    return output_img, blended_img

# Function to blend the result
def blend_result(output_img, input_img, blend_width=64):
    in_factor = input_size / output_size
    if input_img.shape[1] < in_factor * output_img.shape[1]:
        out_width, out_height = output_img.shape[1], output_img.shape[0]
        in_width, in_height = int(out_width * in_factor), int(out_height * in_factor)
        input_img = skimage.transform.resize(input_img, (in_height, in_width), anti_aliasing=True)
    else:
        in_width, in_height = input_img.shape[1], input_img.shape[0]
        out_width, out_height = int(in_width / in_factor), int(in_height / in_factor)
        output_img = skimage.transform.resize(output_img, (out_height, out_width), anti_aliasing=True)
    
    src_mask = np.zeros((output_size, output_size))
    src_mask[expand_size+1:-expand_size-1, expand_size+1:-expand_size-1] = 1
    src_mask = distance_transform_edt(src_mask) / blend_width
    src_mask = np.minimum(src_mask, 1)
    src_mask = skimage.transform.resize(src_mask, (out_height, out_width), anti_aliasing=True)
    src_mask = np.tile(src_mask[:, :, np.newaxis], (1, 1, 3))
    
    input_pad = np.zeros((out_height, out_width, 3))
    x1 = (out_width - in_width) // 2
    y1 = (out_height - in_height) // 2
    input_pad[y1:y1+in_height, x1:x1+in_width, :] = input_img
    
    blended = input_pad * src_mask + output_img * (1 - src_mask)

#   Additionlal code for sharpening the outpainted region

        # Sharpen the outpainted region
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    outpainted_region = blended.copy()
    outpainted_region[:, :, 0] *= (1 - src_mask[:, :, 0])
    outpainted_region[:, :, 1] *= (1 - src_mask[:, :, 1])
    outpainted_region[:, :, 2] *= (1 - src_mask[:, :, 2])

    # Apply multiple sharpening techniques
    sharpened_outpainted = cv2.filter2D(outpainted_region, -1, kernel)  # Unsharp masking
    blended += sharpened_outpainted - outpainted_region
    return blended, src_mask