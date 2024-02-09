import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T 

from torchvision.transforms.functional import crop

from .constants import *

# Method for tactile augmentations in BYOL
def get_tactile_augmentations(img_means, img_stds, img_size):
    tactile_aug = T.Compose([
        T.RandomApply(
            nn.ModuleList([T.RandomResizedCrop(img_size, scale=(.9, 1))]),
            p = 0.5
        ), 
        T.RandomApply(
            nn.ModuleList([T.GaussianBlur((3, 3), (1.0, 2.0))]), 
            p = 0.5
        ),
        T.Normalize(
            mean = img_means,
            std = img_stds
        )
    ])
    return tactile_aug

def get_vision_augmentations(img_means, img_stds):
    color_aug = T.Compose([
        T.RandomApply(
            nn.ModuleList([T.ColorJitter(.8,.8,.8,.2)]), 
            p = 0.2
        ),
        T.RandomGrayscale(p=0.2), 
        T.RandomApply(
            nn.ModuleList([T.GaussianBlur((3, 3), (1.0, 2.0))]), 
            p = 0.2
        ),
        T.Normalize(
            mean = img_means,
            std =  img_stds
        )
    ])

    return color_aug 

# Vision transforms used
def crop_transform(image, camera_view=0): # This is significant to the setup
    if camera_view == 0:
        return crop(image, 0,0,480,480)
    elif camera_view == 1:
        return crop(image, 0,90,480,480)
    
def get_inverse_image_norm():
    np_means = np.asarray(VISION_IMAGE_MEANS)
    np_stds = np.asarray(VISION_IMAGE_STDS)

    inv_normalization_transform = T.Compose([
        T.Normalize(mean = [0,0,0], std = 1 / np_stds ), 
        T.Normalize(mean = -np_means, std = [1,1,1])
    ])

    return inv_normalization_transform

# Tactile transforms used
def tactile_scale_transform(image):
    image = (image - TACTILE_PLAY_DATA_CLAMP_MIN) / (TACTILE_PLAY_DATA_CLAMP_MAX - TACTILE_PLAY_DATA_CLAMP_MIN)
    return image

def tactile_clamp_transform(image):
    image = torch.clamp(image, min=TACTILE_PLAY_DATA_CLAMP_MIN, max=TACTILE_PLAY_DATA_CLAMP_MAX)
    return image