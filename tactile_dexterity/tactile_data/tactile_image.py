import random
import torch

import torch.nn.functional as F
import torchvision.transforms as T

from copy import deepcopy as copy

from tactile_dexterity.utils import tactile_clamp_transform, tactile_scale_transform

# Class to retrieve tactile images depending on the type

class TactileImage:
    def __init__(
        self,
        tactile_image_size=224,
        shuffle_type=None
    ):
        self.shuffle_type = shuffle_type
        self.size = tactile_image_size

        self.transform = T.Compose([
            T.Resize(tactile_image_size),
            T.Lambda(tactile_clamp_transform),
            T.Lambda(tactile_scale_transform)
        ])

    def get(self, type, tactile_values):
        if type == 'whole_hand':
            return self.get_whole_hand_tactile_image(tactile_values)
        if type == 'single_sensor':
            return self.get_single_tactile_image(tactile_values)
        if type == 'stacked':
            return self.get_stacked_tactile_image(tactile_values)

    def get_stacked_tactile_image(self, tactile_values):
        tactile_image = torch.FloatTensor(tactile_values)
        tactile_image = tactile_image.view(15,4,4,3) # Just making sure that everything stays the same
        tactile_image = torch.permute(tactile_image, (0,3,1,2))
        tactile_image = tactile_image.reshape(-1,4,4)
        return self.transform(tactile_image)

    def get_single_tactile_image(self, tactile_value):
        tactile_image = torch.FloatTensor(tactile_value) # tactile_value.shape: (16,3)
        tactile_image = tactile_image.view(4,4,3)
        tactile_image = torch.permute(tactile_image, (2,0,1))
        return self.transform(tactile_image) 

    def get_whole_hand_tactile_image(self, tactile_values):
        # tactile_values: (15,16,3) - turn it into 16,16,3 by concatenating 0z
        tactile_image = torch.FloatTensor(tactile_values)
        tactile_image = F.pad(tactile_image, (0,0,0,0,1,0), 'constant', 0)
        # reshape it to 4x4
        tactile_image = tactile_image.view(16,4,4,3)

        pad_idx = list(range(16))
        if self.shuffle_type == 'pad':
            random.seed(10)
            random.shuffle(pad_idx)
            
        tactile_image = torch.concat([
            torch.concat([tactile_image[pad_idx[i*4+j]] for j in range(4)], dim=0)
            for i in range(4)
        ], dim=1)

        if self.shuffle_type == 'whole':
            copy_tactile_image = copy(tactile_image)
            sensor_idx = list(range(16*16))
            random.seed(10)
            random.shuffle(sensor_idx)
            for i in range(16):
                for j in range(16):
                    rand_id = sensor_idx[i*16+j]
                    rand_i = int(rand_id / 16)
                    rand_j = int(rand_id % 16)
                    tactile_image[i,j,:] = copy_tactile_image[rand_i, rand_j, :]

        tactile_image = torch.permute(tactile_image, (2,0,1))

        return self.transform(tactile_image)
    
    def get_tactile_image_for_visualization(self, tactile_values):
        tactile_image = self._get_whole_hand_tactile_image(tactile_values)
        tactile_image = T.Resize(224)(tactile_image) # Don't need another normalization
        tactile_image = (tactile_image - tactile_image.min()) / (tactile_image.max() - tactile_image.min())
        return tactile_image  
