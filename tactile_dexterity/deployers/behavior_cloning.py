# Helper script to load models
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn

from PIL import Image as im
from omegaconf import OmegaConf

from holobot.constants import *


from tactile_dexterity.utils import *
from tactile_dexterity.tactile_data import *

from .deployer import Deployer

class BC(Deployer):
    def __init__(
        self,
        data_path,
        deployment_dump_dir,
        out_dir, # We will be experimenting with the trained encoders with bc
        view_num = 1,
        representation_type = 'all' # 'tactile, image, all
    ):
        self.set_up_env()

        self.vision_view_num = view_num
        self.device = torch.device('cuda:0')
        self._init_encoder_info(self.device, out_dir)
        self.representation_type = representation_type

        self.state_id = 0
        self.inv_image_transform = get_inverse_image_norm()
        self.deployment_dump_dir = deployment_dump_dir
        os.makedirs(self.deployment_dump_dir, exist_ok=True)

    def _init_encoder_info(self, device, out_dir):
        cfg = OmegaConf.load(os.path.join(out_dir, '.hydra/config.yaml'))
        image_encoder_path = os.path.join(out_dir, 'models/bc_image_encoder_best.pt')
        self.image_encoder = load_model(cfg, device, image_encoder_path, bc_model_type='image')
        
        tactile_encoder_path = os.path.join(out_dir, 'models/bc_tactile_encoder_best.pt')
        tactile_encoder = load_model(cfg, device, tactile_encoder_path, bc_model_type='tactile')
        
        last_layer_path = os.path.join(out_dir, 'models/bc_last_layer_best.pt')
        self.last_layer = load_model(cfg, device, last_layer_path, bc_model_type='last_layer')

        # Set up the transforms for tactile and image encoders
        self.image_transform = T.Compose([
            T.Resize((480,640)),
            T.Lambda(crop_transform),
            T.ToTensor(),
            T.Normalize(VISION_IMAGE_MEANS, VISION_IMAGE_STDS),
        ])
        
        # Create the tactile representation and image modules
        self.tactile_img = TactileImage(
            tactile_image_size = 224, # This is set for alexnet 
            shuffle_type = None
        )
        self.tactile_repr = TactileRepresentation(
            encoder_out_dim = cfg.encoder.tactile_encoder.out_dim,
            tactile_encoder = tactile_encoder,
            tactile_image = self.tactile_img,
            representation_type = 'tdex'
        )

    def get_action(self, tactile_values, recv_robot_state, visualize=False):
        # Get the current visual image
        image = self._get_curr_image()

        tactile_repr = self.tactile_repr.get(tactile_values)
        image_repr = self.image_encoder(image.unsqueeze(dim=0)) # Add a dimension to the first axis so that it could be considered as a batch     
        all_repr = torch.concat((tactile_repr, image_repr), dim=-1)
        if self.representation_type == 'all':
            pred_action = self.last_layer(all_repr)
        elif self.representation_type == 'tactile':
            pred_action = self.last_layer(tactile_repr)
        elif self.representation_type == 'image':
            pred_action = self.last_layer(image_repr)
        pred_action = pred_action.squeeze().detach().cpu().numpy()
        
        action = dict(
            allegro = pred_action[:16],
            kinova = pred_action[16:]
        )
        
        if visualize:
            self._visualize_state(tactile_values, image)

        self.state_id += 1

        return action

    def _visualize_state(self, tactile_values, image):
        curr_image = self.inv_image_transform(image).numpy().transpose(1,2,0)
        curr_image_cv2 = cv2.cvtColor(curr_image*255, cv2.COLOR_RGB2BGR)

        tactile_image = self.tactile_img.get_tactile_image_for_visualization(tactile_values)
        dump_whole_state(tactile_values, tactile_image, None, None, title='curr_state', vision_state=curr_image_cv2)
        curr_state = cv2.imread('curr_state.png')
        image_path = os.path.join(self.deployment_dump_dir, f'state_{str(self.state_id).zfill(2)}.png')
        cv2.imwrite(image_path, curr_state)


