import glob
import numpy as np
import os
import pickle
import torch
import torchvision.transforms as T

from sklearn.decomposition import PCA as sklearn_PCA
from tqdm import tqdm

from tactile_dexterity.utils import load_data
from tactile_dexterity.utils import PLAY_DATA_PATH, TACTILE_IMAGE_MEANS, TACTILE_IMAGE_STDS

# Class for receiving tactile representation

class TactileRepresentation:

    def __init__(
        self,
        encoder_out_dim,
        tactile_encoder,
        tactile_image,
        representation_type, # raw, shared, stacked, tdex, sumpool, pca
    ):
        # self.size = cfg.encoder.out_dim * 15 if representation_type == 'shared' else cfg.encoder.out_dim
        self.tactile_image = self.tactile_image
        self.transform = T.Compose([
            T.Resize(tactile_image.size),
            T.Normalize(TACTILE_IMAGE_MEANS, TACTILE_IMAGE_STDS)
        ])
        self.encoder = tactile_encoder

        if representation_type == 'tdex':
            self.get = self._get_tdex_repr
            self.size = encoder_out_dim
        elif representation_type == 'shared':
            self.transform = T.Compose([
                T.Resize(tactile_image.size),
                T.Normalize(TACTILE_IMAGE_MEANS*15, TACTILE_IMAGE_STDS*15)
            ])
            self.get = self._get_shared_repr
            self.size = encoder_out_dim * 15
        elif representation_type == 'stacked':
            self.get = self._get_stacked_repr
            self.size = encoder_out_dim
        elif representation_type == 'pca':
            component_num = 100
            self.pca = self._fit_pca_to_play_data(n_components=component_num)
            self.get = self._get_pca_repr
            self.size = component_num
        elif representation_type == 'sumpool':
            self.sumpool_stats = None
            std, mean = self._find_sumpool_stats()
            self.sumpool_stats = [mean, std]
            self.get = self._get_sumpool_repr
            self.size = 45 # 15 * 3 only
        elif representation_type == 'raw':
            self.get = self._get_raw_repr
            self.size = 15 * 16 * 3

    def _get_tdex_repr(self, tactile_values):
        img = self.tactile_image.get_whole_hand_tactile_image(tactile_values)
        img = self.transform(img)
        return self.encoder(img.unsqueeze(0)).squeeze().detach().cpu().numpy()
    
    def _get_stacked_repr(self, tactile_values):
        img = self.tactile_image.get_stacked_tactile_image(tactile_values)
        img = self.transform(img)
        return self.encoder(img.unsqueeze(0)).squeeze().detach().cpu().numpy()
    
    def _get_shared_repr(self, tactile_values):
        for sensor_id in range(len(tactile_values)):
            curr_tactile_value = tactile_values[sensor_id]
            curr_tactile_image = self.tactile_image.get_single_tactile_image(curr_tactile_value).unsqueeze(0) # To make it as if it's a batch
            curr_tactile_image = self.transform(curr_tactile_image)
            if sensor_id == 0:
                curr_repr = self.encoder(curr_tactile_image).squeeze() # shape: (64)
            else:
                curr_repr =  torch.cat([curr_repr, self.encoder(curr_tactile_image).squeeze()], dim=0)

        return curr_repr.detach().cpu().numpy()
    
    def _get_raw_repr(self, tactile_values):
        return tactile_values.flatten()

    def _get_pca_repr(self, tactile_values):
        transformed_tactile_values = self.pca.transform(np.expand_dims(tactile_values.flatten(), axis=0))
        tactile_image = torch.FloatTensor(transformed_tactile_values)
        return tactile_image.squeeze()
    
    def _get_sumpool_repr(self, tactile_values):
        tactile_image = torch.FloatTensor(tactile_values)
        tactile_image = torch.sum(tactile_image, dim=1).flatten() # Shape: 15,3
        
        if self.sumpool_stats is None:
            return tactile_image
        
        return ((tactile_image - self.sumpool_stats[0]) / self.sumpool_stats[1]).flatten()

    def _find_sumpool_stats(self, sumpool_stats_path='sumpool_stats.pkl'):
        # Traverse through the whole play data and find the mean and stds of the sumpooled images
        if os.path.exists(sumpool_stats_path):
            roots = glob.glob(f'{PLAY_DATA_PATH}/demonstration_*')
            data = load_data(roots) 
            
            all_sumpooled_tactile_images = torch.zeros((len(data['tactile']['indices'])), 45)
            for i in tqdm(range(len(data['tactile']['indices']))):
                demo_id, tactile_id = data['tactile']['indices'][i]
                tactile_values = data['tactile']['values'][demo_id][tactile_id]
                sumpooled_tactile_image = self._get_tactile_repr_with_sumpool(tactile_values)
                all_sumpooled_tactile_images[i,:] = sumpooled_tactile_image[:]

            # Get the std and mean
            std, mean = torch.std_mean(all_sumpooled_tactile_images, dim=0)   
            
            # Dump them 
            with open(sumpool_stats_path, 'wb') as f:
                pickle.dump([std, mean], f)
            return std, mean
        
        with open(sumpool_stats_path, 'rb') as f: 
            std, mean = pickle.load(f)
        
        return std, mean
        
    def _fit_pca_to_play_data(self, n_components=100, pca_path='pca_play_data.pkl'):
        if os.path.exists(pca_path):
            # Create and dump the pca pickle file
            roots = glob.glob(f'{PLAY_DATA_PATH}/demonstration_*')
            data = load_data(roots)
            all_tactile_values = np.zeros((
                len(data['tactile']['indices']), 15*16*3
            ))
            for i in tqdm(range(len(data['tactile']['indices']))):
                demo_id, tactile_id = data['tactile']['indices'][i]
                all_tactile_values[i,:] = data['tactile']['values'][demo_id][tactile_id].flatten()
            
            # Fit the PCA
            pca = sklearn_PCA(n_components).fit(all_tactile_values)
            
            # Dump and save the PCA module so that it can be used afterwards
            with open(pca_path, 'wb') as f:
                pickle.dump(pca, f)
            return pca

        with open(pca_path, 'rb') as f:
            pca = pickle.load(f)
        return pca