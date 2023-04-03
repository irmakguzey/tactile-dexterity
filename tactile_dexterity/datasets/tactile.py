import glob
import numpy as np
import torchvision.transforms as T 

from torch.utils import data

from tactile_dexterity.tactile_data import TactileImage
from tactile_dexterity.utils import load_data

class TactileSSLDataset(data.Dataset):
    # Dataset for all possible tactile types (stacked, whole hand, one sensor)
    def __init__(
        self,
        data_path,
        tactile_information_type, # It could be either one of - stacked, whole_hand, single_sensor
        tactile_img_size,
        duration=120, # Duration in minutes - the max is 120 minutes (it is considered max) - from now on the play 
        shuffle_type=None # Can either be pad, whole or None
    ):
        super().__init__()
        self.roots = glob.glob(f'{data_path}/demonstration_*')
        self.roots = sorted(self.roots)
        self.data = load_data(self.roots, demos_to_use=[], duration=duration)
        assert tactile_information_type in ['stacked', 'whole_hand', 'single_sensor'], 'tactile_information_type can either be "stacked", "whole_hand" or "single_sensor"'
        self.tactile_information_type = tactile_information_type
        self.shuffle_type = shuffle_type
        
        # Set the transforms accordingly
        self.tactile_img = TactileImage(
            tactile_image_size = tactile_img_size,
            shuffle_type = shuffle_type
        )

    def _preprocess_tactile_indices(self):
        self.tactile_mapper = np.zeros(len(self.data['tactile']['indices'])*15).astype(int)
        for data_id in range(len(self.data['tactile']['indices'])):
            for sensor_id in range(15):
                self.tactile_mapper[data_id*15+sensor_id] = data_id # Assign each finger to an index basically

    def _get_sensor_id(self, index):
        return index % 15
            
    def __len__(self):
        if self.tactile_information_type == 'single_sensor':
            return len(self.tactile_mapper)
        else: 
            return len(self.data['tactile']['indices'])
        
    def _get_proper_tactile_value(self, index):
        if self.tactile_information_type == 'single_sensor':
            data_id = self.tactile_mapper[index]
            demo_id, tactile_id = self.data['tactile']['indices'][data_id]
            sensor_id = self._get_sensor_id(index)
            tactile_value = self.data['tactile']['values'][demo_id][tactile_id][sensor_id]
            
            return tactile_value
        
        else:
            demo_id, tactile_id = self.data['tactile']['indices'][index]
            tactile_values = self.data['tactile']['values'][demo_id][tactile_id]
            
            return tactile_values

    def _get_tactile_image(self, tactile_values):
        return self.tactile_img.get(
            type = self.tactile_information_type,
            tactile_values = tactile_values
        )

    def __getitem__(self, index):
        tactile_value = self._get_proper_tactile_value(index)
        tactile_image = self._get_tactile_image(tactile_value)
        
        return tactile_image
        