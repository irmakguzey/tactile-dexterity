import glob
import os
import torch
import torchvision.transforms as T 

from torchvision.datasets.folder import default_loader as loader 
from torch.utils import data

from tactile_dexterity.utils import load_data, crop_transform, VISION_IMAGE_MEANS, VISION_IMAGE_STDS

# Vision only dataset
class VisionDataset(data.Dataset):
    def __init__(
        self,
        data_path,
        view_num
    ):
        super().__init__()
        self.roots = glob.glob(f'{data_path}/demonstration_*')
        self.roots = sorted(self.roots)
        self.data = load_data(self.roots, demos_to_use=[])
        self.view_num = view_num

        self.transform = T.Compose([
            T.Resize((480,640)),
            T.Lambda(crop_transform),
            T.ToTensor(),
            T.Normalize(VISION_IMAGE_MEANS, VISION_IMAGE_STDS),
        ])

    def __len__(self):
        return len(self.data['image']['indices'])
        
    def _get_image(self, index):
        demo_id, image_id = self.data['image']['indices'][index]
        image_root = self.roots[demo_id]
        image_path = os.path.join(image_root, 'cam_{}_rgb_images/frame_{}.png'.format(self.view_num, str(image_id).zfill(5)))
        img = self.transform(loader(image_path))
        return torch.FloatTensor(img)

    def __getitem__(self, index):
        vision_image = self._get_image(index)
        
        return vision_image
