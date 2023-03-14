import os 
import torch

from abc import ABC, abstractmethod
from PIL import Image as im

from tactile_dexterity.utils import *

# Base class for all deployment modules
class Deployer(ABC):
    def set_up_env(self):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29505"

        torch.distributed.init_process_group(backend='gloo', rank=0, world_size=1)
        torch.cuda.set_device(0)

    @abstractmethod
    def get_action(self, tactile_values, recv_robot_state, visualize=False):
        pass

    @abstractmethod
    def save_deployment(self):
        pass 

    def _get_curr_image(self, host='172.24.71.240', port=10005):
        image_subscriber = ZMQCameraSubscriber(
            host = host,
            port = port + self.view_num,
            topic_type = 'RGB'
        )
        image, _ = image_subscriber.recv_rgb_image()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = im.fromarray(image)
        img = self.image_transform(image)
        return torch.FloatTensor(img)