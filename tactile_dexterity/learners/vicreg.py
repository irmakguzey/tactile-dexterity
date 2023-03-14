import os 
import torch

from .learner import Learner 

class VICRegLearner(Learner):
    def __init__(
        self,
        vicreg_wrapper,
        optimizer
    ):
        self.optimizer = optimizer
        self.vicreg_wrapper = vicreg_wrapper 

    def to(self, device):
        self.vicreg_wrapper.to(device)

    def train(self):
        self.vicreg_wrapper.train() 

    def eval(self): 
        self.vicreg_wrapper.eval()  

    def save(self, checkpoint_dir, model_type='best'):
        vicreg_encoder_weights = self.vicreg_wrapper.get_encoder_weights()
        torch.save(
            vicreg_encoder_weights, 
            os.path.join(checkpoint_dir, f'vicreg_encoder_{model_type}.pt'),
            _use_new_zipfile_serialization=False
        )

    def train_epoch(self, train_loader):
        self.train() 

        train_loss = 0.

        for batch in train_loader:
            image = batch.to(self.device)

            self.optimier.zero_grad() 
            
            loss, _ = self.vicreg_wrapper.forward(image)
            train_loss += loss.item() 

            loss.backward() 
            self.optimizer.step() 

        return train_loss / len(train_loader)