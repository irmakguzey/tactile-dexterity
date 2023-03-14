import os
import torch

from .learner import Learner 

class BYOLLearner(Learner):
    def __init__(
        self,
        byol,
        optimizer,
        byol_type
    ):

        self.optimizer = optimizer 
        self.byol = byol
        self.byol_type = byol_type # Tactile or Image

    def to(self, device):
        self.device = device 
        self.byol.to(device)

    def train(self):
        self.byol.train()

    def eval(self):
        self.byol.eval()

    def save(self, checkpoint_dir, model_type='best'):
        torch.save(self.byol.state_dict(),
                   os.path.join(checkpoint_dir, f'byol_encoder_{model_type}.pt'),
                   _use_new_zipfile_serialization=False)

    def train_epoch(self, train_loader):
        self.train() 

        # Save the train loss
        train_loss = 0.0 

        # Training loop 
        for batch in train_loader: 
            image = batch.to(self.device)
            self.optimizer.zero_grad()

            # Get the loss by the byol            
            loss = self.byol(image)
            train_loss += loss.item() 

            # Backprop
            loss.backward() 
            self.optimizer.step()
            self.byol.update_moving_average() 

        return train_loss / len(train_loader)