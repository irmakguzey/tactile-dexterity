from abc import ABC, abstractmethod

# Main class for all learner modules

class Learner(ABC):
    @abstractmethod
    def to(self, device):
        pass 

    @abstractmethod
    def train(self):
        pass 
    
    @abstractmethod
    def eval(self):
        pass 

    @abstractmethod
    def save(self, checkpoint_dir, model_type='best'):
        pass 

    @abstractmethod
    def train_epoch(self, train_loader):
        pass
