# Main training script - trains distributedly accordi
import os
import hydra

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm 

# Custom imports 
from tactile_dexterity.datasets import get_dataloaders
from tactile_dexterity.learners import init_learner
from tactile_dexterity.datasets import *
from tactile_dexterity.utils import *

class Workspace:
    def __init__(self, cfg : DictConfig) -> None:
        print(f'Workspace config: {OmegaConf.to_yaml(cfg)}')

        # Initialize hydra
        self.hydra_dir = HydraConfig.get().run.dir

        # Create the checkpoint directory - it will be inside the hydra directory
        cfg.checkpoint_dir = os.path.join(self.hydra_dir, 'models')
        os.makedirs(cfg.checkpoint_dir, exist_ok=True) # Doesn't give an error if dir exists when exist_ok is set to True 
        
        # Set the world size according to the number of gpus
        cfg.num_gpus = torch.cuda.device_count()
        cfg.world_size = cfg.world_size * cfg.num_gpus

        # Set device and config
        self.cfg = cfg

    def train(self, rank) -> None:
        # Create default process group
        dist.init_process_group("gloo", rank=rank, world_size=self.cfg.world_size)
        dist.barrier() # Wait for all of the processes to start
        
        # Set the device
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')

        # It looks at the datatype type and returns the train and test loader accordingly
        train_loader, test_loader, _ = get_dataloaders(self.cfg)

        # Initialize the learner - looks at the type of the agent to be initialized first
        learner = init_learner(self.cfg, device, rank)

        best_loss = torch.inf 

        # Logging
        if rank == 0:
            pbar = tqdm(total=self.cfg.train_epochs)
            # Initialize logger (wandb)
            if self.cfg.logger:
                wandb_exp_name = '-'.join(self.hydra_dir.split('/')[-2:])
                self.logger = Logger(self.cfg, wandb_exp_name, out_dir=self.hydra_dir)

        # Start the training
        for epoch in range(self.cfg.train_epochs):
            # Distributed settings
            if self.cfg.distributed:
                train_loader.sampler.set_epoch(epoch)
                dist.barrier()

            # Train the models for one epoch
            train_loss = learner.train_epoch(train_loader)

            if self.cfg.distributed:
                dist.barrier()

            if rank == 0: # Will only print after everything is finished
                pbar.set_description(f'Epoch {epoch}, Train loss: {train_loss:.5f}, Best loss: {best_loss:.5f}')
                pbar.update(1) # Update for each batch

            # Logging
            if self.cfg.logger and rank == 0 and epoch % self.cfg.log_frequency == 0:
                self.logger.log({'epoch': epoch,
                                 'train loss': train_loss})

            # Testing and saving the model
            if epoch % self.cfg.save_frequency == 0 and rank == 0: # NOTE: Not sure why this is a problem but this could be the fix
                learner.save(self.cfg.checkpoint_dir, model_type='latest') # Always save the latest encoder
                # Test for one epoch
                if not self.cfg.self_supervised:
                    test_loss = learner.test_epoch(test_loader)
                else:
                    test_loss = train_loss # In BYOL (for ex) test loss is not important

                # Get the best loss
                if test_loss < best_loss:
                    best_loss = test_loss
                    learner.save(self.cfg.checkpoint_dir, model_type='best')

                # Logging
                if rank == 0:
                    pbar.set_description(f'Epoch {epoch}, Test loss: {test_loss:.5f}')
                    if self.cfg.logger:
                        self.logger.log({'epoch': epoch,
                                        'test loss': test_loss})
                        self.logger.log({'epoch': epoch,
                                        'best loss': best_loss})

        if rank == 0: 
            pbar.close()

@hydra.main(version_base=None,config_path='tactile_dexterity/configs', config_name = 'train')
def main(cfg : DictConfig) -> None:
    # We are only training everything distributedly
    print('CFG.PREPROCESS: {}'.format(cfg.preprocess))
    assert cfg.distributed is True, "Use script only to train distributed"
    workspace = Workspace(cfg)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29503"
    
    print("Distributed training enabled. Spawning {} processes.".format(workspace.cfg.world_size))
    mp.spawn(workspace.train, nprocs=workspace.cfg.world_size)
    
if __name__ == '__main__':
    main()