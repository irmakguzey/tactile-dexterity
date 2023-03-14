import hydra 

from torch.nn.parallel import DistributedDataParallel as DDP

from .byol import BYOLLearner
from .vicreg import VICRegLearner
from .behavior_cloning import ImageTactileBC

from tactile_dexterity.utils import *
from tactile_dexterity.models import BYOL, VICReg, create_fc 

def init_learner(cfg, device, rank):
    if cfg.learner_type == 'bc':
        return init_bc(cfg, device, rank)
    elif 'tactile' in cfg.learner_type:
        return init_tactile_byol(
            cfg,
            device, 
            rank,
            aug_stat_multiplier=cfg.learner.aug_stat_multiplier,
            byol_in_channels=cfg.learner.byol_in_channels,
            byol_hidden_layer=cfg.learner.byol_hidden_layer
        )
    elif cfg.learner_type == 'image_byol':
        return init_image_byol(cfg, device, rank)
    elif cfg.learner_type == 'vicreg':
        return init_tactile_vicreg(
            cfg,
            device,
            rank,
            sim_coef=cfg.learner.sim_coef,
            std_coef=cfg.learner.std_coef,
            cov_coef=cfg.learner.cov_coef)
    
    return None

def init_tactile_vicreg(cfg, device, rank, sim_coef, std_coef, cov_coef):

    backbone = hydra.utils.instantiate(cfg.encoder).to(device)
    augment_fn = get_tactile_augmentations(
        img_means = TACTILE_IMAGE_MEANS,
        img_stds = TACTILE_IMAGE_STDS,
        img_size = (cfg.tactile_image_size, cfg.tactile_image_size)
    )

    # Initialize the vicreg projector
    projector = create_fc(
        input_dim = cfg.encoder.out_dim,
        output_dim = 8192,
        hidden_dims = [8192],
        use_batchnorm = True
    )

    # Initialize the vicreg wrapper 
    vicreg_wrapper = VICReg(
        backbone = backbone,
        projector = projector, 
        augment_fn = augment_fn, 
        sim_coef = sim_coef, 
        std_coef = std_coef, 
        cov_coef = cov_coef
    ).to(device)
    backbone = DDP(backbone, device_ids=[rank], output_device=rank, broadcast_buffers=False)
    projector = DDP(projector, device[rank], output_device=rank, broadcast_buffers=False)

    # Initialize the optimizer 
    optimizer = hydra.utils.instantiate(cfg. optimizer,
                                        params = vicreg_wrapper.parameters())

    learner = VICRegLearner(
        vicreg_wrapper = vicreg_wrapper, 
        optimizer = optimizer
    )                 
    learner.to(device)          

    return learner

def init_tactile_byol(cfg, device, rank, aug_stat_multiplier=1, byol_in_channels=3, byol_hidden_layer=-2):
    # Start the encoder
    encoder = hydra.utils.instantiate(cfg.encoder).to(device)

    augment_fn = get_tactile_augmentations(
        img_means = TACTILE_IMAGE_MEANS*aug_stat_multiplier,
        img_stds = TACTILE_IMAGE_STDS*aug_stat_multiplier,
        img_size = (cfg.tactile_image_size, cfg.tactile_image_size)
    )
    # Initialize the byol wrapper
    byol = BYOL(
        net = encoder,
        image_size = cfg.tactile_image_size,
        augment_fn = augment_fn,
        hidden_layer = byol_hidden_layer,
        in_channels = byol_in_channels
    ).to(device)
    encoder = DDP(encoder, device_ids=[rank], output_device=rank, broadcast_buffers=False)
    
    # Initialize the optimizer 
    optimizer = hydra.utils.instantiate(cfg.optimizer,
                                        params = byol.parameters())
    
    # Initialize the agent
    learner = BYOLLearner(
        byol = byol,
        optimizer = optimizer,
        byol_type = 'tactile'
    )

    learner.to(device)

    return learner

def init_image_byol(cfg, device, rank):
    # Start the encoder
    encoder = hydra.utils.instantiate(cfg.encoder).to(device)

    augment_fn = get_vision_augmentations(
        img_means = VISION_IMAGE_MEANS,
        img_stds = VISION_IMAGE_STDS
    )
    # Initialize the byol wrapper
    byol = BYOL(
        net = encoder,
        image_size = cfg.vision_image_size,
        augment_fn = augment_fn
    ).to(device)
    encoder = DDP(encoder, device_ids=[rank], output_device=rank, broadcast_buffers=False)
    
    # Initialize the optimizer 
    optimizer = hydra.utils.instantiate(cfg.optimizer,
                                        params = byol.parameters())
    
    # Initialize the agent
    learner = BYOLLearner(
        byol = byol,
        optimizer = optimizer,
        byol_type = 'image'
    )

    learner.to(device)

    return learner


def init_bc(cfg, device, rank):
    image_encoder = hydra.utils.instantiate(cfg.encoder.image_encoder).to(device)
    image_encoder = DDP(image_encoder, device_ids=[rank], output_device=rank, broadcast_buffers=False)

    tactile_encoder = hydra.utils.instantiate(cfg.encoder.tactile_encoder).to(device)
    tactile_encoder = DDP(tactile_encoder, device_ids=[rank], output_device=rank, broadcast_buffers=False)

    last_layer = hydra.utils.instantiate(cfg.encoder.last_layer).to(device)
    last_layer = DDP(last_layer, device_ids=[rank], output_device=rank, broadcast_buffers=False)

    optim_params = list(image_encoder.parameters()) + list(tactile_encoder.parameters()) + list(last_layer.parameters())
    optimizer = hydra.utils.instantiate(cfg.optimizer, params = optim_params)

    learner = ImageTactileBC(
        image_encoder = image_encoder, 
        tactile_encoder = tactile_encoder,
        last_layer = last_layer,
        optimizer = optimizer,
        loss_fn = 'mse',
        representation_type='all'
    )
    learner.to(device) 
    
    return learner