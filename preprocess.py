import glob
import hydra 
from omegaconf import DictConfig

from tactile_dexterity.datasets import *

@hydra.main(version_base=None, config_path='tactile_dexterity/configs', config_name='preprocess')
def main(cfg : DictConfig) -> None:
    if cfg.process_single_demo:
        roots = [cfg.data_path]
    else:
        roots = glob.glob(f'{cfg.data_path}/demonstration_*') # TODO: change this in the future
        roots = sorted(roots)
    
    for demo_id, root in enumerate(roots):
        if demo_id > cfg.dump_after:
            if cfg.dump_fingertips:
                dump_fingertips(root=root)
            if dump_data_indices:
                dump_data_indices(
                    demo_id = demo_id, 
                    root = root, 
                    is_byol_tactile = cfg.tactile_byol, 
                    is_byol_image = cfg.vision_byol, 
                    threshold_step_size = cfg.threshold_step_size,
                    cam_view_num = cfg.view_num
                )
            if cfg.vision_byol:
                dump_video_to_images(root, view_num=cfg.view_num, dump_all=True) # If dump_all == False then it will use the desired images only
            elif cfg.dump_images:
                dump_video_to_images(root, view_num=cfg.view_num, dump_all=False)
        print('-----')    

if __name__ == '__main__':
    main()