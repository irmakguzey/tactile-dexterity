# Script to use some of the deployment wrappers and apply the actions
import hydra
import torch 
import sys

from holobot_api.api import DeployAPI # This import could be changed depending on how it's used
from holobot.utils.timer import FrequencyTimer
from omegaconf import DictConfig

class Deploy:
    def __init__(self, cfg, deployed_module):
        self.module = deployed_module
        required_data = {
            'rgb_idxs': [0],
            'depth_idxs': [0]
        }
        self.deploy_api = DeployAPI(
            host_address = '172.24.71.240',
            required_data = required_data
        )
        self.cfg = cfg
        self.data_path = cfg.data_path
        self.device = torch.device('cuda:0')
        self.frequency_timer = FrequencyTimer(cfg.frequency)

    def solve(self):
        sys.stdin = open(0) # To get inputs while spawning multiple processes

        while True:
            
            try:

                if self.cfg['loop']:
                    self.frequency_timer.start_loop()

                print('\n***************************************************************')
                print('\nGetting state information...') 

                # Get the robot state and the tactile info
                robot_state = self.deploy_api.get_robot_state() 
                sensor_state = self.deploy_api.get_sensor_state()

                allegro_joint_pos = robot_state['allegro']['position']
                allegro_joint_torque = robot_state['allegro']['effort']
                tactile_info = sensor_state['xela']['sensor_values']
                send_robot_state = dict(
                    allegro = allegro_joint_pos,
                    torque = allegro_joint_torque
                )
                kinova_state = robot_state['kinova']
                send_robot_state['kinova'] = kinova_state

                pred_action = self.module.get_action(
                    tactile_info,
                    send_robot_state,
                    visualize=self.cfg['visualize']
                )
                if not self.cfg['loop']:
                    register = input('\nPress a key to perform the action...')

                action_dict = dict() 
                action_dict['allegro'] = pred_action['allegro'] # Should be a numpy array
                action_dict['kinova'] = pred_action['kinova']
                self.deploy_api.send_robot_action(action_dict)

                if self.cfg['loop']: 
                    self.frequency_timer.end_loop()

            except KeyboardInterrupt:
                self.module.save_deployment() # This is supposed to save all the representaitons and run things 

@hydra.main(version_base=None, config_path='tactile_dexterity/configs', config_name='deploy')
def main(cfg : DictConfig) -> None:

    deployer = hydra.utils.instantiate(
        cfg.deployer,
        data_path = cfg.data_path
    )
    deploy = Deploy(cfg, deployer)
    deploy.solve()

if __name__ == '__main__':
    main()
