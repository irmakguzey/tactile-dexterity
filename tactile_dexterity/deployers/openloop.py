# Script to deploy already created demo
import glob

from holobot.constants import *

from .deployer import Deployer
from tactile_dexterity.utils import load_data

class OpenLoop(Deployer):
    def __init__(
        self,
        data_path, # root in string
        demo_to_run,
        apply_allegro_states = False, # boolean to indicate if we should apply commanded allegro states or actual allegro states
    ):

        roots = glob.glob(f'{data_path}/demonstration_*')
        roots = sorted(roots)
        self.data = load_data(roots, demos_to_use=[demo_to_run])
        self.state_id = 0
        self.allegro_action_key = 'allegro_joint_states' if apply_allegro_states else 'allegro_actions'

    def get_action(self, tactile_values, recv_robot_state, visualize=False):
        demo_id, action_id = self.data[self.allegro_action_key]['indices'][self.state_id] 
        allegro_action = self.data[self.allegro_action_key]['values'][demo_id][action_id] # Get the next commanded action (commanded actions are saved in that timestamp)
        
        # Thumb should be set anyways
        _, allegro_state_id = self.data['allegro_joint_states']['indices'][self.state_id] 
        allegro_state = self.data['allegro_joint_states']['values'][demo_id][allegro_state_id]
        allegro_action[-4:] = allegro_state[-4:] # Fix the thumb
        action = dict(
            allegro = allegro_action
        )

        _, kinova_id = self.data['kinova']['indices'][self.state_id] 
        kinova_action = self.data['kinova']['values'][demo_id][kinova_id] # Get the next saved kinova_state
        action['kinova'] = kinova_action

        self.state_id += 1

        return action

    def save_deployment(self): # We don't really need to do anything here
        pass