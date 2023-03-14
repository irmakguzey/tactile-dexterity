import numpy as np

from copy import deepcopy as copy

from tactile_dexterity.utils import *

# Custom nearest neighbor implementation
class KNearestNeighbors(object):
    def __init__(self, input_values, output_values):
        self.input_values = input_values
        self.output_values = output_values

    def get_sorted_idxs(self, datapoint):
        l1_distances = self.input_values - datapoint
        l2_distances = np.linalg.norm(l1_distances, axis = 1)

        sorted_idxs = np.argsort(l2_distances)
        return sorted_idxs

    def get_nearest_neighbor(self, datapoint):
        sorted_idxs = self.get_sorted_idxs(datapoint)
        nn_idx = sorted_idxs[0]
        return self.output_values[nn_idx], nn_idx

    def get_k_nearest_neighbors(self, datapoint, k):
        if k == 1:
            return self.get_nearest_neighbor(datapoint)

        assert datapoint.shape == self.input_values[0].shape

        sorted_idxs = self.get_sorted_idxs(datapoint)
        k_nn_idxs = sorted_idxs[:k]
        return self.output_values[k_nn_idxs], k_nn_idxs

# It uses each part separately to scale the distances 
class ScaledKNearestNeighbors(object):
    def __init__(
        self,
        input_values,
        output_values,
        repr_types,
        repr_importance,
        tactile_repr_size=64,
        image_repr_size = 512
    ):
        self.input_values = input_values 
        self.output_values = output_values 
        self.repr_types = self._preprocess_reprs(repr_types)
        self.repr_importance = repr_importance
        self.tactile_repr_size = tactile_repr_size
        self.image_repr_size = image_repr_size
        self._get_index_values() # Will set the beginning and ending indices for each repr type

    def _preprocess_reprs(self, repr_types):
        # Method to order the repesentations - if they are given in the incorrect order
        ordered_repr = []
        for correct_repr in ['image', 'tactile', 'kinova', 'allegro', 'torque']:
            if correct_repr in repr_types:
                ordered_repr.append(correct_repr)

        return ordered_repr

    def _get_index_values(self):
        self.index_values = {}
        last_index = 0
        for repr_type in self.repr_types:
            # Get each representation type - representation types should be given in order
            # [image, tactile, kinova, allegro, torque]
            # Even if some of them are missing they should follow the same order
            # All the representations other than the allegro and the kinova are already constant for this project
            if repr_type == 'image':
                self.index_values['image'] = [last_index, last_index+self.image_repr_size] # We are using
                last_index += self.image_repr_size
            elif repr_type == 'tactile':
                self.index_values['tactile'] = [last_index, last_index+self.tactile_repr_size]
                last_index += self.tactile_repr_size
            elif repr_type == 'kinova':
                self.index_values['kinova'] = [last_index, last_index+KINOVA_JOINT_NUM]
                last_index += KINOVA_JOINT_NUM
            elif repr_type == 'allegro':
                self.index_values['allegro'] = [last_index, last_index+ALLEGRO_EE_REPR_SIZE]
                last_index += ALLEGRO_EE_REPR_SIZE
            elif repr_type == 'torque':
                self.index_values['torque'] = [last_index, last_index+ALLEGRO_JOINT_NUM]
                last_index += ALLEGRO_JOINT_NUM

    def _get_type_based_dist(self, l1_distances, repr_type):
        type_based_idx = self.index_values[repr_type]
        type_l1_dist = l1_distances[:,type_based_idx[0]:type_based_idx[1]]
        type_l2_dist = np.linalg.norm(type_l1_dist, axis = 1)
        
        type_l2_dist = (type_l2_dist-type_l2_dist.min()) / (type_l2_dist.max() - type_l2_dist.min())
        repr_id = self.repr_types.index(repr_type)
        repr_importance = self.repr_importance[repr_id]
        return type_l2_dist * repr_importance

    def _get_l2_distances(self, datapoint):
        l1_distances = self.input_values - datapoint
        for i,repr_type in enumerate(self.repr_types):
            curr_l2_dist = self._get_type_based_dist(l1_distances, repr_type)
            if i == 0: 
                final_l2_dist = copy(curr_l2_dist)
                final_l2_dist_arr = np.expand_dims(curr_l2_dist, 1)
            else:
                final_l2_dist += curr_l2_dist
                final_l2_dist_arr = np.concatenate([final_l2_dist_arr, np.expand_dims(curr_l2_dist, 1)], axis=1)

        return final_l2_dist, final_l2_dist_arr

    def get_sorted_idxs(self, datapoint):
        l2_distances, separate_l2_distances = self._get_l2_distances(datapoint)
        
        sorted_idxs = np.argsort(l2_distances)
        sorted_separate_l2_dists = separate_l2_distances[sorted_idxs]
        return sorted_idxs, sorted_separate_l2_dists

    def get_nearest_neighbor(self, datapoint):
        sorted_idxs = self.get_sorted_idxs(datapoint)
        nn_idx = sorted_idxs[0]
        return self.output_values[nn_idx], nn_idx

    def get_k_nearest_neighbors(self, datapoint, k):
        if k == 1:
            return self.get_nearest_neighbor(datapoint)

        assert datapoint.shape == self.input_values[0].shape

        sorted_idxs, sorted_separate_l2_dists = self.get_sorted_idxs(datapoint)
        k_nn_idxs = sorted_idxs[:k]
        k_nn_separate_dists = sorted_separate_l2_dists[:k] # This will have separate dists for each representation
        return self.output_values[k_nn_idxs], k_nn_idxs, k_nn_separate_dists