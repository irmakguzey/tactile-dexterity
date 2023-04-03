import os 
import cv2
import h5py 
import numpy as np
import pickle 
import shutil

from tqdm import tqdm

from holobot.robot.allegro.allegro_kdl import AllegroKDL

# Method to dump a video to images - in order to receive images for timestamps

# Dumping video to images
# Creating pickle files to pick images
def dump_video_to_images(root: str, video_type: str ='rgb', view_num: int=0, dump_all=True) -> None:
    # Convert the video into image sequences and name them with the frames
    video_path = os.path.join(root, f'cam_{view_num}_{video_type}_video.avi')
    images_path = os.path.join(root, f'cam_{view_num}_{video_type}_images')
    if os.path.exists(images_path):
        print(f'{images_path} exists it is being removed')
        shutil.rmtree(images_path)

    os.makedirs(images_path, exist_ok=True)

    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not dump_all:
        with open(os.path.join(root, 'image_indices.pkl'), 'rb') as f:
            desired_indices = pickle.load(f)

    frame_id = 0
    desired_img_id = 0
    print(f'dumping video in {root}')
    pbar = tqdm(total = frame_count)
    while success: 
        pbar.update(1)
        if (not dump_all and frame_id == desired_indices[desired_img_id][1]) or dump_all:
            cv2.imwrite('{}.png'.format(os.path.join(images_path, 'frame_{}'.format(str(frame_id).zfill(5)))), image)
            curr_id = desired_img_id
            while desired_img_id < len(desired_indices)-1 and desired_indices[curr_id][1] == desired_indices[desired_img_id][1]:
                desired_img_id += 1
        success, image = vidcap.read()
        frame_id += 1

    print(f'Dumping finished in {root}')

# Method to dump fingertip positions
def dump_fingertips(root):
    print(f'Dumping fingertip positions in root {root}')
    allegro_states_path = os.path.join(root, 'allegro_joint_states.h5')
    with h5py.File(allegro_states_path, 'r') as f:
        joint_positions = f['positions'][()]
        timestamps = f['timestamps'][()]
    
    # Start the kdl solver
    fingertip_states = dict() # Will have timestamps as the allegro_joint_states and the tip positions of each finger
    allegro_kdl_solver = AllegroKDL()

    for i in range(len(joint_positions)):
        joint_position = joint_positions[i] 
        timestamp = timestamps[i]
        # There are 3 (x,y,z) fingertip positions for each finger
        fingertip_position = allegro_kdl_solver.get_fingertip_coords(joint_position)
        if i == 0:
            fingertip_states['positions'] = [fingertip_position]
            fingertip_states['timestamps'] = [timestamp]
        else:
            fingertip_states['positions'].append(fingertip_position)
            fingertip_states['timestamps'].append(timestamp)

    # Compress the data file
    fingertip_state_file = os.path.join(root, 'allegro_fingertip_states.h5')
    with h5py.File(fingertip_state_file, 'w') as file:
        for key in fingertip_states.keys():
            if key == 'timestamps':
                fingertip_states[key] = np.array(fingertip_states[key], dtype=np.float64)
            else:
                fingertip_states[key] = np.array(fingertip_states[key], dtype=np.float32)

            file.create_dataset(key, data = fingertip_states[key], compression='gzip', compression_opts=6)

    print(f'Saved fingertip positions in {fingertip_state_file}')

def dump_data_indices(
        demo_id,
        root,
        is_byol_tactile=False,
        is_byol_image=False,
        threshold_step_size=0.012,
        cam_view_num=0,
        find_steps_separately=True
    ):
    # Matches the index -> demo_id, datapoint_id according to the timestamps saved
    allegro_indices, image_indices, tactile_indices, allegro_action_indices, kinova_indices = [], [], [], [], []
    allegro_states_path = os.path.join(root, 'allegro_joint_states.h5')
    image_metadata_path = os.path.join(root, f'cam_{cam_view_num}_rgb_video.metadata')
    tactile_info_path = os.path.join(root, 'touch_sensor_values.h5')
    allegro_commanded_joint_path = os.path.join(root, 'allegro_commanded_joint_states.h5')
    kinova_states_path = os.path.join(root, 'kinova_cartesian_states.h5')

    with h5py.File(allegro_states_path, 'r') as f:
        allegro_timestamps = f['timestamps'][()]
        allegro_positions = f['positions'][()]
    with h5py.File(tactile_info_path, 'r') as f:
        tactile_timestamps = f['timestamps'][()]
    with h5py.File(allegro_commanded_joint_path, 'r') as f:
        allegro_action_timestamps = f['timestamps'][()]
    with h5py.File(kinova_states_path, 'r') as f:
        kinova_timestamps = f['timestamps'][()]
        kinova_positions = f['positions'][()]
    with open(image_metadata_path, 'rb') as f:
        image_metadata = pickle.load(f)
        image_timestamps = np.asarray(image_metadata['timestamps']) / 1000.
    
    # Start the allegro kdl solver 
    allegro_kdl_solver = AllegroKDL()

    allegro_id, image_id, tactile_id, allegro_action_id, kinova_id = 0, 0, 0, 0, 0
    # Get the first timestamps
    tactile_timestamp = tactile_timestamps[0]
    allegro_id = get_closest_id(allegro_id, tactile_timestamp, allegro_timestamps)
    image_id = get_closest_id(image_id, tactile_timestamp, image_timestamps)
    allegro_action_id = get_closest_id(allegro_action_id, tactile_timestamp, allegro_action_timestamps)
    kinova_id = get_closest_id(kinova_id, tactile_timestamp, kinova_timestamps)

    tactile_indices.append([demo_id, tactile_id])
    allegro_indices.append([demo_id, allegro_id])
    image_indices.append([demo_id, image_id])
    allegro_action_indices.append([demo_id, allegro_action_id])
    kinova_indices.append([demo_id, kinova_id])

    while (True):
        if is_byol_tactile:
            tactile_id += 1
            if tactile_id >= len(tactile_timestamps):
                break
            metric_timestamp = tactile_timestamps[tactile_id]
        elif is_byol_image:
            image_id += 1
            if image_id >= len(image_timestamps):
                break
            metric_timestamp = image_timestamps[image_id]        
        else:

            if find_steps_separately:
                # If this is true we will subsample robot states separately and then find the smallest one for the
                # step to use
                pos_allegro_id = find_next_allegro_id(
                    allegro_kdl_solver,
                    allegro_positions,
                    allegro_id,
                    threshold_step_size=threshold_step_size # When you preprocess for training, one should decrease this size - we need more data
                )
                pos_kinova_id = find_next_kinova_id(
                    kinova_positions,
                    kinova_id,
                    threshold_step_size=threshold_step_size*2
                )

                if pos_allegro_id >= len(allegro_positions) or pos_kinova_id >= len(kinova_positions):
                    break
                metric_timestamp = min(kinova_timestamps[pos_kinova_id], allegro_timestamps[pos_allegro_id])

            else: # If this is true then we'll find the common robot step with separation from both of them
                metric_timestamp = find_next_robot_state_timestamp(
                    allegro_kdl_solver, allegro_positions, allegro_timestamps, allegro_id, 
                    kinova_positions, kinova_id, kinova_timestamps,
                    threshold_step_size = threshold_step_size 
                )
                if metric_timestamp is None:
                    break

        # Get the ids from this metric timestamp
        allegro_id = get_closest_id(allegro_id, metric_timestamp, allegro_timestamps)
        allegro_action_id = get_closest_id(allegro_action_id, metric_timestamp, allegro_action_timestamps)
        kinova_id = get_closest_id(kinova_id, metric_timestamp, kinova_timestamps)
        image_id = get_closest_id(image_id, metric_timestamp, image_timestamps)
        tactile_id = get_closest_id(tactile_id, metric_timestamp, tactile_timestamps)

        # If some of the data has ended then consider the trajectory finished
        if image_id >= len(image_timestamps)-1 or \
           tactile_id >= len(tactile_timestamps)-1 or \
           allegro_id >= len(allegro_timestamps)-1 or \
           allegro_action_id >= len(allegro_action_timestamps)-1 or \
           kinova_id >= len(kinova_timestamps)-1:
            break

        tactile_indices.append([demo_id, tactile_id])
        allegro_indices.append([demo_id, allegro_id])
        image_indices.append([demo_id, image_id])
        allegro_action_indices.append([demo_id, allegro_action_id])
        kinova_indices.append([demo_id, kinova_id])

    assert len(tactile_indices) == len(allegro_indices) and \
           len(tactile_indices) == len(image_indices) and \
           len(tactile_indices) == len(allegro_action_indices) and \
           len(tactile_indices) == len(kinova_indices)

    # Save the indices for that root
    last_stopping_index = 1 # Not including the last few frames since they are noisy a bit
    with open(os.path.join(root, 'tactile_indices.pkl'), 'wb') as f:
        pickle.dump(tactile_indices[:-last_stopping_index], f)
    with open(os.path.join(root, 'allegro_indices.pkl'), 'wb') as f:
        pickle.dump(allegro_indices[:-last_stopping_index], f)
    with open(os.path.join(root, 'image_indices.pkl'), 'wb') as f:
        pickle.dump(image_indices[:-last_stopping_index], f)
    with open(os.path.join(root, 'allegro_action_indices.pkl'), 'wb') as f:
        pickle.dump(allegro_action_indices[:-last_stopping_index], f)
    with open(os.path.join(root, 'kinova_indices.pkl'), 'wb') as f:
        pickle.dump(kinova_indices[:-last_stopping_index], f)

    
def find_next_allegro_id(kdl_solver, positions, pos_id, threshold_step_size=0.01):
    old_allegro_pos = positions[pos_id]
    old_allegro_fingertip_pos = kdl_solver.get_fingertip_coords(old_allegro_pos)
    for i in range(pos_id, len(positions)):
        curr_allegro_fingertip_pos = kdl_solver.get_fingertip_coords(positions[i])
        step_size = np.linalg.norm(old_allegro_fingertip_pos - curr_allegro_fingertip_pos)
        if step_size > threshold_step_size: 
            return i

    return i # This will return len(positions)-1 when there are not enough 

def find_next_kinova_id(positions, pos_id, threshold_step_size=0.1):
    old_kinova_pos = positions[pos_id]
    for i in range(pos_id, len(positions)):
        curr_kinova_pos = positions[i]
        step_size = np.linalg.norm(old_kinova_pos - curr_kinova_pos)
        if step_size > threshold_step_size:
            return i 

    return i

def find_next_robot_state_timestamp(kdl_solver, allegro_positions, allegro_timestamps, allegro_id, kinova_positions, kinova_id, kinova_timestamps, threshold_step_size=0.015):
    # Find the timestamp where allegro and kinova difference in total is larger than the given threshold
    old_allegro_pos = allegro_positions[allegro_id]
    old_allegro_fingertip_pos = kdl_solver.get_fingertip_coords(old_allegro_pos)
    old_kinova_pos = kinova_positions[kinova_id]
    for i in range(allegro_id, len(allegro_positions)):
        curr_allegro_fingertip_pos = kdl_solver.get_fingertip_coords(allegro_positions[i])
        curr_allegro_timestamp = allegro_timestamps[i]
        kinova_id = get_closest_id(kinova_id, curr_allegro_timestamp, kinova_timestamps)
        curr_kinova_pos = kinova_positions[kinova_id]
        step_size = np.linalg.norm(old_allegro_fingertip_pos - curr_allegro_fingertip_pos) + np.linalg.norm(old_kinova_pos - curr_kinova_pos)
        if step_size > threshold_step_size:
            return curr_allegro_timestamp

        if kinova_id >= len(kinova_positions):
            return None
            
    return None

def get_closest_id(curr_id, desired_timestamp, all_timestamps):
    for i in range(curr_id, len(all_timestamps)):
        if all_timestamps[i] >= desired_timestamp:
            return i # Find the first timestamp that is after that
    
    return i
