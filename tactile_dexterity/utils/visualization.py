import cv2
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os

from holobot.utils.network import ZMQCameraSubscriber

def plot_tactile_sensor(ax, sensor_values, use_img=False, img=None, title='Tip Position'):
    # sensor_values: (16, 3) - 3 values for each tactile - x and y represents the position, z represents the pressure on the tactile point
    img_shape = (240, 240, 3) # For one sensor
    blank_image = np.ones(img_shape, np.uint8) * 255
    if use_img == False: 
        img = ax.imshow(blank_image.copy())
    ax.set_title(title)

    # Set the coordinates for each circle
    tactile_coordinates = []
    for j in range(48, 192+1, 48): # Y
        for i in range(48, 192+1, 48): # X - It goes from top left to bottom right row first 
            tactile_coordinates.append([i,j])

    # Plot the circles 
    for i in range(sensor_values.shape[0]):
        center_coordinates = (
            tactile_coordinates[i][0] + int(sensor_values[i,0]/20), # NOTE: Change this
            tactile_coordinates[i][1] + int(sensor_values[i,1]/20)
        )
        radius = max(10 + int(sensor_values[i,2]/10), 2)
      
        if i == 0:
            frame_axis = cv2.circle(blank_image.copy(), center_coordinates, radius, color=(0,255,0), thickness=-1)
        else:
            frame_axis = cv2.circle(frame_axis.copy(), center_coordinates, radius, color=(0,255,0), thickness=-1)

    img.set_array(frame_axis)

    return img, frame_axis

def dump_camera_image(host='172.24.71.240', image_stream_port=10005):
    image_subscriber = ZMQCameraSubscriber(
        host = host,
        port = image_stream_port,
        topic_type = 'RGB'
    )
    image, _ = image_subscriber.recv_rgb_image()
    cv2.imwrite('camera_image.png', image)

def plot_xyz_position(ax, position, title, color='blue', ylims=None):
    types = ['X', 'Y', 'Z']
    if ylims is None:
        ax.set_ylim(-0.05, 0.15)
    else:
        ax.set_ylim(ylims)
    ax.bar(types, position, color=color)
    ax.set_title(title)

def dump_tactile_state(tactile_values):
    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(10,10))
    for col_id in range(4):
        for row_id in range(4):
            if col_id + row_id > 0:
                plot_tactile_sensor(
                    ax = axs[row_id, col_id],
                    sensor_values = tactile_values[col_id*4+row_id-1],
                    title=f'Finger: {col_id}, Sensor: {row_id}'
                )
            axs[row_id, col_id].get_yaxis().set_ticks([])
            axs[row_id, col_id].get_xaxis().set_ticks([])
    fig.savefig('tactile_state.png', bbox_inches='tight')
    fig.clf()
    plt.close()

def dump_tactile_image(tactile_image):
    npimg = tactile_image.numpy()
    plt.axis('off')
    plt.imsave('tactile_image.png', np.transpose(npimg, (1,2,0)))

def dump_robot_state(allegro_tip_pos, kinova_cart_pos):
    fig = plt.figure(figsize=(10,10))
    allegro_axs = []
    for i in range(2):
        for j in range(2):
            allegro_axs.append(
                plt.subplot2grid((2,4), (i,j), fig=fig)
            )
    kinova_ax = plt.subplot2grid((2,4), (0,2), colspan=2, rowspan=2, fig=fig)
    for i,ax in enumerate(allegro_axs):
        plot_xyz_position(ax = ax, position = allegro_tip_pos[i*3:(i+1)*3], title=f'Finger {i}', color='mediumturquoise', ylims=(-0.1,0.2))
        ax.get_yaxis().set_ticks([])
    plot_xyz_position(ax = kinova_ax, position = kinova_cart_pos, title=f'Arm Wrist Position', color = 'darkolivegreen', ylims=(-0.75,0.75))
    kinova_ax.get_yaxis().set_ticks([])
    plt.savefig('robot_state.png', bbox_inches='tight')
    plt.close()

def dump_whole_state(tactile_values, tactile_image, allegro_tip_pos, kinova_cart_pos, title='curr_state', vision_state=None):
    dump_tactile_state(tactile_values)
    dump_tactile_image(tactile_image)
    tactile_state = cv2.imread('tactile_state.png')
    tactile_image = cv2.imread('tactile_image.png')
    tactile_state = concat_imgs(tactile_state, tactile_image, 'horizontal')
    if vision_state is None:
        dump_robot_state(allegro_tip_pos, kinova_cart_pos)
        robot_state = cv2.imread('robot_state.png')
        state_img = concat_imgs(tactile_state, robot_state, orientation='horizontal')
    else:
        cv2.imwrite(f'{title}_vision.png', vision_state)
        vision_img = cv2.imread(f'{title}_vision.png')
        state_img = concat_imgs(vision_img, tactile_state, orientation='horizontal')
    cv2.imwrite(f'{title}.png', state_img)

# If include temporal states it will plot closest neighbor to the right side of the current state
# and previous nn state to the bottom of the current and the next nn state to the bottom of the next nn
def dump_knn_state(dump_dir, img_name, image_repr=False, add_repr_effects=False, include_temporal_states=False): # image_repr - if image is part of repr we only show tactile and image
    os.makedirs(dump_dir, exist_ok=True)
    knn_state = cv2.imread('knn_state.png')
    curr_state = cv2.imread('curr_state.png')
    if include_temporal_states:
        next_knn_state = cv2.imread('next_knn_state.png')
        prev_knn_state = cv2.imread('prev_knn_state.png')
    if not image_repr:
        camera_img = cv2.imread('camera_image.png')
        state_img = concat_imgs(curr_state, knn_state, 'vertical')
        all_state_img = concat_imgs(camera_img, state_img, 'vertical')
    else:
        if include_temporal_states:
            curr_knn_state_img = concat_imgs(curr_state, knn_state, 'horizontal')
            prev_next_knn_state_img = concat_imgs(prev_knn_state, next_knn_state, 'horizontal')
            all_state_img = concat_imgs(curr_knn_state_img, prev_next_knn_state_img, 'vertical')
        else:
            all_state_img = concat_imgs(curr_state, knn_state, 'vertical')
        if add_repr_effects:
            repr_img = cv2.imread('repr_effects.png')
            all_state_img = concat_imgs(all_state_img, repr_img, 'vertical')
        
    cv2.imwrite(os.path.join(dump_dir, img_name), all_state_img)

def dump_repr_effects(nn_separate_dists, viz_id_of_nns, demo_nums, representation_types): # nn_separate_dists has all the dists from the closest applicable one to the rest 
    _, axs = plt.subplots(nrows=1,ncols=3,figsize=(10,3))
    for i in range(3):
        axs[i].set_ylim(0,1)
        axs[i].set_title(f'Demo Num: {demo_nums[i]} - NN ID: {viz_id_of_nns[i]}')
        axs[i].bar(representation_types, nn_separate_dists[i])

    plt.savefig('repr_effects.png', bbox_inches='tight')
    plt.close()

def concat_imgs(img1, img2, orientation='horizontal'): # Or it could be vertical as well
    metric_id = 0 if orientation == 'horizontal' else 1
    max_metric = max(img1.shape[metric_id], img2.shape[metric_id])
    min_metric = min(img1.shape[metric_id], img2.shape[metric_id])
    scale = min_metric / max_metric
    large_img_idx = np.argmax([img1.shape[metric_id], img2.shape[metric_id]])

    if large_img_idx == 0: 
        img1 = cv2.resize(
            img1, 
            (int(img1.shape[1]*scale),
             int(img1.shape[0]*scale))
        )
    else: 
        img2 = cv2.resize(
            img2, 
            (int(img2.shape[1]*scale),
             int(img2.shape[0]*scale))
        )

    concat_img = cv2.hconcat([img1, img2]) if orientation == 'horizontal' else cv2.vconcat([img1, img2])
    return concat_img

def turn_images_to_video(viz_dir, video_fps, video_name='visualization.mp4'):
    video_path = os.path.join(viz_dir, video_name)
    if os.path.exists(video_path):
        os.remove(video_path)
    os.system('ffmpeg -r {} -i {}/%*.png -vf setsar=1:1 {}'.format(
        video_fps, # fps
        viz_dir,
        video_path
    ))

def turn_video_to_images(dir_path, video_name, images_dir_name, images_fps):
    images_path = os.path.join(dir_path, images_dir_name)
    video_path = os.path.join(dir_path, video_name)
    os.makedirs(images_path, exist_ok=True)
    os.system(f'ffmpeg -i {video_path} -vf fps={images_fps} {images_path}/out%d.png')