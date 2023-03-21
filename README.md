# Dexterity from Touch: Self-Supervised Pre-Training of Tactile Representations with Robotic Play
[[Paper]](TODO) [[Project Website]](https://tactile-dexterity.github.io/) [[Data]](https://drive.google.com/drive/folders/148ycBmuvqkdESvAhStlhYJ7ISMgelEvL?usp=sharing)

<p align="center">
  <img width="30%" src="https://github.com/tactile-dexterity/tactile-dexterity.github.io/blob/main/mfiles/main_task/gamepad.gif">
  <img width="30%" src="https://github.com/tactile-dexterity/tactile-dexterity.github.io/blob/main/mfiles/main_task/bottle_opening.gif">
  <img width="30%" src="https://github.com/tactile-dexterity/tactile-dexterity.github.io/blob/main/mfiles/main_task/book_opening.gif">
 </p>

 <p align="center">
  <img width="30%" src="https://github.com/tactile-dexterity/tactile-dexterity.github.io/blob/main/mfiles/main_task/bowl_picking.gif">
  <img width="30%" src="https://github.com/tactile-dexterity/tactile-dexterity.github.io/blob/main/mfiles/main_task/cup_picking.gif">
</p>

**Authors**: [Irmak Guzey](https://irmakguzey.github.io/), [Ben Evans](https://bennevans.github.io/), [Soumith Chintala](https://soumith.ch/) and [Lerrel Pinto](https://www.lerrelpinto.com/), New York University and Meta AI

This repository includes the official implementation of [T-Dex](https://tactile-dexterity.github.io/), including the training pipeline of tactile encoders and the real-world deployment of the non-parametric imitation learning policies for dexterous manipulation tasks using [Allegro hand](https://www.wonikrobotics.com/research-robot-hand) with [XELA sensors integration](https://xelarobotics.com/en/integrations) and [Kinova arm](https://assistive.kinovarobotics.com/product/jaco-robotic-arm). 

Datasets for the play data and the demonstrations is uploaded in [this Google Drive link](https://drive.google.com/drive/folders/148ycBmuvqkdESvAhStlhYJ7ISMgelEvL). Instructions on how to use this dataset is given below.

## Getting started
The following assumes our current working directory is the root folder of this project repository; tested on Ubuntu 20.04 LTS (amd64).

### Setting up the project environments
- Install the project environment:
  ```
  conda env create --file=conda_env.yml
  ```
  This will create a conda environment with the name `tactile_dexterity`. 
- Activate the environment:
  ```
  conda activate tactile_dexterity
  ```
- Install the `tactile_dexterity` package by using `setup.py`.
  ```
  pip install -e .
  ```
  This command should be done inside the conda environment.
  You can test if the project package has been installed correctly by running `import tactile_dexterity` from a python shell. 
- To enable logging, log in with a `wandb` account:
  ```
  wandb login
  ```

### Pipeline Installation for Demonstration Collection
This work uses [Holo-Dex](https://github.com/SridharPandian/Holo-Dex) pipeline for demonstration collection and few of the interfaces implemented there. Also uses Holo-Dex package and API in order to connect to the robots.
You can install `holodex` as a separate package and follow the instructions to install pipeline and collect demonstrations. Same procedure is required for deployment as well. 

### Getting the tactile play data and the demonstration datasets
Datasets used for training and evaluation will be uploaded in [this Google Drive link](https://drive.google.com/drive/folders/148ycBmuvqkdESvAhStlhYJ7ISMgelEvL). 
- There will be two separate folders: 
    1. `play_data`: All the tactile play data. Kinova and Allegro commanded and current states are also saved along with tactile and visual observations.
    2. `evaluation`: Successful demonstrations used during robot runs. Each separate directory contains demonstrations for different tasks.
- Download and unzip the datasets. 
- Dataset paths will be updated in the configuration files. 

## Reproducing experiments
The following assumes our current working directory is the root folder of this project repository and the data provided above is being used. 
For each of these parts you should first activate the conda environment by running:
```
conda activate tactile_dexterity
```

### Preprocess
For both the play data trainings and the evaluation dataset we need a preprocessing procedure.
Steps for this are as follows:
1. Preprocess by running:
    ```
    python preprocess.py data_path=<data to preprocess>
    ```
    Here you should set the `data_path` variable to either the root task directory (which will be `<dataset_location>/evaluation/<task_name>`) or the play data directory (which will be `<dataset_location>/play_data`) .
    You can set the necessary parameters in `tactile_dexterity/configs/preprocess.yaml` file. 
2. Preprocessing should be done separately with different procedures.
    - If the preprocessing is done for tactile SSL training set the following parameters:
    ```
    vision_byol: false
    tactile_byol: true
    dump_images: false
    threshold_step_size: 0.1
    ```
    - If the preprocessing is done for image SSL training set the following parameters:
    ```
    vision_byol: true
    tactile_byol: false
    dump_images: true
    threshold_step_size: 0.1
    view_num: <camera-id>
    ```
    `view_num` parameter should be 0 if there is only one camera, otherwise it should be set to whichever camera you'd like to use.
    - If the preprocessing is done for robot deployments, then you should set the following parameters:
    ```
    vision_byol: true
    tactile_byol: false
    dump_images: true
    threshold_step_size: 0.2
    view_num: <camera-id>
    ```
    `threshold_step_size` can be changed according to the task but this is the default value. This parameter is the difference in the end effector position during subsampling. Please refer to the paper for more detailed information.

### Training
You can train encoders using Self-Supervised methods such as [BYOL](https://arxiv.org/abs/2006.07733) and [VICReg](https://arxiv.org/abs/2105.04906) for tactile and visual images. Also Behavior Cloning models with both image and the tactile. 
1. Use the following command to train a resnet encoder using the above mentioned SSL methods:
    ```
    python3 train.py data_dir=<path-to-desired-root> experiment=<experiment-name>
    ```
    Experiment name and the data directory to train can be changed accordingly. Experiment name is used for `wandb` especially.
    Dataroots used for preprocessing are the same for training.
2. Currently this command will train with BYOL with tactile images.In order to change the training type and the training object (tactile|image) one should modify `tactile_dexterity/configs/train.yaml` file. `learner`, `learner_type` and `dataset` variables should be changed accordingly. Namings are done similarly to the paper.

### Deploying Models
Trainings will save a snapshot of the used models under the `tactile-dexterity/out` with the time and experiment name. At each `deployer` model used the desired model paths should be retrieved from there.
You can deploy models by running:
```
python3 deploy.py data_path:<path-to-evaluation-task> deployer:<vinn|bc|openloop>
```
For each deployer module you should set the model directories to the snapshots saved. 
1. For VINN, set `tactile_out_dir` and `image_out_dir` to the desired encoders paths to be used.
2. For BC, set `out_dir` to the root of all the BC encoders that were saved during training.
3. For OpenLoop, you will not need an encoder.

**NOTE**: These instructions assume that you are running the `Holo-Dex` deployment API on a separate shell. Without communication to the robot these deployments cannot be done.


## Citation
**TODO**
If you use this repo in your research, please consider citing the paper as follows:
```
@article{arunachalam2022holodex,
  title={Holo-Dex: Teaching Dexterity with Immersive Mixed Reality},
  author={Sridhar Pandian Arunachalam and Irmak Guzey and Soumith Chintala and Lerrel Pinto},
  journal={arXiv preprint arXiv:2210.06463},
  year={2022}
}