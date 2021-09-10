# Nerfies: Deformable Neural Radiance Fields

This is the code for Nerfies: Deformable Neural Radiance Fields.

 * [Project Page](https://nerfies.github.io)
 * [Paper](https://arxiv.org/abs/2011.12948)
 * [Video](https://www.youtube.com/watch?v=MrKrnHhk8IA)
 
This codebase is implemented using [JAX](https://github.com/google/jax), 
building on [JaxNeRF](https://github.com/google-research/google-research/tree/master/jaxnerf).

This repository has been updated to reflect the version used for our ICCV 2021 submission.

## Demo

We provide an easy-to-get-started demo using Google Colab!

These Colabs will allow you to train a basic version of our method using 
Cloud TPUs (or GPUs) on Google Colab. 

Note that due to limited compute resources available, these are not the fully 
featured models. If you would like to train a fully featured Nerfie, please 
refer to the instructions below on how to train on your own machine.

| Description      | Link |
| ----------- | ----------- |
| Process a video into a Nerfie dataset| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/nerfies/blob/main/notebooks/Nerfies_Capture_Processing.ipynb)|
| Train a Nerfie| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/nerfies/blob/main/notebooks/Nerfies_Training.ipynb)|
| Render a Nerfie video| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/nerfies/blob/main/notebooks/Nerfies_Render_Video.ipynb)|
 
## Setup
The code can be run under any environment with Python 3.8 and above.
(It may run with lower versions, but we have not tested it).

We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and setting up an environment:

    conda create --name nerfies python=3.8

Next, install the required packages:

    pip install -r requirements.txt
    
Install the appropriate JAX distribution for your environment by  [following the instructions here](https://github.com/google/jax#installation). For example:

    # For CUDA version 11.0
    pip install --upgrade "jax[cuda111]" -f https://storage.googleapis.com/jax-releases/jax_releases.html


## Training
After preparing a dataset, you can train a Nerfie by running:

    export DATASET_PATH=/path/to/dataset
    export EXPERIMENT_PATH=/path/to/save/experiment/to
    python train.py \
        --data_dir $DATASET_PATH \
        --base_folder $EXPERIMENT_PATH \
        --gin_configs configs/test_vrig.gin
 
To plot telemetry to Tensorboard and render checkpoints on the fly, also
launch an evaluation job by running:

    python eval.py \
        --data_dir $DATASET_PATH \
        --base_folder $EXPERIMENT_PATH \
        --gin_configs configs/test_vrig.gin

The two jobs should use a mutually exclusive set of GPUs. This division allows the
training job to run without having to stop for evaluation.

## Configuration
 * We use [Gin](https://github.com/google/gin-config) for configuration.
 * We provide a couple preset configurations.
 * Please refer to `config.py` for documentation on what each configuration does.
 * Preset configs:
    - `gpu_vrig_paper.gin`: This is the configuration we used to generate the table in the paper. It requires 8 GPUs for training.
    - `gpu_fullhd.gin`: This is a high-resolution model and will take around 3 days to train on 8 GPUs.
    - `gpu_quarterhd.gin`: This is a low-resolution model and will take around 14 hours to train on 8 GPUs.
    - `test_local.gin`: This is a test configuration to see if the code runs. It probably will not result in a good looking result.
    - `test_vrig.gin`: This is a test configuration to see if the code runs for validation rig captures. It probably will not result in a good looking result.
 * Training on fewer GPUs will require tuning of the batch size and learning rates. We've provided an example configuration for 4 GPUs in `gpu_quarterhd_4gpu.gin` but we have not tested it, so please only use it as a reference.

## Datasets
A dataset is a directory with the following structure:

    dataset
        ├── camera
        │   └── ${item_id}.json
        ├── camera-paths
        ├── rgb
        │   ├── ${scale}x
        │   └── └── ${item_id}.png
        ├── metadata.json
        ├── points.npy
        ├── dataset.json
        └── scene.json

At a high level, a dataset is simply the following:
 * A collection of images (e.g., from a video).
 * Camera parameters for each image.
 
We have a unique identifier for each image which we call `item_id`, and this is
used to match the camera and images. An `item_id` can be any string, but typically
it is some alphanumeric string such as `000054`.

### `camera`

 * This directory contains cameras corresponding to each image.
 * We use a camera model identical to the [OpenCV camera model](https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html), which is also supported by COLMAP.
 * Each camera is a serialized version of the `Camera` class defined in `camera.py` and looks like this:

```javascript
{
  // A 3x3 world-to-camera rotation matrix representing the camera orientation.
  "orientation": [
    [0.9839, -0.0968, 0.1499],
    [-0.0350, -0.9284, -0.3699],
    [0.1749, 0.358, -0.9168]
  ],
  // The 3D position of the camera in world-space.
  "position": [-0.3236, -3.26428, 5.4160],
  // The focal length of the camera.
  "focal_length": 2691,
  // The principle point [u_0, v_0] of the camera.
  "principal_point": [1220, 1652],
  // The skew of the camera.
  "skew": 0.0,
  // The aspect ratio for the camera pixels.
  "pixel_aspect_ratio": 1.0,
  // Parameters for the radial distortion of the camera.
  "radial_distortion": [0.1004, -0.2090, 0.0],
  // Parameters for the tangential distortion of the camera.
  "tangential": [0.001109, -2.5733e-05],
  // The image width and height in pixels.
  "image_size": [2448, 3264]
}
```

### `camera-paths`
 * This directory contains test-time camera paths which can be used to render videos.
 * Each sub-directory in this path should contain a sequence of JSON files. 
 * The naming scheme does not matter, but the cameras will be sorted by their filenames.

### `rgb`
 * This directory contains images at various scales.
 * Each subdirectory should be named `${scale}x` where `${scale}` is an integer scaling factor. For example, `1x` would contain the original images while `4x` would contain images a quarter of the size.
 * We assume the images are in PNG format.
 * It is important the scaled images are integer factors of the original to allow the use of area relation when scaling the images to prevent Moiré. A simple way to do this is to simply trim the borders of the image to be divisible by the maximum scale factor you want.

### `metadata.json`
 * This defines the 'metadata' IDs used for embedding lookups.
 * Contains a dictionary of the following format:

```javascript
{
    "${item_id}": {
        // The embedding ID used to fetch the deformation latent code
        // passed to the deformation field.
        "warp_id": 0,
        // The embedding ID used to fetch the appearance latent code
        // which is passed to the second branch of the template NeRF.
        "appearance_id": 0,
        // For validation rig datasets, we use the camera ID instead
        // of the appearance ID. For example, this would be '0' for the
        // left camera and '1' for the right camera. This can potentially
        // also be used for multi-view setups as well.
        "camera_id": 0
    },
    ...
},
```
### `scene.json`
 * Contains information about how we will parse the scene.
 * See comments inline.
 
```javascript
{
  // The scale factor we will apply to the pointcloud and cameras. This is
  // important since it controls what scale is used when computing the positional
  // encoding.
  "scale": 0.0387243672920458,
  // Defines the origin of the scene. The scene will be translated such that
  // this point becomes the origin. Defined in unscaled coordinates.
  "center": [
    1.1770838526103944e-08,
    -2.58235339289195,
    -1.29117656263135
  ],
  // The distance of the near plane from the camera center in scaled coordinates.
  "near": 0.02057418950149491,
  // The distance of the far plane from the camera center in scaled coordinates.
  "far": 0.8261601717667288
}
```
 
### `dataset.json`
 * Defines the training/validation split of the dataset.
 * See inline comments:
 
```javascript
{
  // The total number of images in the dataset.
  "count": 114,
  // The total number of training images (exemplars) in the dataset.
  "num_exemplars": 57,
  // A list containins all item IDs in the dataset.
  "ids": [...],
  // A list containing all training item IDs in the dataset.
  "train_ids": [...],
  // A list containing all validation item IDs in the dataset.
  // This should be mutually exclusive with `train_ids`.
  "val_ids": [...],
}
```

### `points.npy`

 * A numpy file containing a single array of size `(N,3)` containing the background points.
 * This is required if you want to use the background regularization loss.
 
## Citing
If you find our work useful, please consider citing:
```BibTeX
@article{park2021nerfies
  author    = {Park, Keunhong 
               and Sinha, Utkarsh 
               and Barron, Jonathan T. 
               and Bouaziz, Sofien 
               and Goldman, Dan B 
               and Seitz, Steven M. 
               and Martin-Brualla, Ricardo},
  title     = {Nerfies: Deformable Neural Radiance Fields},
  journal   = {ICCV},
  year      = {2021},
}
```
