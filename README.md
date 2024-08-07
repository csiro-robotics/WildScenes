# WildScenes: A Benchmark for 2D and 3D Semantic Segmentation in Natural Environments

This is the official repo for the WildScenes dataset, which provides benchmarks for semantic segmentation in 2D and 3D. Training is performed using the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) and [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) toolboxs. We thank and acknowledge the contributions of these toolboxes.

### Abstract

Recent progress in semantic scene understanding has primarily been enabled by the availability of semantically annotated bi-modal (camera and LiDAR) datasets in urban environments. However such annotated datasets are also needed for natural, unstructured environments to enable semantic perception for important applications including search and rescue, conservation monitoring and agricultural automation. Therefore we introduce WildScenes, a bi-modal dataset including densely annotated semantic annotations in both 2D (images) and 3D (lidar point clouds), alongside 6-DoF ground truth information. We introduce benchmarks on 2D and 3D semantic segmentation and evaluate using a variety of recent deep learning techniques, to demonstrate the challenges in semantic inference in natural environments. Additionally, our dataset also provides a method for accurately projecting 2D labels into 3D, and we also provide a custom split generation algorithm to produce an optimal label distribution for training neural networks on our dataset. Finally, for the first time our dataset provides label distributions for all 3D points, enabling future applications such as label distribution learning.

### Package Requirements

Please refer to the installation.md file for detailed instructions. The full list of required packages are also detailed in the requirements.txt file. Requires Python >= 3.8 and recommend Python = 3.10.

### Dataset Setup

The dataset has been released and can be accessed at: https://data.csiro.au/collection/csiro:61541

To download the dataset, it is recommended to use an S3 client. Please select the download method "Download files via S3 Client". A command line tool such as rclone could be used to perform the download.

After downloading the WildScenes dataset to a directory of your choice, before you can use the dataset with this repository, the setup_data.py script needs to be run (scripts/data/setup_data.py).

#### setup_data.py

This script creates full paths to individual files in the WildScenes dataset, based on your dataset save location and the split details in /data/splits.
The input argument --dataset_rootdir is required, and must be the full path to the root directory of WildScenes. 
This script will produce pickle files for 3D training and evaluation, and setup symlinks for 2D training and evaluation.

### Visualization Scripts

This repo includes some scripts for viewing the dataset.

Scripts:
1) view_cloud.py (scripts/visualisation/view_cloud.py)
2) view_image.py (scripts/visualisation/view_image.py)

#### view_cloud.py

View cloud allows the reader to visualise our labelled 3D point clouds. 
To run this script, you need to include the --loaddir argument, where --loaddir points to the full path to a wildscenes data folder on 
your computer, e.g. YOURPATH/WildScenes/WildScenes3d/K-01. Clouds can be viewed individually, sequentially or as a video.
Default is to display a single cloud, then the program terminates after the user closes the visualization window. 
The visualization window can be closed using the "q" key.

Input arguments include:

- --loaddir: define the full path to a WildScenes3d subfolder.
- --viewpoint: Options: [BEV, FPV]. Default: BEV. BEV displays the point cloud from a birds-eye perspective, FPV displays the cloud from a first person perspective.
- --loadidx: sets the index of the specific cloud you want to view (or specific cloud to start the video from). Defaults to a random cloud.
- --sequential: rather than viewing a single point cloud, sequential will iteratively load each cloud in a folder. Each subsequent cloud is loaded after the user closes the current visualizer window using the "q" key. This input argument will override the --video argument. To terminate the program, please use Control-C.
- --video: loads the point clouds sequentially in a video, with no user input required. The playback speed is set using --videospeed. To terminate the video, please use Control-C.

#### view_image.py

View image allows the reader to visualise our labelled images. 
To run this script, you need to include the --loaddir argument, where --loaddir points to the full path to a wildscenes data folder on 
your computer, e.g. YOURPATH/WildScenes/WildScenes2d/V-01. Images can be viewed individually, sequentially or as a video.
Default is to display a single image, then the program terminates after the user closes the visualization window.

Input arguments include:

- --loaddir: define the full path to a WildScenes3d subfolder.
- --loadidx: sets the index of the specific image you want to view (or specific image to start the video from). Defaults to a random image.
- --sequential: rather than viewing a single image, sequential will iteratively load each image in a folder. Each subsequent image is displayed after the user closes the current visualizer window using the "q" key. This input argument will override the --video argument. To terminate the program, please use Control-C.
- --video: loads the images sequentially in a video, with no user input required. The playback speed is set using --videospeed. To terminate the video, please use Control-C.
- --raw: by default, only the benchmark annotated classes are shown. To display the raw labels including merged and excluded classes, please use this argument.

### Trained Models

Will be released in the next few weeks.

### Training code

Update August 2024: release of version two of our WildScenes training code is currently in progress and will be finished updating in the coming days.

All training and eval scripts are located in the directory scripts/benchmark. We use the open source mmsegmentation and mmdetection3d repositories for training and evaluating semantic segmentation on WildScenes.

Important: before running any scripts, please first modify the "data_root" path in wildscenes/configs/base/datasets/wildscenes.py
and in wildscenes/configs3d/base/datasets/wildscenes.py.
This needs to point to the path to the root directory where you save the repository.

#### 2D Training

Using mmsegmentation, 2D models can be trained using scripts/benchmark/train2d.py. We have released pre-configured scripts
for running training on some existing 2D semantic segmentation methods.

Please note that your own 2D training may not exactly reproduce the results in the paper, since we are still in the process of refactoring our training code.
For the best numbers, please follow the training configurations provided in the paper. All provided scripts are still WIP.
We have now set a fixed seed (seed=0). The seed can be manually changed by editing wildscenes/configs/base/datasets/wildscenes.py. 

#### 3D Training

Using mmdetection3d, 3D models can be trained using scripts/benchmark/train2d.py. We have released pre-configured scripts
for running training on some existing 3D semantic segmentation methods.
Please note that your own 3D training may not exactly reproduce the results in the paper, since we are still in the process of refactoring our training code.

### Evaluation code

#### 2D Evaluation

2D models can be evaluated using scripts/benchmark/eval2d.py.

#### 3D Evaluation

3D models can be evaluated using scripts/benchmark/eval3d.py.

### Citation
<p>
If you find this dataset helpful for your research, please cite our paper using the following reference:

```
@misc{vidanapathirana2023wildscenes,
      title={WildScenes: A Benchmark for 2D and 3D Semantic Segmentation in Large-scale Natural Environments}, 
      author={Kavisha Vidanapathirana and Joshua Knights and Stephen Hausler and Mark Cox and Milad Ramezani and Jason Jooste and Ethan Griffiths and Shaheer Mohamed and Sridha Sridharan and Clinton Fookes and Peyman Moghadam},
      year={2023},
      eprint={2312.15364},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

### Acknowledgements

We kindly thank the authors of [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) and [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) for open sourcing their methods and training scripts.