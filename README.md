# WildScenes: A Benchmark for 2D and 3D Semantic Segmentation in Natural Environments

This is the official repo for the WildScenes dataset, which provides benchmarks for semantic segmentation in 2D and 3D. Training is performed using the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) and [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) toolboxs. We thank and acknowledge the contributions of these toolboxes.

### Abstract

Recent progress in semantic scene understanding has primarily been enabled by the availability of semantically annotated bi-modal (camera and LiDAR) datasets in urban environments. However such annotated datasets are also needed for natural, unstructured environments to enable semantic perception for important applications including search and rescue, conservation monitoring and agricultural automation. Therefore we introduce WildScenes, a bi-modal dataset including densely annotated semantic annotations in both 2D (images) and 3D (lidar point clouds), alongside 6-DoF ground truth information. We introduce benchmarks on 2D and 3D semantic segmentation and evaluate using a variety of recent deep learning techniques, to demonstrate the challenges in semantic inference in natural environments. Additionally, our dataset also provides a method for accurately projecting 2D labels into 3D, and we also provide a custom split generation algorithm to produce an optimal label distribution for training neural networks on our dataset. Finally, for the first time our dataset provides label distributions for all 3D points, enabling future applications such as label distribution learning.

### Package Requirements

Please refer to the installation.md file for detailed instructions. The full list of required packages are also detailed in the requirements.txt file. Requires Python >= 3.8 and recommend Python = 3.10.

### Dataset Setup

Dataset has been released and can be accessed at: https://data.csiro.au/collection/csiro:61541

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

View cloud allows the reader to visualise our labelled 3D point clouds. Input argument options detailed in the code file. 
To run, you need to at least have --loaddir set, where --loaddir points to the full path to a wildscenes data folder on 
your computer, e.g. YOURPATH/wildscenes3d/Karawatha-1. Clouds can be viewed individually, sequentially or as a video.

#### view_image.py

View image allows the reader to visualise our labelled images. Input argument options detailed in the code file. 
To run, you need to at least have --loaddir set, where --loaddir points to the full path to a wildscenes data folder on 
your computer, e.g. YOURPATH/wildscenes2d/Karawatha-1. Images can be viewed individually, sequentially or as a video.

### Trained Models

Will be released soon.

### Training code

All training and eval scripts are located in the directory scripts/benchmark. We use the open source mmsegmentation and mmdetection3d repositories for training and evaluating semantic segmentation on WildScenes.

#### 2D Training

2D Training has not been fully released yet, please don't raise issues yet as this section of the code is still being debugged.

Using mmsegmentation, 2D models can be trained using train2d.py. 
We will release pre-configured bash scripts for training soon.

Additional documentation will be released soon.

Please note that the seed is not set by default, therefore your own 2D training will not be able to exactly reproduce the results in the paper.

#### 3D Training

3D Training is still under development and will be released soon.

### Evaluation code

Evaluation can be performed using eval2d.py and eval3d.py. Documentation and bash scripts will be released soon.

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