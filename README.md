# WildScenes: A Benchmark for 2D and 3D Semantic Segmentation in Natural Environments

This is the official repo for the WildScenes dataset, which provides benchmarks for semantic segmentation in 2D and 3D.

### Abstract

Recent progress in semantic scene understanding has primarily been enabled by the availability of semantically annotated bi-modal (camera and LiDAR) datasets in urban environments. However such annotated datasets are also needed for natural, unstructured environments to enable semantic perception for important applications including search and rescue, conservation monitoring and agricultural automation. Therefore we introduce WildScenes, a bi-modal dataset including densely annotated semantic annotations in both 2D (images) and 3D (lidar point clouds), alongside 6-DoF ground truth information. We introduce benchmarks on 2D and 3D semantic segmentation and evaluate using a variety of recent deep learning techniques, to demonstrate the challenges in semantic inference in natural environments. Additionally, our dataset also provides a method for accurately projecting 2D labels into 3D, and we also provide a custom split generation algorithm to produce an optimal label distribution for training neural networks on our dataset. Finally, for the first time our dataset provides label distributions for all 3D points, enabling future applications such as label distribution learning.

### Package Requirements

Please refer to the requirements.txt file. Recommend using Python >= 3.8.


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

Will be released soon.

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
