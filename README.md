# WildScenes
The repo for WildScenes - a collection of functions provided for the review process. 
The benchmark code with trained models and the training code to train new models, will be released soon.

This repo contains some scripts for viewing the dataset.

Scripts:
1) view_cloud.py (scripts/visualisation/view_cloud.py)
2) view_image.py (scripts/visualisation/view_image.py)

Please setup a conda/pip environment with the following packages:
opencv == 4.9.0,
open3d == 0.16.0,
numpy == 1.24.4,
pynput == 1.7.6,
python == 3.8.16.

Note that the exact versions are not necessary, these are just the versions that the code has been tested with.


### view_cloud.py

View cloud allows the reader to visualise our labelled 3D point clouds. Input argument options detailed in the code file. 
To run, you need to at least have --loaddir set, where --loaddir points to the full path to a wildscenes data folder on 
your computer, e.g. YOURPATH/wildscenes3d/Karawatha-1. Clouds can be viewed individually, sequentially or as a video.

### view_image.py

View image allows the reader to visualise our labelled images. Input argument options detailed in the code file. 
To run, you need to at least have --loaddir set, where --loaddir points to the full path to a wildscenes data folder on 
your computer, e.g. YOURPATH/wildscenes2d/Karawatha-1. Images can be viewed individually, sequentially or as a video.
