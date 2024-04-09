import argparse
import sys
import os
import open3d as o3d
import numpy as np
from pynput import keyboard
import quaternion
import cv2 as cv
import time
from threading import Thread
from glob import glob
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import wildscenes.tools.utils as utils

root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from wildscenes.tools.utils3d import cidx_2_rgb


def load_pcd(cloud, labels=None):
    # Load points
    pcd = o3d.geometry.PointCloud()
    points = np.fromfile(cloud, dtype=np.float32).reshape(-1,3)
    index = points[:, 0] >= 0
    points = points[index]
    pcd.points = o3d.utility.Vector3dVector(points)

    if labels is not None:
        # Load colours
        labels = np.fromfile(labels, dtype = np.int32)
        # Need to remap colours to the output palette
        colors = np.array([list(cidx_2_rgb[x]) for x in labels]) / 255

        colors = colors[index]

        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd



def project_labelcloud_to_image(
    cloud_wrt_camera, cameraparams, rgb_img, label_img, image_name, vizoutfolder=None
):
    # assumes the points are in the local coordinate frame of the camera, not the global coordinate frame

    intrinsics_K, intrinsics_D = utils.get_intrinsics(
        cameraparams["centre-camera"]["intrinsics"]
    )
    imgpoints, _ = cv.projectPoints(
        cloud_wrt_camera[:, :3].astype(np.float64),
        np.zeros((1, 3)),
        np.zeros((1, 3)),
        intrinsics_K,
        np.zeros(5),
    )

    imgW, imgH = 2016, 1512

    depth_map = np.ones((imgW, imgH)) * np.inf
    dcount = 0
    for p_id, p in enumerate(imgpoints):
        # x, y = int(p[0][0]), int(p[0][1])
        x, y = int(imgpoints[p_id][0][0]), int(imgpoints[p_id][0][1])
        d = cloud_wrt_camera[p_id][2]
        if (x < 0) or (x >= imgW) or (y < 0) or (y >= imgH) or (d <= 1):  # or (d >= 30)
            continue

        # d = point_depths[p_id]
        if d >= depth_map[x, y]:
            continue
        depth_map[x, y] = d
        dcount += 1

        anno_color = label_img[y, x, :]
        # paint_color = np.array(
        #     [
        #         int(cloud_wrt_camera[p_id, 6]),
        #         int(cloud_wrt_camera[p_id, 7]),
        #         int(cloud_wrt_camera[p_id, 8]),
        #     ]
        # )
        #
        # if np.array_equal(anno_color, paint_color):
        #     drawcolor = (0, 255, 0)
        # else:
        #     drawcolor = (255, 0, 0)
        drawcolor = (255, 0, 0)

        rgb_img = cv.circle(rgb_img, (x, y), radius=2, color=drawcolor, thickness=-1)

        # rgb_img = cv.circle(
        #     rgb_img, (x, y), radius=2,
        #     color=(int(cloud_wrt_camera[p_id, 6]), int(cloud_wrt_camera[p_id, 7]), int(cloud_wrt_camera[p_id, 8])),
        #     thickness=-1
        # )

    plt.imshow(rgb_img)

    if vizoutfolder is not None:
        utils.viz_image(rgb_img, vizpath=os.path.join(vizoutfolder, "lidarproject_" + image_name))

    return dcount




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--loaddir', default=str(root_dir.parent / 'WildScenes3d' / 'K-01'),
                        help="Path to directory in WildScenes to read data, for example K-01")
    args = parser.parse_args()

    cloud_xyz = sorted(glob(os.path.join(args.loaddir, 'Clouds', '*')))
    labels = sorted(glob(os.path.join(args.loaddir, 'Labels', '*')))

    twopath = args.loaddir.replace('WildScenes3d', 'WildScenes2d')
    images = sorted(glob(os.path.join(twopath, 'image', '*')))
    labelimages = sorted(glob(os.path.join(twopath, 'label', '*')))

    image_timestamp_strings = [
        utils.timestamp_to_bag_time(t) for t in utils.get_ids_2d(Path(twopath))
    ]
    cloud_timestamp_strings = [
        utils.timestamp_to_bag_time(t) for t in utils.get_ids_3d(Path(args.loaddir))
    ]
    image_timestamp_strings = sorted(image_timestamp_strings)
    cloud_timestamp_strings = sorted(cloud_timestamp_strings)

    image_timestamps = utils.convert_ts_to_float(image_timestamp_strings)
    cloud_timestamps = utils.convert_ts_to_float(cloud_timestamp_strings)

    idx = 200 # temp
    thistimestamp = image_timestamps[idx]

    # search for nearest cloud for this image:
    bestdiff = 1e10
    bestcloudidx = -1
    for cloudidx, cloudts in enumerate(cloud_timestamps):
        tdiff = np.abs(thistimestamp - cloudts)
        if tdiff < bestdiff:
            bestdiff = tdiff
            bestcloudidx = cloudidx
    if bestdiff > 5:
        raise ValueError("For this image idx there is no suitable point cloud anywhere near this image")

    pcd = load_pcd(cloud_xyz[bestcloudidx], labels[bestcloudidx])

    local2global = np.load(os.path.join(args.loaddir, "align.npy"))

    # load intrinsics and extrinsics
    cameraparams = utils.read_yaml_params(os.path.join(twopath, 'camera_calibration.yaml'))

    # get extrinsics
    camextdata = cameraparams['centre-camera']['extrinsics']

    extT = utils.get_extrinsics_yaml(camextdata)

    # load pose information
    poses2d = pd.read_csv(os.path.join(twopath, 'poses2d.csv'), sep=' ').sort_index()
    poses2d = poses2d.rename(columns={poses2d.columns[0]: "ts"})
    poses2d = poses2d.set_index('ts')

    image_timestamps_datetime = pd.to_datetime(
        [float(ts) for ts in image_timestamp_strings], unit="s"
    )

    this2dpose = poses2d.loc[str(image_timestamps_datetime[idx])]

    q = quaternion.quaternion(this2dpose.qw, this2dpose.qx, this2dpose.qy, this2dpose.qz)
    Rm = quaternion.as_rotation_matrix(q)
    T = np.zeros((4, 4))
    T[:3, :3] = Rm
    T[3, 3] = 1.0
    T[:3, 3] = np.array([this2dpose.x, this2dpose.y, this2dpose.z])

    cloud_wrt_camera = pcd.transform(np.linalg.inv(extT)) # this is the lidar points in the local reference frame.

    # If want to convert points into a global ref frame:
    # test = cloud_wrt_camera.transform(T)

    viewcloud = np.asarray(cloud_wrt_camera.points)

    # pcd_transformed = pcd.transform(global2local)
    # viewcloud = np.asarray(pcd_transformed.points)
    # cloud_wrt_camera = pcd.transform(np.linalg.inv(T))
    # viewcloud = np.asarray(cloud_wrt_camera.points)

    # load raw image and labelimg
    raw_rgb_img = utils.read_image_cv(images[idx])
    label_img = utils.read_image_cv(labelimages[idx])

    print('')

    successcount = project_labelcloud_to_image(
        viewcloud.astype(np.float64),
        cameraparams,
        raw_rgb_img,
        label_img,
        image_name=image_timestamp_strings[idx],
        vizoutfolder=None,
    )
    print("Number of lidar points projected: ", successcount)


