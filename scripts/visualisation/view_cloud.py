import argparse
import sys
import os
import open3d as o3d
import numpy as np
import time
from glob import glob
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from wildscenes.tools.utils3d import cidx_2_rgb


'''
view_cloud.py

This script allows for viewing 3D labeled point clouds. Input argument options provided are:
--loaddir
    Set to full path to a Wildscenes3d subfolder, e.g. V-01
--viewpoint
    Options: BEV, FPV. Displays the cloud from either a birds eye view or perspective view
--loadidx
    Specify which specific cloud to load, or from which cloud to begin viewing a sequence of clouds
--sequential
    Default is to load a single cloud then exit. Sequential loads clouds one after another. Press q to go to the next
        cloud.
--video
    Plays the clouds sequentially in a continuous video
--videospeed
    Defines the video playback speed, lower is faster
--update_viewpoint_jsons
    Allows user to change the default viewpoints for FPV or BEV modes
    
In default and sequential modes, when viewing a cloud, the viewer can be exited by pressing ESC
'''


def load_pcd(cloud, labels, fpv=False):
    # Load points
    pcd = o3d.geometry.PointCloud()
    points = np.fromfile(cloud, dtype=np.float32).reshape(-1,3)
    # Load colours
    labels = np.fromfile(labels, dtype = np.int32)
    # Need to remap colours to the output palette
    colors = np.array([list(cidx_2_rgb[x]) for x in labels]) / 255

    if fpv:
        index = points[:, 0] >= 0
        points = points[index]
        colors = colors[index]

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def create_axis_arrow(size=3.0):
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[-0, -0, -0])
    return mesh_frame


def view_cloud_video(pcd, coord, view_params, render_param_file, firstloop=True, video=False, videospeed=0.1):
    vis.add_geometry(pcd)
    vis.add_geometry(coord)
    vis.get_render_option().load_from_json(render_param_file)
    ctr.convert_from_pinhole_camera_parameters(view_params, True)
    vis.update_geometry(pcd)
    vis.update_geometry(coord)

    vis.poll_events()
    vis.update_renderer()
    time.sleep(videospeed)

    vis.remove_geometry(pcd)
    vis.remove_geometry(coord)


def save_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1080, height=1080)
    vis.add_geometry(pcd)
    vis.add_geometry(create_axis_arrow())
    vis.run() # adjust and press 'q'
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--loaddir', default=str(root_dir.parent / 'WildScenes' / 'WildScenes3d' / 'K-01'),
                        help="Path to directory in WildScenes to read data, for example K-01")
    parser.add_argument('--viewpoint', choices=['BEV', 'FPV'], default='BEV',
                        help="Choice of viewpoints for rendering the labeled 3D point clouds, either birds eye view or first person view")
    parser.add_argument('--loadidx', default=-1, type=int,
                        help="Specify which cloud index you want to view. Defaults to a random cloud from the traverse")
    parser.add_argument('--sequential', default=False, action='store_true',
                        help="Iteratively view all clouds in a traverse, starting from 0 or loadidx")
    parser.add_argument('--video', default=False, action='store_true',
                        help="View the clouds as a continuous video, starting from 0 or loadidx")
    parser.add_argument('--videospeed', default=0.5, type=float,
                        help='Video playback speed, lower is faster')
    parser.add_argument('--update_viewpoint_jsons', default=False, action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.loaddir):
        raise ValueError('Please set the loaddir argument to a subfolder (e.g. K-01) inside the WildScenes dataset. '
                         'This path (full path) should be set to whereever you downloaded the WildScenes dataset to.')

    cloud_xyz = sorted(glob(os.path.join(args.loaddir, 'Clouds', '*')))
    labels = sorted(glob(os.path.join(args.loaddir, 'Labels', '*')))

    if args.loadidx >= len(labels):
        raise ValueError('Your loadidx is greater than the number of clouds in this traverse')

    if args.update_viewpoint_jsons: # if want to resave viewpoint settings for a custom viewpoint:
        dir_path = os.path.join(root_dir, 'wildscenes', 'configs')
        if args.viewpoint == 'BEV':
            pcd = load_pcd(cloud_xyz[args.loadidx], labels[args.loadidx])
            save_view_point(pcd, os.path.join(dir_path, 'viewpoint_bev.json'))
        else:
            pcd = load_pcd(cloud_xyz[args.loadidx], labels[args.loadidx], fpv=True)
            save_view_point(pcd, os.path.join(dir_path, 'viewpoint_fpv.json'))

    if args.viewpoint == 'BEV':
        view_params = o3d.io.read_pinhole_camera_parameters(
            os.path.join(root_dir, 'wildscenes', 'configs', 'viewpoint_bev.json'))
        render_param_file = os.path.join(root_dir, 'wildscenes', 'configs', 'render_bev.json')
    else:
        view_params = o3d.io.read_pinhole_camera_parameters(
            os.path.join(root_dir, 'wildscenes', 'configs', 'viewpoint_fpv.json'))
        render_param_file = os.path.join(root_dir, 'wildscenes', 'configs', 'render_fpv.json')

    if args.sequential:
        if args.loadidx == -1:
            args.loadidx = 0
        for idx in range(args.loadidx, len(labels)):
            if args.viewpoint == 'BEV':
                pcd = load_pcd(cloud_xyz[idx], labels[idx])
                o3d.visualization.draw_geometries([pcd],
                                                  zoom=0.3412,
                                                  front=[-0.268, 1.51238, 66.3],
                                                  lookat=[2.6172, 2.0475, 1.532],
                                                  up=[-0.0694, -0.9768, 0.2024])
            else:
                pcd = load_pcd(cloud_xyz[idx], labels[idx], fpv=True)
                o3d.visualization.draw_geometries([pcd],
                                                  zoom=0.3412,
                                                  front=[-10, 0, 0],
                                                  lookat=[0, -10, 10],
                                                  up=[0, 0, 1])
    elif args.video:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1080, height=1080, visible=True)
        ctr = vis.get_view_control()
        coord = create_axis_arrow(1)
        if args.loadidx == -1:
            args.loadidx = 0
        for idx in range(args.loadidx, len(labels)):
            if args.viewpoint == 'BEV':
                pcd = load_pcd(cloud_xyz[idx], labels[idx])
            else:
                pcd = load_pcd(cloud_xyz[idx], labels[idx], fpv=True)
            view_cloud_video(pcd, coord, view_params, render_param_file, video=True, videospeed=args.videospeed)

    else:
        if args.loadidx == -1:
            args.loadidx = np.random.randint(len(labels))
        if args.viewpoint == 'BEV':
            pcd = load_pcd(cloud_xyz[args.loadidx], labels[args.loadidx])
            o3d.visualization.draw_geometries([pcd],
                                              zoom=0.3412,
                                              front=[-0.268, 1.51238, 66.3],
                                              lookat=[2.6172, 2.0475, 1.532],
                                              up=[-0.0694, -0.9768, 0.2024])
        else:
            pcd = load_pcd(cloud_xyz[args.loadidx], labels[args.loadidx], fpv=True)
            o3d.visualization.draw_geometries([pcd],
                                              zoom=0.3412,
                                              front=[-10, 0, 0],
                                              lookat=[0, -10, 10],
                                              up=[0, 0, 1])

    print('EXITING')