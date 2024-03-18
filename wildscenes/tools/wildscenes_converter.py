import os
from pathlib import Path
import pandas as pd
import mmengine
from tqdm import tqdm


def get_info(csv_path: Path, dset_path: Path):
    df = pd.read_csv(csv_path, dtype=str)

    df['lidar_path'] = df['lidar_path'].str.replace('WildScenes3d', str(dset_path) + '/WildScenes3d')
    df['label_path'] = df['label_path'].str.replace('WildScenes3d', str(dset_path) + '/WildScenes3d')
    df['hist_path'] = df['hist_path'].str.replace('WildScenes3d', str(dset_path) + '/WildScenes3d')

    data_list = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        assert os.path.exists(row['lidar_path']), f"ERROR: {row['lidar_path']}"
        assert os.path.exists(row['label_path']), f"ERROR: {row['label_path']}"
        assert os.path.exists(row['hist_path']), f"ERROR: {row['hist_path']}"
        data_item = {
            'lidar_points': {
                'lidar_path': row['lidar_path'],
                'num_pts_feats': 3
            },
            'pts_semantic_mask_path': row['label_path'],
            'hist_path': row['hist_path'],
            'sample_id': row['id']
        }
        data_list.append(data_item)
    info = dict(metainfo={'DATASET': 'WildScenes'}, data_list=data_list)
    return info


def get_wildscenes_info(splitdir: Path, dset_path: Path):
    train_csv = splitdir / 'train.csv'
    test_csv = splitdir / 'test.csv'
    val_csv = splitdir / 'val.csv'

    train_info = get_info(train_csv, dset_path)
    test_info = get_info(test_csv, dset_path)
    val_info = get_info(val_csv, dset_path)

    return train_info, test_info, val_info


def create_wildscenes_info_file(splitdir: Path, dset_path: str, pkl_prefix: str, save_path: Path):
    """Create info file of WildScenes dataset (similar style as SemanticKITTI dataset) - for 3D.

    Directly generate info file without raw data.

    Args:
        splitdir (Path): Path to the split files dir to use.
        dset_path (str): Path to where you downloaded and saved the WildScenes dataset.
        pkl_prefix (str): Prefix of the info file to be generated.
        save_path (Path): Path to save the info file.
    """
    print('Generate 3D info')
    dset_path = Path(dset_path)

    save_path.mkdir(parents=True, exist_ok=True)

    train, test, val = get_wildscenes_info(splitdir, dset_path)

    filename = save_path / f'{pkl_prefix}_infos_train.pkl'
    print(f'WildScenes info train file is saved to {filename}')
    mmengine.dump(train, filename)

    filename = save_path / f'{pkl_prefix}_infos_val.pkl'
    print(f'WildScenes info val file is saved to {filename}')
    mmengine.dump(val, filename)

    filename = save_path / f'{pkl_prefix}_infos_test.pkl'
    print(f'WildScenes info test file is saved to {filename}')
    mmengine.dump(test, filename)


def create_split_subdir(out_dir: Path, split_file: Path, dset_path: Path):
    """Create a series of directories with symlinks to the original image files based on the split file.

    dataset_dir: the directory in which the split dirs will reside
    split_file: the path to the split file

    NOTE: It would be better to take in the df directly, but then we would need to maintain conistency with the relative paths in the df (which are relative to the file path.

    NOTE: This is very slow. A bash script would be faster. Convert to absolute paths and then use -r
    """
    # todo: need to also convert the rel paths in split_file to point to the full path to the dataset itself
    out_dir = out_dir / split_file.stem
    (out_dir / "image").mkdir(parents=True, exist_ok=True)
    (out_dir / "indexLabel").mkdir(parents=True, exist_ok=True)
    # Read in the split file
    split_dir = split_file.parent  # Paths in the df are relative to this dir
    split_df = pd.read_csv(split_file, index_col="id")
    for ID, row in tqdm(split_df.iterrows()):
        img_path, label_path = row
        # Get abs paths for the images
        img_path = (dset_path / img_path).resolve() # ???
        label_path = (dset_path / label_path).resolve()
        assert (
            img_path.exists() and label_path.exists()
        ), f"Paths to image {img_path} or {label_path} does not exist"
        # Remove existing symlinks
        dest_img = out_dir / "image" / (ID + ".png")
        dest_label = out_dir / "indexLabel" / (ID + ".png")
        if (
            dest_img.exists() or dest_img.is_symlink()
        ):  # bad symlinks don't "exist" but still need to be removed
            assert (
                dest_img.is_symlink()
            ), f"File {dest_img} should be a symlink but isn't"
            os.remove(dest_img)
        if dest_label.exists():
            assert (
                dest_label.is_symlink()
            ), f"File {dest_label} should be a symlink but isn't"
            os.remove(dest_label)
        # Get paths relative to output dir
        img_out_rel = os.path.relpath(img_path, dest_img.parent)
        label_out_rel = os.path.relpath(label_path, dest_label.parent)
        # Create the symlinks
        dest_img.symlink_to(img_out_rel)
        dest_label.symlink_to(label_out_rel)

'''
mmsegmentation (2D) requires the following file format:

dsetname
    test
        image
        indexLabel
    train
        image
        indexLabel
    val
        image
        indexLabel

Files within directories are symlinks to the original save location of the WildScenes dataset
'''
def create_mmseg_filestructure(split_dir: Path, dataset_dir: str, save_path: Path):
    """Convert a set of split files to an mmseg-style hierarchical directory structure

    Args:
        splitdir (Path): Path to the split files dir to use.
        dataset_dir (str): Path to where you downloaded and saved the WildScenes dataset.
        save_path (Path): Path to save the generated symlinks for mmsegmentation2d.
    """
    dataset_dir = Path(dataset_dir)
    if not split_dir.exists():
        raise ValueError(f"Split dir {split_dir} does not exist")
    # Create the directory structure and symlinks
    split_files = split_dir.glob("*")
    for split_file in split_files:
        if split_file.stem not in ["test", "train", "val"]:
            if not split_file.stem.startswith("ex_"):
                print(
                    f"Warning: file ignored in split dir that doesn't belong to test/train/val {split_file.name}"
                )
            continue
        create_split_subdir(save_path, split_file, dataset_dir)
    print('done')