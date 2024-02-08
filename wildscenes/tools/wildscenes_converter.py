import os
from pathlib import Path
import pandas as pd
import mmengine
from tqdm import tqdm


def get_info(csv_path: Path, dset_path: Path):
    df = pd.read_csv(csv_path, dtype=str)

    df['lidar_path'] = df['lidar_path'].str.replace('/Wildscenes3d', str(dset_path) + '/Wildscenes3d')
    df['label_path'] = df['label_path'].str.replace('/Wildscenes3d', str(dset_path) + '/Wildscenes3d')
    df['hist_path'] = df['hist_path'].str.replace('/Wildscenes3d', str(dset_path) + '/Wildscenes3d')

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


def create_wildscenes_info_file(splitdir: str, dset_path: str, pkl_prefix: str, save_path: str):
    """Create info file of WildScenes dataset (similar style as SemanticKITTI dataset) - for 3D.

    Directly generate info file without raw data.

    Args:
        splitdir (str): Path to the split files dir to use.
        dset_path (str): Path to where you downloaded and saved the WildScenes dataset.
        pkl_prefix (str): Prefix of the info file to be generated.
        save_path (str): Path to save the info file.
    """
    print('Generate 3D info')
    save_path = Path(save_path)
    splitdir = Path(splitdir)
    dset_path = Path(dset_path)

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