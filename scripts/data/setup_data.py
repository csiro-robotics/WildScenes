import os
import sys
from pathlib import Path
import argparse

from wildscenes.tools import wildscenes_converter


root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))


'''
Need to run setup_data each time start using our dataset. This file converts the raw split data into full path info
based on your download location of Wildscenes for use in training and evaluation.
'''


def main(args):
        skipflagdebug = True
        if not skipflagdebug:
                print('Setting up 3D data...')
                wildscenes_converter.create_wildscenes_info_file(
                        splitdir=args.splitdir / "opt3d",
                        dset_path=args.dataset_rootdir,
                        pkl_prefix='wildscenes3d',
                        save_path=args.savedir)
        # todo: fix camelcase bug in wildscenes folder names
        print('Setting up 2D data...')
        wildscenes_converter.create_mmseg_filestructure(
                split_dir=args.splitdir / "opt2d",
                dataset_dir=args.dataset_rootdir,
                save_path=args.savedir
        )

        print('Complete. Exiting')


if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset_rootdir', type=str, required=True, default=None)
        parser.add_argument('--splitdir', type=Path, default=root_dir / "data" / "splits")
        parser.add_argument('--savedir', type=Path, default=root_dir / "data" / "processed")
        args = parser.parse_args()

        if args.dataset_rootdir is None or not os.path.exists(args.dataset_rootdir):
                raise ValueError("No dataset directory was provided or provided directory does not exist")
        if not os.path.exists(args.dataset_rootdir / Path("Wildscenes3d")):
                raise ValueError("Could not find WildScenes dataset in the provided directory path")

        main(args)