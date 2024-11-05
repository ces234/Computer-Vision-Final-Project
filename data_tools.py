import os
from pathlib import Path
from argparse import ArgumentParser

# DATA_DIR = Path('marine_debris_data')

def remove_file_prefix(dir_path: Path, prefix: str):
    for filename in os.listdir(str(dir_path)):
        if filename.startswith(prefix):
            new_filename = filename[len(prefix):]
            old_file = dir_path / filename
            new_file = dir_path / new_filename
            os.rename(str(old_file), str(new_file))
            # print(f'Renamed "{old_file}" to "{new_file}"')

if __name__ == '__main__':
    parser = ArgumentParser(description='Process data to get it ready for use in training')
    parser.add_argument('dir', type=str, help='The local path to the directory containing "annotations" and "images" folders')
    parser.add_argument('--rename', action='store_true', help='If provided, rename the files by removing the "marine-debris-aris3k-" prefix')

    args = parser.parse_args()

    data_dir = Path(args.dir)

    if args.rename:
        print('Renaming annotations...')
        remove_file_prefix(data_dir / 'annotations', 'marine-debris-aris3k-')
        print('Renaming images...')
        remove_file_prefix(data_dir / 'images', 'marine-debris-aris3k-')
        print('Done')