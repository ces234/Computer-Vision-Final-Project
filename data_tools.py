import os
from pathlib import Path
from argparse import ArgumentParser
from PIL import Image
from xml.etree import ElementTree
from xml.etree.ElementTree import Element

# DATA_DIR = Path('marine_debris_data')

IMG_WIDTH = 128
IMG_HEIGHT = 128

def remove_file_prefix(dir_path: Path, prefix: str):
    for filename in os.listdir(str(dir_path)):
        if filename.startswith(prefix):
            new_filename = filename[len(prefix):]
            old_file = dir_path / filename
            new_file = dir_path / new_filename
            os.rename(str(old_file), str(new_file))
            # print(f'Renamed "{old_file}" to "{new_file}"')

def scale_images(dir_path: Path, target_width: int, target_height: int):
    filenames = os.listdir(str(dir_path))
    total_files = len(filenames)
    for i, filename in enumerate(filenames):
        print(f'Resizing image {filename}\t({i + 1} / {total_files})')
        img = Image.open(dir_path / filename)
        resized_img = img.resize((target_width, target_height))
        resized_img.save(dir_path / filename)

def scale_xml(dir_path: Path, target_width: int, target_height: int):
    filenames = os.listdir(str(dir_path))
    total_files = len(filenames)
    for i, filename in enumerate(filenames):
        print(f'Resizing XML {filename}\t({i + 1} / {total_files})')

        file_path = str(dir_path / filename)

        tree = ElementTree.parse(file_path)
        root = tree.getroot()

        size_el = root.find('size')
        width_el = size_el.find('width')
        height_el = size_el.find('height')
        current_width = int(width_el.text)
        current_height = int(height_el.text)

        width_el.text = str(target_width)
        height_el.text = str(target_height)

        for object in root.findall('object'):
            bounding_box = object.find('bndbox')
            x_el = bounding_box.find('x')
            w_el = bounding_box.find('w')
            y_el = bounding_box.find('y')
            h_el = bounding_box.find('h')
            current_x = int(x_el.text)
            current_w = int(w_el.text)
            current_y = int(y_el.text)
            current_h = int(h_el.text)

            x_el.text = str(round((current_x / current_width) * target_width))
            w_el.text = str(round((current_w / current_width) * target_width))
            y_el.text = str(round((current_y / current_height) * target_height))
            h_el.text = str(round((current_h / current_height) * target_height))

        tree.write(file_path)

        print(current_width, current_height)

if __name__ == '__main__':
    parser = ArgumentParser(description='Process data to get it ready for use in training')
    parser.add_argument('dir', type=str, help='The local path to the directory containing "annotations" and "images" folders')
    parser.add_argument('--rename', action='store_true', help='If provided, rename the files by removing the "marine-debris-aris3k-" prefix')
    parser.add_argument('--scale', action='store_true', help='If provided, scale XML & images so images are 128x128')

    args = parser.parse_args()

    data_dir = Path(args.dir)

    if args.rename:
        print('Renaming annotations...')
        remove_file_prefix(data_dir / 'annotations', 'marine-debris-aris3k-')
        print('Renaming images...')
        remove_file_prefix(data_dir / 'images', 'marine-debris-aris3k-')
        print('Done')

    if args.scale:
        print('Scaling images...')
        # scale_images(data_dir / 'images', IMG_WIDTH, IMG_HEIGHT)
        print('Scaling XML...')
        scale_xml(data_dir / 'annotations', IMG_WIDTH, IMG_HEIGHT)