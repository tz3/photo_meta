import logging
import os
from pathlib import Path

import pyheif
import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', 'HEIC']


def create_image_list_based_on(folder: str, file_name: str = '../artifacts/image_list.txt') -> list[str]:
    """"
    Create a list of images based on a folder
    :param file_name:
    :param folder:
    :return: list[str]
    """
    image_list = []
    for path, _, files in os.walk(folder):
        for filename in files:
            if any(filename.lower().endswith(ext) for ext in IMG_EXTENSIONS):
                image_list.append(os.path.join(path, filename))
    write_list_to_file(image_list, file_name)
    return image_list


def write_list_to_file(image_list, file_name):
    """"
    Write a list of images to a file
    """
    with open(file_name, 'w') as f:
        for image in image_list:
            f.write(image + '\n')


class CustomDataset(Dataset):

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(csv_file, 'r') as f:
            self.image_list = f.read().splitlines()
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        try:
            if torch.is_tensor(idx):
                idx = idx.tolist()

            img_path = self.image_list[idx]
            img_name = Path(img_path).name
            # Check if image format is .heic
            if img_path.lower().endswith('heic'):
                heif_file = pyheif.read(img_path)
                image = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw", heif_file.mode,
                                        heif_file.stride)
            else:  # assume "normal" image that pillow can open
                image = Image.open(img_path)
            # image = Image.open(img_path)
            sample = {'image': image, 'name': img_name, "img_path": img_path}

            if self.transform:
                sample['image'] = self.transform(image)
        except Exception as e:
            logging.error(e)
            logging.error(img_name)
            assert e

        return sample


def read_image(img_path) -> Image:
    """
    Read an image
    :param img_path:
    :return: Image
    """
    if img_path.lower().endswith('heic'):
        heif_file = pyheif.read(img_path)
        image = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw", heif_file.mode,
                                heif_file.stride)
    else:  # assume "normal" image that pillow can open
        image = Image.open(img_path)
    return image
