import os
from pathlib import Path
import pyheif


import clip
import pandas as pd
import torch
from PIL import Image
from fire import Fire
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# image_path = '/Users/arrtz3/code/home/PhotoMeta/19700101_030816.jpg'

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', 'HEIC']


def get_list_of_images(folder: str):
    image_list = []
    for path, _, files in os.walk(folder):
        for filename in files:
            if any(filename.endswith(ext) for ext in IMG_EXTENSIONS):
                image_list.append(os.path.join(path, filename))
    with open('image_list.txt', 'w') as f:
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
            print(e)
            print(img_name)
            assert e

        return sample


def generate_features_for(img_list: str):
    dataset = CustomDataset(img_list, transform=preprocess)
    get_features(dataset)


def get_features(dataset, batch_size=100):
    all_features = []
    all_img_paths = []
    all_names = []

    with torch.no_grad():
        for entity in tqdm(DataLoader(dataset, batch_size=batch_size)):
            images = entity['image'].to(device)
            names = entity['name']
            img_paths = entity['img_path']
            features = model.encode_image(images)

            all_features.append(features)
            all_names.append(names)
            all_img_paths.append(img_paths)

    all_names = [item for sublist in all_names for item in sublist]
    all_img_paths = [item for sublist in all_img_paths for item in sublist]
    data = dict(names=all_names, features=torch.cat(all_features).cpu().numpy().tolist(), img_paths=all_img_paths)
    df = pd.DataFrame(data=data)
    df.to_csv('features.csv', index=False)
    return df


if __name__ == '__main__':
    Fire(generate_features_for)
