from typing import Any

import clip
import pandas as pd
import torch
from fire import Fire
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import CustomDataset, create_image_list_based_on


def load_model(name: str = "ViT-B/32") -> tuple[Any, Any, str]:
    """
    Load a model
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(name, device=device)
    return model, preprocess, device


model, preprocess, device = load_model()


def generate_features_for_list(img_list: str):
    """
    Generate features for a list of images
    :param img_list:
    :return: None
    """
    dataset = CustomDataset(img_list, transform=preprocess)
    df = generate_features(dataset)
    df.to_csv('features.csv', index=False)


def generate_features_for_folder(folder: str):
    """
    Generate features for a folder
    :param folder:
    :return:
    """
    file_name = create_image_list_based_on(folder)
    generate_features_for_list(file_name)


def generate_features(dataset, batch_size=100) -> pd.DataFrame:
    """
    Generate features for a dataset
    :param dataset:
    :param batch_size:
    :return: pd.DataFrame
    """
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
            all_names += names
            all_img_paths += img_paths

    data = dict(names=all_names, features=torch.cat(all_features).cpu().numpy().tolist(), img_paths=all_img_paths)
    df = pd.DataFrame(data=data)
    return df


if __name__ == '__main__':
    Fire(generate_features_for_folder)
