import clip
import pandas as pd
import torch
from fire import Fire
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import CustomDataset, create_image_list_based_on


class FeatureProcessor:
    def __init__(self, name: str = "ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(name, device=self.device)

    def generate_features_for_folder(self, folder: str):
        """
        Generate features for a folder
        :param folder:
        :return:
        """
        file_name = '../artifacts/image_list.txt'
        create_image_list_based_on(folder, file_name=file_name)
        self.generate_features_for_list(file_name)

    def generate_features_for_list(self, img_list: str):
        """
        Generate features for a list of images
        :param img_list:
        :return: None
        """
        dataset = CustomDataset(img_list, transform=self.preprocess)
        df = self.generate_features(dataset)
        df.to_csv('features2.csv', index=False)

    def generate_features(self, dataset, batch_size=100) -> pd.DataFrame:
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
                images = entity['image'].to(self.device)
                names = entity['name']
                img_paths = entity['img_path']
                features = self.model.encode_image(images)

                all_features.append(features)
                all_names += names
                all_img_paths += img_paths

        data = dict(names=all_names, features=torch.cat(all_features).cpu().numpy().tolist(), img_paths=all_img_paths)
        df = pd.DataFrame(data=data)
        return df


if __name__ == '__main__':
    Fire(FeatureProcessor, name="features")
