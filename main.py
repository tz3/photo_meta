from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS
from fire import Fire
from tqdm import tqdm


def get_file_extension(file_path: Path):
    """
    Return the file extension of a file
    """
    return file_path.suffix


def get_list_image_paths(folder_path: Path):
    """
    Return a list of image paths in a folder
    """
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    # return [str(path) for path in folder_path.glob('**/*.jpg')]
    return [str(path) for path in folder_path.glob('**/*.*')]


def read_from_pandas_cache(cvs_file: str):
    df = None
    if Path(cvs_file).exists():
        df = pd.read_csv(cvs_file)
    return df


def main(image_folder: str = '/Users/arrtz3/Pictures/Photos Library.photoslibrary/originals/',
         cvs_output_file: str = 'image_paths.csv'):
    file_paths = get_list_image_paths(Path(image_folder))
    extensions = [get_file_extension(Path(path)) for path in file_paths]
    counter = Counter(extensions)
    # df = pd.DataFrame(data={"path": file_paths, "extension": extensions})
    # df.to_csv(cvs_output_file, index=False)
    image_metadata = [get_tags_from_image(path) for path in tqdm(file_paths)]
    image_metadata = filter(None, image_metadata)
    df = pd.DataFrame(data=image_metadata)
    df.to_csv(cvs_output_file, index=False, escapechar='\\')


def get_tags_from_image(image_path: str):
    """
    Return a list of tags from an image
    """
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]  # , '.heic'
    if Path(image_path).suffix not in image_extensions:
        return None
        return {'path': image_path}

    image = Image.open(image_path)

    # extracting the exif metadata
    exifdata = image.getexif()
    result = {'path': image_path}

    # looping through all the tags present in exifdata
    for tagid in exifdata:
        # getting the tag name instead of tag id
        tagname = TAGS.get(tagid, tagid)

        # passing the tagid to get its respective value
        value = exifdata.get(tagid)
        if type(value) is str:
            value = value.encode('utf-8')
        result[tagname] = value

        # printing the final result
        # print(f"{tagname:25}: {value}")
    return result


if __name__ == '__main__':
    Fire(main)
