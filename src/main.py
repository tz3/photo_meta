import pandas as pd
from PIL.ExifTags import TAGS
from fire import Fire
from tqdm import tqdm

from utils import read_image, create_image_list_based_on


def main(image_folder: str, cvs_output_file: str = '../artifacts/image_meta.csv'):
    file_paths = create_image_list_based_on(image_folder, file_name='../artifacts/image_list2.0.txt')
    image_metadata = [get_tags_from_image(path) for path in tqdm(file_paths)]
    df = pd.DataFrame(data=image_metadata)
    df.to_csv(cvs_output_file, index=False, escapechar='\\')


def get_tags_from_image(image_path: str) -> dict:
    """
    Return a list of tags from an image
    """
    # image_extensions = [".jpg", ".jpeg", ".png", ".bmp", '.heic']  #
    # if Path(image_path).suffix not in image_extensions:
    #     return None
    #     return {'path': image_path}

    image = read_image(image_path)

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
