# encoding: utf-8
from PIL import Image, ImageFile
import errno
import json
import pickle as pkl
import os
import os.path as osp
import yaml
import time
from easydict import EasyDict as edict

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path, max_retries=10, retry_delay=0.1):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))

    for attempt in range(1, max_retries + 1):
        try:
            return Image.open(img_path).convert('RGB')
        except IOError as err:
            if attempt == max_retries:
                raise IOError(
                    "Failed to read image '{}' after {} attempts".format(
                        img_path, max_retries
                    )
                ) from err
            print(
                "IOError incurred when reading '{}'. Retry {}/{}.".format(
                    img_path, attempt, max_retries
                )
            )
            time.sleep(retry_delay)


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def check_isfile(path):
    isfile = osp.isfile(path)
    if not isfile:
        print("=> Warning: no file found at '{}' (ignored)".format(path))
    return isfile


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def get_text_embedding(path, length):
    with open(path, 'rb') as f:
        word_frequency = pkl.load(f)


def save_train_configs(path, args):
    if not os.path.exists(path):
        os.makedirs(path)
    config_dict = vars(args)
    for filename in ("config.yaml", "configs.yaml"):
        with open(osp.join(path, filename), 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

def load_train_configs(path):
    with open(path, 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    return edict(args)
