import copy

import os
from matplotlib import collections
import tensorflow as tf

def save_filenames_to_txt(name_list, target_path):
    """Save data set (filenames) to .txt files"""
    f = open(target_path, 'w')
    for name in name_list:
        f.write(name)
        f.write('\n')
    f.close()

def read_filenames_from_txt(path):
    """Load file names from pre-generated txt file"""
    names = []
    with open(path, 'r') as f:
        if not f:
            raise ValueError("Invalid path to .txt file")
        names = f.readlines()
    return names

def read_filenames(self, path_to_dir, type="jpg"):
    """Read filenames in path"""
    assert not os.path.isfile(path_to_dir), "This is not a folder"
    file_list = []
    for name in os.listdir(path_to_dir):
        if name.lower().endswith(type):
            file_list.append(os.path.join(path_to_dir, name))

    print("Found {} files.".format(len(file_list)))
    return file_list
