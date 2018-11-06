import os
import numpy as np

from PIL import Image

classes = {'none', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
           'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
           'sofa', 'train', 'tvmonitor'}

# colour map
label_colours = [(0, 0, 0),  # 0=background
                 # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                 (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                 # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                 (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                 # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                 (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                 # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                 (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]

def decode_labels(mask, num_images=1, num_classes=21):
    """Decode batch of segmentation masks"""

    n, h, w, c = mask.shape
    assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' \
                              % (n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :, 0]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_, j_] = label_colours[k]
        outputs[i] = np.array(img)
    return outputs



class ImageReader(object):
    def __init__(self, img_dir, input_size, img_type="jpg", ignore_label=None):
        self.img_dir = img_dir  # path/to/images
        self.name_list = self.read_filenames(img_dir, img_type)  # list of file names
        self.img_type = img_type  # image format (e.g., JPEG, PNG)
        self.input_size = input_size  # [width, height]
        self.ignore_label = ignore_label  # class number to ignore in ground-truth image

    def read_filenames(self, path, type="jpg"):
        """Read filenames in path"""

        # assert os.path.isfile(path), "This is not a folder"
        file_list = []
        for name in os.listdir(path):
            if name.lower().endswith(type):
                file_list.append(os.path.join(path, name))

        print("Found {} files.".format(len(file_list)))
        return file_list

    def load_images(self, file_list, resize=True):
        images = []
        for name in file_list:
            img = Image.open(name)

            if resize:
                img = img.resize([self.input_size, self.input_size],
                                 resample=Image.NEAREST)

            img = np.array(img)
            images.append(img)

        return images