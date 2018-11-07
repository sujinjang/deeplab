import argparse
import os
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from image_utils import ImageReader
import common


def create_labels(path_to_raw_img, path_to_labeled_img, path_to_outdir, valid_ratio=0.15, test_ratio=0.15, raw_type='jpg', label_type='png'):
    """Read file names of labeled images and generate train/valid/test images"""

    assert not os.path.isfile(path_to_labeled_img) or not os.path.isfile(path_to_raw_img), "Check data folders."

    org_img_list = []
    labeled_img_list = []

    # Collect valid labeled and raw image file names
    org_img_names = os.listdir(path_to_raw_img)
    for label_name in os.listdir(path_to_labeled_img):
        org_name = os.path.splitext(label_name)[0] + '.' + raw_type
        if os.path.splitext(label_name.lower())[1] == ('.'+label_type) and org_name in org_img_names:
            labeled_img_list.append(os.path.join(path_to_labeled_img, label_name))
            org_img_list.append(os.path.join(path_to_raw_img, org_name))

    # Shuffle data set
    assert len(org_img_list)==len(labeled_img_list), "The size of loaded images and labels are not equal."

    # Split/shuffle training, valid, test data set
    org_img_train, org_img_test, labeled_img_train, labeled_img_test = train_test_split(org_img_list,
                                                                                        labeled_img_list,
                                                                                        test_size=test_ratio,
                                                                                        shuffle=True)

    org_img_train, org_img_valid, labeled_img_train, labeled_img_valid= train_test_split(org_img_train,
                                                                                        labeled_img_train,
                                                                                        test_size=len(org_img_list)*valid_ratio/len(org_img_train),
                                                                                        shuffle=True)

    common.save_filenames_to_txt(org_img_train, os.path.join(path_to_outdir,'raw_train.txt'))
    common.save_filenames_to_txt(labeled_img_train, os.path.join(path_to_outdir, 'label_train.txt'))

    common.save_filenames_to_txt(org_img_valid, os.path.join(path_to_outdir, 'raw_valid.txt'))
    common.save_filenames_to_txt(labeled_img_valid, os.path.join(path_to_outdir, 'label_valid.txt'))

    common.save_filenames_to_txt(org_img_test, os.path.join(path_to_outdir, 'raw_test.txt'))
    common.save_filenames_to_txt(labeled_img_test, os.path.join(path_to_outdir, 'label_test.txt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_raw_img', type=str, default='./datasets/pascal_voc_seg/VOCdevkit/VOC2012/JPEGImages')
    parser.add_argument('--path_to_labeled_img', type=str, default='./datasets/pascal_voc_seg/VOCdevkit/VOC2012/SegmentationClassRaw')
    parser.add_argument('--path_to_outdir', type=str, default='./datasets/pascal_voc_seg/VOCdevkit/VOC2012/DataList')
    FLAGS, unparsed = parser.parse_known_args()

    if not os.path.exists(FLAGS.path_to_outdir):
        os.makedirs(FLAGS.path_to_outdir)

    create_labels(FLAGS.path_to_raw_img, FLAGS.path_to_labeled_img, FLAGS.path_to_outdir)