# Benchmarking TensorFlow DeepLab on PASCAL VOC dataset
import argparse
import os
import numpy as np
import tensorflow as tf
import common
import models
from image_utils import ImageReader


_NUM_CLASSES = 21
_HEIGHT = 513
_WIDTH = 513
_DEPTH = 3
_MIN_SCALE = 0.5
_MAX_SCALE = 2.0
_IGNORE_LABEL = 255
_POWER = 0.9
_MOMENTUM = 0.9
_BATCH_NORM_DECAY = 0.9997
_NUM_IMAGES = {
    'train': 10582,
    'validation': 1449,
}


def main():

    model_params={
          'input_size': FLAGS.input_size,
          'output_stride': FLAGS.output_stride,
          'batch_size': FLAGS.batch_size,
          'train_epochs': FLAGS.train_epochs,
          'pre_trained_model_path': FLAGS.pre_trained_model_path,
          'pre_trained_model_name': FLAGS.pre_trained_model_name,
          'batch_norm_decay': _BATCH_NORM_DECAY,
          'num_classes': _NUM_CLASSES,
          'tensorboard_images_max_outputs': FLAGS.tensorboard_images_max_outputs,
          'weight_decay': FLAGS.weight_decay,
          'learning_rate_policy': FLAGS.learning_rate_policy,
          'num_train': _NUM_IMAGES['train'],
          'initial_learning_rate': FLAGS.initial_learning_rate,
          'max_iter': FLAGS.max_iter,
          'end_learning_rate': FLAGS.end_learning_rate,
          'power': _POWER,
          'momentum': _MOMENTUM,
          'freeze_batch_norm': FLAGS.freeze_batch_norm,
          'initial_global_step': FLAGS.initial_global_step
      }



    graph = tf.Graph()
    with graph.as_default():

        ### Load image file names of training and validation set
        img_list_train = common.read_filenames_from_txt(os.path.join(FLAGS.path_to_filelist, 'raw_train.txt'))
        img_list_valid = common.read_filenames_from_txt(os.path.join(FLAGS.path_to_filelist, 'raw_valid.txt'))
        label_list_train = common.read_filenames_from_txt(os.path.join(FLAGS.path_to_filelist, 'label_train.txt'))
        label_list_valid = common.read_filenames_from_txt(os.path.join(FLAGS.path_to_filelist, 'label_valid.txt'))

        ### Build image input/output tensor
        inputs = models.build_input_tensor(model_params['input_size'])
        labels = models.build_label_tensor(model_params['input_size'])

        ### Build DeepLab network
        train_op, loss, predictions = models.build_deeplabv3_model_fn(inputs=inputs,
                                                                      labels=labels,
                                                                      is_training=True,
                                                                      model_params=model_params)

        ### Initialize TF variables and session
        init_op = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init_op)

        ### Run training loop
        n_steps_per_epoch = int(len(img_list_train) / model_params['batch_size'])
        n_last_batch = len(img_list_train) - model_params['batch_size']* n_steps_per_epoch
        for epoch in range(model_params['train_epochs']):
            samples = image_utils.
        # _, loss_val, pred_val = sess.run([train_op, loss, predictions],
        #          feed_dict={inputs:img_samples, labels:img_labels})
        # print(loss_val)






if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir', type=str, default='./model',
                        help='Base directory for the outpu model.')
    parser.add_argument('--path_to_filelist', type=str, default='./datasets/pascal_voc_seg/VOCdevkit/VOC2012/DataList')
    # parser.add_argument('--path_to_data', type=str, default='./datasets/JPEGImages')
    # parser.add_argument('--path_to_label', type=str, default='./datasets/SegmentationClassRaw')

    parser.add_argument('--clean_model_dir', action='store_true',
                        help='Whether to clean up the model directory if present.')

    parser.add_argument('--train_epochs', type=int, default=26,
                        help='Number of training epochs: '
                             'For 30K iteration with batch size 6, train_epoch = 17.01 (= 30K * 6 / 10,582). '
                             'For 30K iteration with batch size 8, train_epoch = 22.68 (= 30K * 8 / 10,582). '
                             'For 30K iteration with batch size 10, train_epoch = 25.52 (= 30K * 10 / 10,582). '
                             'For 30K iteration with batch size 11, train_epoch = 31.19 (= 30K * 11 / 10,582). '
                             'For 30K iteration with batch size 15, train_epoch = 42.53 (= 30K * 15 / 10,582). '
                             'For 30K iteration with batch size 16, train_epoch = 45.36 (= 30K * 16 / 10,582).')

    parser.add_argument('--epochs_per_eval', type=int, default=1,
                        help='The number of training epochs to run between evaluations.')

    parser.add_argument('--input_size', type=int, default=513,
                        help='width(height) of')

    parser.add_argument('--tensorboard_images_max_outputs', type=int, default=6,
                        help='Max number of batch elements to generate for Tensorboard.')

    parser.add_argument('--batch_size', type=int, default=10,
                        help='Number of examples per batch.')

    parser.add_argument('--learning_rate_policy', type=str, default='poly',
                        choices=['poly', 'piecewise'],
                        help='Learning rate policy to optimize loss.')

    parser.add_argument('--max_iter', type=int, default=30000,
                        help='Number of maximum iteration used for "poly" learning rate policy.')

    parser.add_argument('--data_dir', type=str, default='./dataset/',
                        help='Path to the directory containing the PASCAL VOC data tf record.')

    parser.add_argument('--pre_trained_model_path', type=str, default='./pretrained/resnet_v2_101/resnet_v2_101.ckpt',
                        help='Path to the pre-trained model checkpoint.')

    parser.add_argument('--pre_trained_model_name', type=str,
                        default='resnet_v2_101',
                        help='Name of the pre-trained model.')

    parser.add_argument('--output_stride', type=int, default=16,
                        choices=[8, 16],
                        help='Output stride for DeepLab v3. Currently 8 or 16 is supported.')

    parser.add_argument('--freeze_batch_norm', action='store_true',
                        help='Freeze batch normalization parameters during the training.')

    parser.add_argument('--initial_learning_rate', type=float, default=7e-3,
                        help='Initial learning rate for the optimizer.')

    parser.add_argument('--end_learning_rate', type=float, default=1e-6,
                        help='End learning rate for the optimizer.')

    parser.add_argument('--initial_global_step', type=int, default=0,
                        help='Initial global step for controlling learning rate when fine-tuning model.')

    parser.add_argument('--weight_decay', type=float, default=2e-4,
                        help='The weight decay to use for regularizing the model.')

    parser.add_argument('--debug', action='store_true',
                        help='Whether to use debugger to track down bad values during training.')

    FLAGS, unparsed = parser.parse_known_args()

    main()