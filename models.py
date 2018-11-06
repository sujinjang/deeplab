import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as base_nets
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers
from PIL import Image
import numpy as np

import image_utils


def atrous_spatial_pyramid_pooling(inputs, output_stride, batch_norm_decay, is_training, depth=256):
  """Atrous Spatial Pyramid Pooling.

  Args:
    inputs: A tensor of size [batch, height, width, channels].
    output_stride: The ResNet unit's stride. Determines the rates for atrous convolution.
      the rates are (6, 12, 18) when the stride is 16, and doubled when 8.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    is_training: A boolean denoting whether the input is for training.
    depth: The depth of the ResNet unit output.

  Returns:
    The atrous spatial pyramid pooling output.
  """
  with tf.variable_scope("aspp"):
    if output_stride not in [8, 16]:
      raise ValueError('output_stride must be either 8 or 16.')

    atrous_rates = [6, 12, 18]
    if output_stride == 8:
      atrous_rates = [2*rate for rate in atrous_rates]

    with tf.contrib.slim.arg_scope(base_nets.resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
      with arg_scope([layers.batch_norm], is_training=is_training):
        inputs_size = tf.shape(inputs)[1:3]
        # (a) one 1x1 convolution and three 3x3 convolutions with rates = (6, 12, 18) when output stride = 16.
        # the rates are doubled when output stride = 8.
        conv_1x1 = layers_lib.conv2d(inputs, depth, [1, 1], stride=1, scope="conv_1x1")
        conv_3x3_1 = base_nets.resnet_utils.conv2d_same(inputs, depth, 3, stride=1, rate=atrous_rates[0], scope='conv_3x3_1')
        conv_3x3_2 = base_nets.resnet_utils.conv2d_same(inputs, depth, 3, stride=1, rate=atrous_rates[1], scope='conv_3x3_2')
        conv_3x3_3 = base_nets.resnet_utils.conv2d_same(inputs, depth, 3, stride=1, rate=atrous_rates[2], scope='conv_3x3_3')

        # (b) the image-level features
        with tf.variable_scope("image_level_features"):
          # global average pooling
          image_level_features = tf.reduce_mean(inputs, [1, 2], name='global_average_pooling', keepdims=True)
          # 1x1 convolution with 256 filters( and batch normalization)
          image_level_features = layers_lib.conv2d(image_level_features, depth, [1, 1], stride=1, scope='conv_1x1')
          # bilinearly upsample features
          image_level_features = tf.image.resize_bilinear(image_level_features, inputs_size, name='upsample')

        net = tf.concat([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, image_level_features], axis=3, name='concat')
        net = layers_lib.conv2d(net, depth, [1, 1], stride=1, scope='conv_1x1_concat')

        return net

def deeplab_v3_generator(num_classes,
                         output_stride,
                         pretrained_model_name,
                         pretrained_model_path,
                         batch_norm_decay):
    """Generator for deeplabv3

    Args:
        num_classes: the number of classes for image classification
        output_stride: the rate for atrous convolution
        pretrained_model_name: the type of base feature extractor (e.g., resnet_v2_50, resnet_v2_100)
        pretrained_model_path: the path to the directory of pre-trained base model
        batch_norm_decay: the moving average decay when estimating layer activation statistics in batch normalization.

    Returns:

    """
    if pretrained_model_name not in ['resnet_v2_50', 'resnet_v2_101']:
        raise ValueError("'base_model_name' is not in the supported based model list.")

    if pretrained_model_name == 'resnet_v2_50':
        base_model = base_nets.resnet_v2.resnet_v2_50
    else:
        base_model = base_nets.resnet_v2.resnet_v2_101

    def model_fn(inputs, is_training):
        """Construct model function based on 'base_model'"""
        with slim.arg_scope(base_nets.resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
            logits, end_points = base_model(inputs=inputs,
                                            num_classes=None,
                                            is_training=is_training,
                                            global_pool=False,
                                            output_stride=output_stride)
        if is_training:
            exclude = [pretrained_model_name + '/logits', 'global_step']
            variables_to_restore = slim.get_variables_to_restore(exclude=exclude)
            tf.train.init_from_checkpoint(pretrained_model_path,
                                          {v.name.split(':')[0]: v for v in variables_to_restore})

        # Pre-trained feature extractor
        feature_net = end_points[pretrained_model_name + '/block4']

        # Atrous CNN
        atrous_net = atrous_spatial_pyramid_pooling(inputs=feature_net,
                                                    output_stride=output_stride,
                                                    batch_norm_decay=batch_norm_decay,
                                                    is_training=is_training)
        with tf.variable_scope("upsampling_logits"):
            up_net = layers_lib.conv2d(inputs=atrous_net,
                                       num_outputs=num_classes,
                                       kernel_size=[1,1],
                                       activation_fn=None,
                                       normalizer_fn=None,
                                       scope='conv_1x1')
            input_size = tf.shape(inputs)[1:3]
            logits = tf.image.resize_bilinear(up_net,
                                              input_size,
                                              name='upsample')
        return logits

    return model_fn

def build_deeplabv3_model_fn(inputs, labels, is_training, model_params):
    """Build deeplab_v3 model"""

    network_fn = deeplab_v3_generator(num_classes=model_params['num_classes'],
                                   output_stride=model_params['output_stride'],
                                   pretrained_model_name=model_params['pre_trained_model_name'],
                                   pretrained_model_path=model_params['pre_trained_model_path'],
                                   batch_norm_decay=model_params['batch_norm_decay'])
    logits = network_fn(inputs=inputs, is_training=is_training)
    pred_classes = tf.expand_dims(tf.argmax(logits, axis=3, output_type=tf.int32), axis=3)
    pred_labels = tf.py_func(func=image_utils.decode_labels,
                             inp=[pred_classes, model_params['batch_size'], model_params['num_classes']],
                             Tout=tf.uint8)

    predictions = {
        'classes': pred_classes,
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor'),
        'decoded_labels': pred_labels
    }

    if not is_training:
        predictions_without_decoded_labels = predictions.copy()
        del predictions_without_decoded_labels['decoded_labels']
        return predictions

    grd_labels = tf.py_func(func=image_utils.decode_labels,
                            inp=[labels, model_params['batch_size'], model_params['num_classes']],
                            Tout=tf.uint8)

    logits_by_num_classes = tf.reshape(logits, [-1, model_params['num_classes']])
    labels = tf.squeeze(labels, axis=3)
    labels_flat = tf.reshape(labels, [-1, ])

    valid_indices = tf.to_int32(labels_flat <= model_params['num_classes'] - 1)
    valid_logits = tf.dynamic_partition(logits_by_num_classes, valid_indices, num_partitions=2)[1]
    valid_labels = tf.dynamic_partition(labels_flat, valid_indices, num_partitions=2)[1]

    preds_flat = tf.reshape(pred_classes, [-1, ])
    valid_preds = tf.dynamic_partition(preds_flat, valid_indices, num_partitions=2)[1]
    confusion_matrix = tf.confusion_matrix(valid_labels, valid_preds, num_classes=model_params['num_classes'])

    predictions['valid_preds'] = valid_preds
    predictions['valid_labels'] = valid_labels
    predictions['confusion_matrix'] = confusion_matrix

    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        logits=valid_logits, labels=valid_labels)

    # Optimizer
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=0.0001,
        momentum=model_params['momentum'])
    train_op = optimizer.minimize(cross_entropy)

    return train_op, cross_entropy, predictions


def build_input_tensor(input_size):
    """Build input placeholder"""
    with tf.name_scope('input'):
        inputs = tf.placeholder(dtype=tf.float32, shape=[None, input_size, input_size, 3], name='input')

    return inputs

def build_label_tensor(label_size):
    with tf.name_scope('label'):
        labels = tf.placeholder(dtype=tf.int32, shape=[None, label_size, label_size, 1], name='label')
    return labels