import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class Capsule_Mul(layers.Layer):
    def __init__(self, num_classes):
        super(Capsule_Mul, self).__init__()
        self.num_classes = num_classes

    def build(self, input_shape):
        # -*- batch_size use 1, when init weight matrix -*-
        izr = keras.initializers.GlorotUniform()
        rzr = keras.regularizers.l2(0.)

        self.W = self.add_weight('W',
            shape=[1, self.num_classes, input_shape[2], input_shape[3]],
            initializer=izr,
            regularizer=rzr,
            trainable=True,
        )

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'num_classes': self.num_classes
        })
        return config

    def call(self, inputs):
        x = tf.reduce_sum(tf.multiply(inputs, self.W), 3)
        return x


class Vote_Mul(layers.Layer):
    def __init__(self, num_votes, num_classes):
        super(Vote_Mul, self).__init__()
        self.num_votes = num_votes
        self.num_classes = num_classes

    def build(self, input_shape):
        # -*- batch_size use 1, when init weight matrix -*-

        self.W = self.add_weight('WW',
            shape=[1, self.num_classes, self.num_votes],
            initializer='ones',
            trainable=True,
        )

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'num_votes': self.num_votes
        })
        return config

    def call(self, inputs):
        return tf.multiply(inputs, self.W)


class PositionalEncoding(object):
    def __init__(self, position, d):
        angle_rads = self._get_angles(np.arange(position)[:, np.newaxis], np.arange(d)[np.newaxis, :], d)
        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])
        self._encoding = np.concatenate([sines, cosines], axis=1)
        self._encoding = np.expand_dims(self._encoding, 0)
    def _get_angles(self, position, i, d):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d))
        return position * angle_rates
    def get_positional_encoding(self):
        return tf.cast(self._encoding, dtype=tf.float32)


class Multicaps:

    def __init__(self):
        self.model = self

    def Conv2d_bn(self, x, filters, kernel_size, dilation=1,
                  strides=1, padding='same',
    			  activation='relu', use_bias=False):
    	"""conv1d + BN
    	"""
    	x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding,
    					  use_bias=use_bias, dilation_rate=(dilation, 1))(x)
    	if not use_bias:
    		bn_axis = 3
    		x = layers.BatchNormalization(axis=bn_axis)(x)
    	if activation is not None:
    		x = layers.Activation(activation)(x)
    	return x

    def Capsule(self, inputs, num_classes=10, num_dims=64):
        x = tf.keras.layers.Permute([3, 1, 2])(inputs)
        x = tf.keras.layers.Reshape([1, num_dims, -1])(x)
        x = Capsule_Mul(num_classes)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        return tf.reduce_sum(x, axis=2)

    def Vote(self, inputs, num_votes=3, num_classes=10):
        x = tf.stack(inputs, axis=2)
        x = Vote_Mul(num_votes, num_classes)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.reduce_sum(x, axis=2)
        x = tf.nn.softmax(x)
        return x

    def Multicaps(self, input_shape, num_classes=10, num_dims=64):

        inputs = keras.Input(shape=input_shape)

        # positional_encoding = PositionalEncoding(50, 4)
        # positional_encoding = positional_encoding.get_positional_encoding()
        # x = tf.keras.layers.Add()([inputs, positional_encoding])

        x = tf.expand_dims(inputs, 2)

        x = self.Conv2d_bn(x, 32, [3, 1])
        x = self.Conv2d_bn(x, 48, [3, 1])
        x = self.Conv2d_bn(x, 64, [3, 1])
        cap3 = self.Capsule(x, num_classes, num_dims)

        x = self.Conv2d_bn(x, 48, [5, 1])
        x = self.Conv2d_bn(x, 64, [5, 1])
        x = self.Conv2d_bn(x, 128, [5, 1])
        cap6 = self.Capsule(x, num_classes, num_dims)

        x = self.Conv2d_bn(x, 64, [7, 1])
        x = self.Conv2d_bn(x, 128, [7, 1])
        x = self.Conv2d_bn(x, 192, [7, 1])
        cap9 = self.Capsule(x, num_classes, num_dims)

        caps = self.Vote([cap3, cap6, cap9], 3, num_classes)

        outputs = tf.keras.layers.Dense(1, activation='linear')(caps)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        return model
