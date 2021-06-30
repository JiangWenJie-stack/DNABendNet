import numpy as np
import tensorflow as tf
import sys
import os
from keras import layers, backend
sys.path.append("../hj_resnet")
from func_model import extraData, apply_rev_comp, train, evaluate

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# tf.config.threading.set_intra_op_parallelism_threads(5)
tf.config.threading.set_inter_op_parallelism_threads(10)


#------------------------------------------------------------------------------#
# pars
#------------------------------------------------------------------------------#
model_path = 'model/cp'
figure_path = 'figure'
mini_batch = 256
num_epochs = 10
use_mini_data = False


#------------------------------------------------------------------------------#
# load data
#------------------------------------------------------------------------------#
if use_mini_data is False:
	train_dir = '../Data/Data1/train_set.txt'
	vaild_dir = '../Data/Data1/vaild_set.txt'
	test_dir = '../Data/Data1/test_set.txt'
else:
	train_dir = '../Data/minidata/train_set.txt'
	vaild_dir = '../Data/minidata/vaild_set.txt'
	test_dir = '../Data/minidata/test_set.txt'

X_train, Y_train, X_valid, Y_valid, X_test, Y_test = \
	extraData(train_dir, vaild_dir, test_dir)

# stochastic reverse complement
X_train = apply_rev_comp(X_train)


#------------------------------------------------------------------------------#
# build model
#------------------------------------------------------------------------------#
def conv1d_bn(x, filters, kernel_size, strides=1, padding='same',
			  activation='relu', use_bias=False):
	"""conv1d + BN
	"""
	x = layers.Conv1D(filters, kernel_size, strides=strides, padding=padding,
					  use_bias=use_bias)(x)
	if not use_bias:
		bn_axis = 2
		x = layers.BatchNormalization(axis=bn_axis)(x)
	if activation is not None:
		x = layers.Activation(activation)(x)
	return x

def inception_resnet_block(x, scale, block_type, activation='relu'):
	"""Inception-ResNet block
	- Inception-ResNet-A: `block_type='block35'`
	- Inception-ResNet-B: `block_type='block17'`
	- Inception-ResNet-C: `block_type='block8'`
	"""
	if block_type == 'A':
		branch_0 = conv1d_bn(x, 32, 1)
		branch_1 = conv1d_bn(x, 32, 1)
		branch_1 = conv1d_bn(branch_1, 32, 3)
		branch_2 = conv1d_bn(x, 32, 1)
		branch_2 = conv1d_bn(branch_2, 48, 3)
		branch_2 = conv1d_bn(branch_2, 64, 3)
		branches = [branch_0, branch_1, branch_2]
	elif block_type == 'B':
		branch_0 = conv1d_bn(x, 192, 1)
		branch_1 = conv1d_bn(x, 128, 1)
		branch_1 = conv1d_bn(branch_1, 160, 7)
		branch_1 = conv1d_bn(branch_1, 192, 7)
		branches = [branch_0, branch_1]
	elif block_type == 'C':
		branch_0 = conv1d_bn(x, 192, 1)
		branch_1 = conv1d_bn(x, 192, 1)
		branch_1 = conv1d_bn(branch_1, 224, 5)
		branch_1 = conv1d_bn(branch_1, 256, 5)
		branches = [branch_0, branch_1]
	else:
		raise ValueError('Unknown Inception-ResNet block type. '
	                 'Expects "A", "B" or "C", '
	                 'but got: ' + str(block_type))
	channel_axis = 2
	mixed = layers.Concatenate(axis=channel_axis)(branches)
	up = conv1d_bn(mixed, backend.int_shape(x)[channel_axis], 1,
				   activation=None, use_bias=True)
	x = layers.Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
					  output_shape=backend.int_shape(x)[1:],
					  arguments={'scale': scale})([x, up])
	if activation is not None:
		x = layers.Activation(activation)(x)
	return x


def InceptionResNetV2(input_shape, num_classes = 1):
	"""Inception-ResNet v2
	"""
	# inputs
	inputs = tf.keras.Input(input_shape)
	# Stem block: 36 * 192
	x = conv1d_bn(inputs, 32, 3, padding='valid')
	x = conv1d_bn(x, 64, 5, padding='valid')
	x = layers.MaxPooling1D(3, strides=1)(x)
	x = conv1d_bn(x, 80, 3, padding='valid')
	x = conv1d_bn(x, 192, 3, padding='valid')
	x = layers.MaxPooling1D(3, strides=1)(x)
	# Mixed 5b (Inception-A block)
	branch_0 = conv1d_bn(x, 96, 1)
	branch_1 = conv1d_bn(x, 48, 1)
	branch_1 = conv1d_bn(branch_1, 64, 5)
	branch_2 = conv1d_bn(x, 64, 1)
	branch_2 = conv1d_bn(branch_2, 96, 3)
	branch_2 = conv1d_bn(branch_2, 96, 3)
	branch_pool = layers.AveragePooling1D(3, strides=1, padding='same')(x)
	branch_pool = conv1d_bn(branch_pool, 64, 1)
	branches = [branch_0, branch_1, branch_2, branch_pool]
	channel_axis = 2
	x = layers.Concatenate(axis=channel_axis, name='Inception_A')(branches)
	# 10x block35 (Inception-ResNet-A block)
	for block_idx in range(1, 11):
		x = inception_resnet_block(x, scale=0.17, block_type='A')
	# Mixed 6a (Reduction-A block): 17 * 1088
	branch_0 = conv1d_bn(x, 384, 3, strides=2, padding='valid')
	branch_1 = conv1d_bn(x, 256, 1)
	branch_1 = conv1d_bn(branch_1, 256, 3)
	branch_1 = conv1d_bn(branch_1, 384, 3, strides=2, padding='valid')
	branch_pool = layers.MaxPooling1D(3, strides=2, padding='valid')(x)
	branches = [branch_0, branch_1, branch_pool]
	x = layers.Concatenate(axis=channel_axis, name='Reduction_A')(branches)
	# 20x block17 (Inception-ResNet-B block)
	for block_idx in range(1, 21):
		x = inception_resnet_block(x, scale=0.1, block_type='B')
	# Mixed 7a (Reduction-B block): 8 * 2080
	branch_0 = conv1d_bn(x, 256, 1)
	branch_0 = conv1d_bn(branch_0, 384, 3, strides=2, padding='valid')
	branch_1 = conv1d_bn(x, 256, 1)
	branch_1 = conv1d_bn(branch_1, 288, 3, strides=2, padding='valid')
	branch_2 = conv1d_bn(x, 256, 1)
	branch_2 = conv1d_bn(branch_2, 288, 3)
	branch_2 = conv1d_bn(branch_2, 320, 3, strides=2, padding='valid')
	branch_pool = layers.MaxPooling1D(3, strides=2, padding='valid')(x)
	branches = [branch_0, branch_1, branch_2, branch_pool]
	x = layers.Concatenate(axis=channel_axis, name='Reduction_B')(branches)
	# 10x block8 (Inception-ResNet-C block): 8 x 2080
	for block_idx in range(1, 10):
		x = inception_resnet_block(x, scale=0.2, block_type='C')
	x = inception_resnet_block(x, scale=1., activation=None, block_type='C')
	# Final convolution block: 8 x 1536
	x = conv1d_bn(x, 1536, 1)
	# outputs
	x = layers.GlobalAveragePooling1D(name='avg_pool')(x)
	outputs = layers.Dense(num_classes, activation='linear')(x)
	model = tf.keras.Model(inputs=inputs, outputs=outputs)
	return model


#------------------------------------------------------------------------------#
# run model
#------------------------------------------------------------------------------#
model = InceptionResNetV2(X_train.shape[1:])
model.summary()
history = train(model, X_train, Y_train, (X_valid, Y_valid),
    			checkpoint_path=model_path,
    			epoch=num_epochs, batch_size=mini_batch)


#------------------------------------------------------------------------------#
# evaluate model
#------------------------------------------------------------------------------#
evaluate(model, X_test, Y_test, history, model_path, figure_path)
