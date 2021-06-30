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

def attnet(input_shape, num_classes=1):
	inputs = tf.keras.Input(input_shape)
	x = conv1d_bn(inputs, 32, 3, padding='valid')
	x = conv1d_bn(x, 64, 5, padding='valid')
	x = layers.MaxPooling1D(3, strides=1)(x)
	x = conv1d_bn(x, 80, 3, padding='valid')
	x = conv1d_bn(x, 192, 5, padding='valid')
	x = layers.MaxPooling1D(3, strides=1)(x)
	x = conv1d_bn(x, 256, 3, padding='valid')
	x = conv1d_bn(x, 384, 5, padding='valid')

	qry = layers.MaxPooling1D(3, strides=2, padding='valid')(x)
	att = layers.Attention()([qry, qry, qry])
	x = layers.Add()([qry, att])

	x = conv1d_bn(x, 1024, 1)
	x = layers.GlobalAveragePooling1D()(x)
	outputs = layers.Dense(num_classes, activation='linear')(x)
	model = tf.keras.Model(inputs=inputs, outputs=outputs)
	return model


#------------------------------------------------------------------------------#
# run model
#------------------------------------------------------------------------------#
model = attnet(X_train.shape[1:])
model.summary()
history = train(model, X_train, Y_train, (X_valid, Y_valid),
    			checkpoint_path=model_path,
    			epoch=num_epochs, batch_size=mini_batch)


#------------------------------------------------------------------------------#
# evaluate model
#------------------------------------------------------------------------------#
evaluate(model, X_test, Y_test, history, model_path, figure_path)
