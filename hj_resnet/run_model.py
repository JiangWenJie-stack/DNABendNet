import numpy as np
import tensorflow as tf
import sys
import os
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
num_epochs = 20
num_blocks = 64
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
def Resnet(input_shape, num_blocks=12, num_classes=1):
	def resnet_block(input):
		x = tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu')(input)
		x = tf.keras.layers.BatchNormalization(axis=2)(x)
		x = tf.keras.layers.Conv1D(64, 3, padding='same', activation=None)(x)
		x = tf.keras.layers.BatchNormalization(axis=2)(x)
		x = tf.keras.layers.Add()([x, input])
		x = tf.keras.layers.Activation('relu')(x)
		return x
	inputs = tf.keras.Input(input_shape)
	x = tf.keras.layers.Conv1D(128, 5, activation='relu')(inputs)
	x = tf.keras.layers.BatchNormalization(axis=2)(x)
	x = tf.keras.layers.Conv1D(64, 3, activation='relu')(x)
	x = tf.keras.layers.BatchNormalization(axis=2)(x)
	for i in range(num_blocks):
		x = resnet_block(x)
	x = tf.keras.layers.Flatten()(x)
	x = tf.keras.layers.Dense(128, activation='relu')(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Dropout(0.5)(x)
	x = tf.keras.layers.Dense(64, activation='relu')(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Dropout(0.3)(x)
	outputs = tf.keras.layers.Dense(1, activation='linear')(x)
	model = tf.keras.Model(inputs=inputs, outputs=outputs)
	return model


#------------------------------------------------------------------------------#
# run model
#------------------------------------------------------------------------------#
model = Resnet(X_train.shape[1:], num_blocks)
model.summary()
history = train(model, X_train, Y_train, (X_valid, Y_valid),
    			checkpoint_path=model_path,
    			epoch=num_epochs, batch_size=mini_batch)


#------------------------------------------------------------------------------#
# evaluate model
#------------------------------------------------------------------------------#
evaluate(model, X_test, Y_test, history, figure_path)
