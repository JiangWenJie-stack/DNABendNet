import numpy as np
import tensorflow as tf
import sys
import os
sys.path.append("../hj_resnet")
from func_model import extraData, apply_rev_comp, train, evaluate
from model_multicaps import Multicaps

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
num_classes = 10
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
# run model
#------------------------------------------------------------------------------#
model = Multicaps().Multicaps(X_train.shape[1:], num_classes)
model.summary()
history = train(model, X_train, Y_train, (X_valid, Y_valid),
    			checkpoint_path=model_path,
    			epoch=num_epochs, batch_size=mini_batch)


#------------------------------------------------------------------------------#
# evaluate model
#------------------------------------------------------------------------------#
evaluate(model, X_test, Y_test, history, model_path, figure_path)
