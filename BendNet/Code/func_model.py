import numpy as np
import os
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import h5py
import tensorflow as tf

plt.switch_backend('agg')

def rev_comp(x):
	return np.flip(x, 1)[:, :, [1, 0, 3, 2]]

def apply_rev_comp(X_train, alpha=0.5):
	X = X_train.copy()
	np.random.seed(12345)
	odr = np.random.permutation(X.shape[0])
	isPick=int(len(odr)*alpha+1)
	X[odr[0:isPick],] = rev_comp(X[odr[0:isPick],])
	return X

def onehot(DNA_seq):
    """
    :param DNA_seq:
    :return: a onehot matrix
    """
    singleDNAFragement = np.zeros([len(DNA_seq),4])
    for i in range(len(DNA_seq)):
        if DNA_seq[i]=='A':
            singleDNAFragement[i,0]=1
        elif DNA_seq[i]=='T':
            singleDNAFragement[i,1]=1
        elif DNA_seq[i]=='G':
            singleDNAFragement[i,2]=1
        elif DNA_seq[i]=='C':
            singleDNAFragement[i,3]=1
        elif DNA_seq[i]=='N':
            singleDNAFragement[i, 0] = 0.25
            singleDNAFragement[i, 1] = 0.25
            singleDNAFragement[i, 2] = 0.25
            singleDNAFragement[i, 3] = 0.25
    return singleDNAFragement

def extraData(train_dir, vaild_dir, test_dir):
    #####################
    #   输入： 三类数据的地址
    #   输出：三类数据对应的数组
    #####################
    train_data = []
    train_label = []
    vaild_data = []
    vaild_label = []
    test_data = []
    test_label = []

    train_count = 0
    vaild_count = 0
    test_count = 0

    for line in open(train_dir,"r"):
        seq = line.split("\t")[0]
        label = line.split("\t")[1]
        train_count += 1
        SingeDnaMateix = onehot(seq)
        train_data.append(SingeDnaMateix)
        train_label.append(float(label))

    for line in open(vaild_dir,"r"):
        seq = line.split("\t")[0]
        label = line.split("\t")[1]
        vaild_count += 1
        SingeDnaMateix = onehot(seq)
        vaild_data.append(SingeDnaMateix)
        vaild_label.append(float(label))

    for line in open(test_dir,"r"):
        seq = line.split("\t")[0]
        label = line.split("\t")[1]
        test_count += 1
        SingeDnaMateix = onehot(seq)
        test_data.append(SingeDnaMateix)
        test_label.append(float(label))

    train_data = np.array(train_data).reshape((train_count,50,4))
    train_label = np.array(train_label)
    vaild_data = np.array(vaild_data).reshape((vaild_count,50,4))
    vaild_label = np.array(vaild_label)
    test_data = np.array(test_data).reshape((test_count,50,4))
    test_label = np.array(test_label)

    return train_data, train_label, vaild_data, vaild_label, test_data, test_label

class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def train(model, X_train, Y_train, validation_data, checkpoint_path="model",
          epoch=10, batch_size=256, lr=0.002):
    # load trained weights
    """
    if checkpoint_path is not None and os.path.exists(checkpoint_path+'.index'):
        print('-' * 30 + 'Previous weights loaded' + '-' * 30)
        model.load_weights(checkpoint_path)
    """
    # compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.025850857923884515), loss='mse')
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path+"\\",
        save_weights_only=True,
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
		log_dir=checkpoint_path+"\\logs")
    # train model
    print('-' * 30 + 'Training' + '-' * 30)
    batch_callback = LossHistory()

    """reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.001)"""

    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epoch,
						validation_data=validation_data,
						callbacks=list([cp_callback, tensorboard_callback, batch_callback]))
    np.savetxt(checkpoint_path+"\\1.txt", batch_callback.losses)
    return history




def evaluate(model, X_test, Y_test, history,
			 checkpoint_path="model", figure_path='figure'):

	if checkpoint_path is not None and os.path.exists(checkpoint_path+'.index'):
		print('-' * 30 + 'Best model weights loaded' + '-' * 30)
		model.load_weights(checkpoint_path)

	# predict
	Y_pred = model.predict(X_test)
	prediction = Y_pred.reshape(Y_test.shape[0],)
	Y_test = Y_test.reshape(Y_test.shape[0],)

	mse = mean_squared_error(Y_test, prediction)
	print('MSE: %.4f' % mse)

	corr = np.corrcoef(Y_test ,prediction)
	print('CPP: %.4f' % corr[0][1])

	if not os.path.exists(figure_path):
		os.mkdir(figure_path)

	plt.figure(figsize=(5,5))
	# plt.scatter(Y_test,prediction,s=1.,marker='o',c=prediction, cmap='summer')
	plt.scatter(Y_test, prediction, s=1., marker='o')
	plt.title(f"correlation:{corr}", fontsize=15)
	plt.ylabel("prediction", fontsize=10, rotation=90)
	plt.xlabel("Y_test", fontsize=10, rotation=0)
	plt.savefig(figure_path+'/scatter.png')

	plt.figure(figsize=(5,5))
	plt.plot(prediction,c="red")
	plt.plot(Y_test,c='blue')
	plt.title("contrast", fontsize=15)
	plt.ylabel("prediction", fontsize=10, rotation=90)
	plt.xlabel("Y_test", fontsize=10, rotation=0)
	plt.savefig(figure_path+'/contrast.png')

	plt.figure(figsize=(5,5))
	plt.plot(history.history['loss'], label="train loss")
	plt.plot(history.history['val_loss'], label="validation loss")
	plt.legend(loc='best')
	plt.savefig(figure_path+'/loss.pdf')


