import tensorflow as tf
import numpy as np
from load2data import extraData
from model_resnet import Resnet
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

####################################
#   load  data(train set、vaild set and test set) and get data shape for model
#####################################
train_dir = 'D:\JWJ\Huangcongcong\Data\\723\\train_data.txt'
vaild_dir = 'D:\JWJ\Huangcongcong\Data\\723\\vaild_data.txt'
test_dir = 'D:\JWJ\Huangcongcong\Data\\723\\test_data.txt'

X_train, Y_train, X_test, Y_test, X_valid, Y_valid = extraData(train_dir, vaild_dir, test_dir)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train,Y_train)).shuffle(1000).batch(128)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test,Y_test)).shuffle(1000).batch(128)
valid_dataset = tf.data.Dataset.from_tensor_slices((X_valid,Y_valid)).shuffle(1000).batch(128)
X_train_shape1, X_train_shape2, X_train_shape3 = X_train.shape[1], X_train.shape[2], X_train.shape[3]

####################################
#   set model parameter such as learning rate、 drop rate and l2
#####################################
lr = 3e-4
filt = 64
den_pa1 = 256 ; den_pa2 = 32
drop1 = 0.5 ; drop2 = 0.5
alpha11 = 0.0005 ; alpha12 = 0.0005
alpha21 = 0.001 ; alpha22 = 0.001

####################################
#   load data
#####################################
resnet_model = Resnet().resnet_model(X_train_shape1, X_train_shape2, X_train_shape3,
                                   lr=lr, filt = filt, den_pa1 = den_pa1, den_pa2 = den_pa2, drop1 = drop1, drop2 = drop2,
                                   alpha11 = alpha11, alpha12 = alpha12, alpha21 = alpha21, alpha22 = alpha22)

####################################
#   train model on train set and vaild set
#####################################
epochs = 5
history = resnet_model.fit(train_dataset,epochs=epochs,validation_data = valid_dataset)
####################################
#   save model, model file is a h5 file, save model like model.save("XXX.h5")
####################################
filepath = 'D:\JWJ\Huangcongcong\se-resnet\\resnet.h5'
resnet_model.save(filepath)

####################################
#    load model to test and plot
####################################
modelPath = 'D:\JWJ\Huangcongcong\se-resnet\\resnet.h5'
resnet_model = load_model(modelPath)
preds = resnet_model.predict(X_test)
prediction = preds.reshape(Y_test.shape[0],)
Y_test = Y_test.reshape(Y_test.shape[0],)

corr = np.corrcoef(Y_test ,prediction)
print(corr[0][1])

plt.scatter(prediction,Y_test,s=1.,marker='o',c=prediction, cmap='summer')
plt.title(f"correlation:{corr}", fontsize=15)
plt.ylabel("prediction", fontsize=10, rotation=90)
plt.xlabel("Y_test", fontsize=10, rotation=0)
plt.savefig('scatter_new.png')

mse = mean_squared_error(Y_test, prediction)
print(mse)
plt.figure(figsize=(10,10))
plt.plot(prediction,c="red")
plt.plot(Y_test,c='blue')
plt.title("contrast", fontsize=15)
plt.ylabel("prediction", fontsize=10, rotation=90)
plt.xlabel("Y_test", fontsize=10, rotation=0)
plt.savefig('contrast_new.png')



