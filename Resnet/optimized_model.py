import tensorflow as tf
import numpy as np
from load2data import extraData
from model_resnet import Resnet
from gpopy import FlowTunning


####################################
#   load  data(train set、vaild set and test set) and get data shape for model
#####################################
train_dir = 'D:\JWJ\PycharmWorkplace\Resnet\\721\\train_minidata.txt'
vaild_dir = 'D:\JWJ\PycharmWorkplace\Resnet\\721\\vaild_data.txt'
test_dir = 'D:\JWJ\PycharmWorkplace\Resnet\\721\\test_data.txt'

X_train, Y_train, X_test, Y_test, X_valid, Y_valid = extraData(train_dir, vaild_dir, test_dir)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train,Y_train)).shuffle(1000).batch(128)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test,Y_test)).shuffle(1000).batch(128)
valid_dataset = tf.data.Dataset.from_tensor_slices((X_valid,Y_valid)).shuffle(1000).batch(128)
X_train_shape1, X_train_shape2, X_train_shape3 = X_train.shape[1], X_train.shape[2], X_train.shape[3]

####################################
#   set innitial parameters range
#####################################
PARAMS = {
    'learning_rate': {
       'func' :  np.random.uniform,
       'params' : [0.001, 0.0001]},
    'filters' : [16, 64, 128],
    'drop1' : {
        'func' : np.random.uniform,
        'params' : [0.3, 0.7]},
    'drop2' : {
        'func' : np.random.uniform,
        'params' : [0.3, 0.7]},
    'alpha11' : {
        'func' : np.random.uniform,
        'params' : [0.0001, 0.001]},
    'alpha12' : {
        'func' : np.random.uniform,
        'params' : [0.0001, 0.001]},
    'alpha21' : {
        'func' : np.random.uniform,
        'params' : [0.0001, 0.005]},
    'alpha22' : {
        'func' : np.random.uniform,
        'params' : [0.0001, 0.005]},

}
fp = open("optimizing_result.txt","w")
def model(parameter):

    ### optimize parameter
    lr = parameter['learning_rate']
    filt = parameter['filters']
    drop1 = parameter['drop1']
    drop2 = parameter['drop2']
    alpha11 = parameter['alpha11']
    alpha12 = parameter['alpha12']
    alpha21 = parameter['alpha21']
    alpha22 = parameter['alpha22']

    den_pa1, den_pa2 = 256, 32  ##soild parameter

    resnet_model = Resnet().resnet_model(X_train_shape1, X_train_shape2, X_train_shape3,
                                         lr=lr, filt=filt, den_pa1=den_pa1, den_pa2=den_pa2, drop1=drop1, drop2=drop2,
                                         alpha11=alpha11, alpha12=alpha12, alpha21=alpha21, alpha22=alpha22)
    epochs = 1### the times of training on train set
    resnet_model.fit(train_dataset, epochs=epochs, validation_data=valid_dataset)


    vaild_preds = resnet_model.predict(X_valid)
    vaild_prediction = vaild_preds.reshape(vaild_preds.shape[0],)
    vaild_corr = np.corrcoef(Y_valid, vaild_prediction)



    test_preds = resnet_model.predict(X_test)
    test_prediction = test_preds.reshape(test_preds.shape[0], )
    test_corr = np.corrcoef(Y_test, test_prediction)

    mode_info = 'lr='+str(lr) +'_filt='+str(filt) +'_den_pa1='+str(den_pa1) +'_den_pal2='+str() +'_drop1='+str(drop1) +'_drop2='+str(drop2) +'_alpha11='+str(alpha11) +'_alpha12='+str(alpha12) +'_alpha21='+str(alpha21) +'_alpha22='+str(alpha22) +'_resnet.h5'
    model_filepath = 'D:\JWJ\PycharmWorkplace\Resnet\models\\' + 'lr='+str(lr) + '_resnet.h5'
    #print(filepath)
    resnet_model.save(model_filepath)
    fp.write(mode_info +"\n"  + "    vaild_corr= " + str(vaild_corr[0][1]) + "    test_corr= " + str(test_corr[0][1]) + "\n")###write every epoch result

    ###optimize vaild set corr
    return (vaild_corr[0][1], resnet_model)


tunning = FlowTunning(params=PARAMS, population_size=4,maximum_generation = 50)
tunning.set_score(model)
tunning.run()

fp.close()




