import keras
from tensorflow.keras import layers
import tensorflow as tf

class Resnet:

    def __init__(self):
        self.model = self

    def res_block(self,layer, filter_out, kernel_size, alpha):
        layer0 = layers.Conv2D(kernel_regularizer=keras.regularizers.l2(alpha), filters=filter_out,
                               kernel_size=kernel_size, strides=(1, 1), padding='same', activation='relu')(layer)

        return layer0


    def res_block3(self,layer, filter_out, kernel_size, alpha):
        layer0 = layers.Conv2D(kernel_regularizer=keras.regularizers.l2(alpha), filters=filter_out,
                               kernel_size=kernel_size, strides=(1, 1), padding='same', activation='relu')(layer)
        layer1 = layers.Conv2D(kernel_regularizer=keras.regularizers.l2(alpha), filters=filter_out,
                               kernel_size=kernel_size, strides=(1, 1), padding='same', activation='relu')(layer0)
        layer2 = layers.Conv2D(kernel_regularizer=keras.regularizers.l2(alpha), filters=filter_out,
                               kernel_size=kernel_size, strides=(1, 1), padding='same', activation='relu')(layer1)
        layer3 = tf.add(layer0, layer2)

        return layer3


    def inception_block(self,layer, filt, alpha):
        layerp = layers.MaxPool2D(pool_size=(2, 1), strides=(1, 1), padding='same')(layer)
        layer1 = layers.Conv2D(kernel_regularizer=keras.regularizers.l2(alpha), filters=filt * 0.25,
                               kernel_size=[1, 1], strides=(1, 1), padding='same', activation='relu', )(layerp)
        layer3 = self.res_block(layer, filt, [3, 1], alpha)
        layer5 = self.res_block(layer, filt, [5, 1], alpha)
        layer7 = self.res_block(layer, filt, [7, 1], alpha)
        feat = layers.concatenate([layer1, layer3, layer5, layer7])

        return feat



    def resnet_model(self,X_train_shape1, X_train_shape2, X_train_shape3, lr, filt, den_pa1, den_pa2, drop1, drop2, alpha11, alpha12, alpha21, alpha22):

        all_input = keras.Input(shape=(X_train_shape1, X_train_shape2, X_train_shape3,), name='all')
        all_layer = layers.Conv2D(kernel_regularizer=keras.regularizers.l2(alpha11), filters=filt,
                                  kernel_size=(4,4), strides=(1,4), padding='same',activation='relu',)(all_input)

        feat = self.inception_block(all_layer, filt, alpha11)
        feat = self.inception_block(feat, filt, alpha12)
        feat = self.inception_block(feat, filt*2, alpha12)
        feat2 = self.inception_block(feat, filt*2, alpha12)
        flat = layers.Flatten()(feat2)


        CT_dense = layers.Dense(den_pa1, activation='relu', kernel_regularizer = keras.regularizers.l2(alpha21), name='dense1')(flat)
        CT_drop = layers.Dropout(drop1)(CT_dense)
        CT_dense = layers.Dense(den_pa2, activation='relu', kernel_regularizer = keras.regularizers.l2(alpha22), name='dense2')(CT_drop)
        CT_drop = layers.Dropout(drop2)(CT_dense)
        CT_pred = layers.Dense(1, activation='linear', name='priority')(CT_drop)


        model = keras.Model(inputs = all_input, outputs = CT_pred)
        model.summary()

        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=lr),
                      loss={'priority': tf.keras.losses.MeanSquaredError()})

        return model
