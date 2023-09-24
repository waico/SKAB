from tensorflow.keras.layers import Input, Conv2D, ConvLSTM2D, Conv2DTranspose, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
import math

class MSCRED: 
    """
    MSCRED - Multi-Scale Convolutional Recurrent Encoder-Decoder first constructs multi-scale (resolution) signature matrices to characterize multiple levels of the system statuses across different time steps.  In particular, different levels of the system statuses are used to indicate the severity of different abnormal incidents. Subsequently, given the signature matrices, a convolutional encoder is employed to encode the inter-sensor (time series) correlations patterns and an attention based Convolutional Long-Short Term Memory (ConvLSTM) network is developed to capture the temporal patterns. Finally, with the feature maps which encode the inter-sensor correlations and temporal information, a convolutional decoder is used to reconstruct the signature matrices and the residual signature matrices are further utilized to detect and diagnose anomalies. The intuition is that MSCRED may not reconstruct the signature matrices well if it never observes similar system statuses before.

    Parameters
    ----------
    params : list
        A list containing configuration parameters for the MSCRED model.

    Attributes
    ----------
    model : Model
        The trained MSCRED model.

    Examples
    --------
    >>> from MSCRED import MSCRED
    >>> PARAMS = [sensor_n, scale_n, step_max]
    >>> model = MSCRED(PARAMS)
    >>> model.fit(X_train, Y_train, X_test, Y_test)
    >>> prediction = model.predict(test_data)
    """
    
    def __init__(self, params):
        self.params = params
        
    def _build_model(self):
        self._Random(0)
        
        input_size = (self.params[2], self.params[0], self.params[0], self.params[1])
        inputs = Input(input_size)

        if self.params[0] % 8 != 0:
            self.sensor_n_pad = (self.params[0] // 8) * 8 + 8
        else:
            self.sensor_n_pad = self.params[0]

        paddings = tf.constant([[0, 0], [0, 0], [0, self.sensor_n_pad-self.params[0]], 
                                [0, self.sensor_n_pad-self.params[0]], [0, 0]])
        inputs_pad = tf.pad(inputs, paddings)

        conv1 = TimeDistributed(Conv2D(filters = 32, kernel_size = 3, strides = 1, 
                       kernel_initializer='glorot_uniform', padding='same', 
                       activation='selu', name = 'conv1'))(inputs_pad)

        conv2 = TimeDistributed(Conv2D(filters = 64, kernel_size = 3, strides = 2, 
                       kernel_initializer='glorot_uniform', padding='same', 
                       activation='selu', name = 'conv2'))(conv1)

        conv3 = TimeDistributed(Conv2D(filters = 128, kernel_size = 2, strides = 2, 
                       kernel_initializer='glorot_uniform', padding='same', 
                       activation='selu', name = 'conv3'))(conv2)

        conv4 = TimeDistributed(Conv2D(filters = 256, kernel_size = 2, strides = 2, 
                       kernel_initializer='glorot_uniform', padding='same', 
                       activation='selu', name = 'conv4'))(conv3)

        convLSTM1 = ConvLSTM2D(filters = 32, kernel_size = 2, padding = 'same',
                               return_sequences = True, name="convLSTM1")(conv1)
        convLSTM1_out = self.attention(convLSTM1, 1)

        convLSTM2 = ConvLSTM2D(filters = 64, kernel_size = 2, padding = 'same',
                               return_sequences = True, name="convLSTM2")(conv2)
        convLSTM2_out = self.attention(convLSTM2, 2)

        convLSTM3 = ConvLSTM2D(filters = 128, kernel_size = 2, padding = 'same',
                               return_sequences = True, name="convLSTM3")(conv3)
        convLSTM3_out = self.attention(convLSTM3, 4)

        convLSTM4 = ConvLSTM2D(filters = 256, kernel_size = 2, padding = 'same',
                               return_sequences = True, name="convLSTM4")(conv4)
        convLSTM4_out = self.attention(convLSTM4, 8)

        deconv4 = Conv2DTranspose(filters = 128, kernel_size = 2, strides = 2, 
                                  kernel_initializer='glorot_uniform', padding = 'same', 
                                  activation='selu', name = 'deconv4')(convLSTM4_out)
        deconv4_out = tf.concat([deconv4, convLSTM3_out], axis = 3, name = 'concat3')

        deconv3 = Conv2DTranspose(filters = 64, kernel_size = 2, strides = 2,
                                  kernel_initializer='glorot_uniform', padding = 'same', 
                                  activation='selu', name = 'deconv3')(deconv4_out)
        deconv3_out = tf.concat([deconv3, convLSTM2_out], axis = 3, name = 'concat2')

        deconv2 = Conv2DTranspose(filters = 32, kernel_size = 3, strides = 2, 
                                  kernel_initializer='glorot_uniform', padding = 'same', 
                                  activation='selu', name = 'deconv2')(deconv3_out)
        deconv2_out = tf.concat([deconv2, convLSTM1_out], axis = 3, name = 'concat1')

        deconv1 = Conv2DTranspose(filters = self.params[1], kernel_size = 3, strides = 1, 
                                  kernel_initializer='glorot_uniform', padding = 'same', 
                                  activation='selu', name = 'deconv1')(deconv2_out)

        model = Model(inputs = inputs, outputs = deconv1[:, :self.params[0], :self.params[0], :])

        return model
        
    def attention(self, outputs, koef):
        """
        Attention mechanism to weigh the importance of each step in the sequence.

        Parameters
        ----------
        outputs : tf.Tensor
            The output tensor from ConvLSTM layers.
        koef : int
            A coefficient to scale the attention mechanism.

        Returns
        -------
        tf.Tensor
            Weighted output tensor.
        """
        
        attention_w = []
        for k in range(self.params[2]):
            attention_w.append(tf.reduce_sum(tf.multiply(outputs[:,k], outputs[:,-1]), axis=(1,2,3))/self.params[2])
        attention_w = tf.reshape(tf.nn.softmax(tf.stack(attention_w, axis=1)), [-1, 1, self.params[2]])
        outputs = tf.reshape(outputs, [-1, self.params[2], tf.reduce_prod(outputs.shape.as_list()[2:])])
        outputs = tf.matmul(attention_w, outputs)
        outputs = tf.reshape(outputs, 
                             [-1, math.ceil(self.sensor_n_pad/koef), math.ceil(self.sensor_n_pad/koef), 32*koef])
        return outputs
        
    def _Random(self, seed_value): 

        import os
        os.environ['PYTHONHASHSEED'] = str(seed_value)

        import random
        random.seed(seed_value)

        import numpy as np
        np.random.seed(seed_value)

        import tensorflow as tf
        tf.random.set_seed(seed_value)
        
    def _loss_fn(self, y_true, y_pred):
    
        return tf.reduce_mean(tf.square(y_true - y_pred))
    
    def fit(self, X_train, Y_train, X_test, Y_test, batch_size=200, epochs=25): 
        """
        Train the MSCRED model on the provided data.

        Parameters
        ----------
        X_train : numpy.ndarray
            The training input data.
        Y_train : numpy.ndarray
            The training target data.
        X_test : numpy.ndarray
            The testing input data.
        Y_test : numpy.ndarray
            The testing target data.
        batch_size : int, optional
            The batch size for training, by default 200.
        epochs : int, optional
            The number of training epochs, by default 25.
        """

        self.model = self._build_model()
            
        self.model.compile(optimizer = Adam(learning_rate=1e-3),
                  loss = self._loss_fn,
                  )
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.8, 
                                      patience=6, min_lr=0.000001, 
                                      verbose = 1)
        self.model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
                       validation_data = (X_test, Y_test),
                       callbacks=reduce_lr)

    def predict(self, data):
        """
        Generate predictions using the trained MSCRED model.

        Parameters
        ----------
        data : numpy.ndarray
            Input data for generating predictions.

        Returns
        -------
        numpy.ndarray
            Predicted output data.
        """
        
        return self.model.predict(data)