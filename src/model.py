import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, InputLayer, Input, add, dot, maximum, average, multiply
from tensorflow.keras.layers import concatenate, Reshape, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

import numpy as np
import pandas as pd

def circulant(shape):
    matrix_shape = shape.get_shape().as_list()
    circulant_set = []
    circulant_ = K.expand_dims(shape, 2)
    circulant_set.append(circulant_)

    for i in range(1, matrix_shape[1]):
        pre=circulant_[:,(matrix_shape[1]-i):,:]
        host=circulant_[:,0:(matrix_shape[1]-i),:]
        circulant_1=tf.concat([pre,host],1)
        circulant_set.append(circulant_1)

    vector_ = K.concatenate(circulant_set)

    return vector_

class Circulant_layer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Circulant_layer, self).__init__(**kwargs)
   
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[0][1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(Circulant_layer, self).build(input_shape) 

    def call(self, x):
        assert isinstance(x, list)
        des, body = x
        
        des = K.dot(des, self.kernel) 
        body = K.dot(body, self.kernel)
        
        multiple_vector1 = K.batch_dot(circulant(body), des)
        multiple_vector2 = K.batch_dot(circulant(des), body)
        
        sum_vector = multiple_vector1 + multiple_vector2
        sum_vector_1 = K.dot(sum_vector, self.kernel)
        
        return sum_vector
    
class network:
    def __init__(self, des_shape, body_shape, hid_layer, output_layer):
        self.des_shape = des_shape
        self.body_shape = body_shape
        self.hid_layer = hid_layer  
        self.output_layer = output_layer
        
    ''' OPCNN '''    
    def OPCNN(self):
        # des modality
        des_input = Input(shape = (self.des_shape)) 
        body_input = Input(shape = (self.body_shape))
        
        des_fc = Dense(self.hid_layer[2])(des_input)
        body_fc = Dense(self.hid_layer[2])(body_input)
        
        des_reshape_2 = Reshape((1,self.hid_layer[2]))(des_fc)
        body_reshape_2 = Reshape((1,self.hid_layer[2]))(body_fc)
        
        dot_layer = dot([des_reshape_2, body_reshape_2], axes = 1, normalize=False)
        dot_reshape = Reshape((self.hid_layer[2], self.hid_layer[2], 1))(dot_layer)
        
        res_a = Conv2D(32, (3, 3), activation='relu', padding="same")(dot_reshape)
        res_a_1 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_a)
        res_a_2 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_a_1)      
        res_ab = res_a + res_a_2
        
        res_b = Conv2D(32, (3, 3), activation='relu', padding="same")(res_ab)
        res_b_1 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_b)
        res_b_2 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_b_1)
        res_bc = res_b + res_b_2
        
        res_c = Conv2D(32, (3, 3), activation='relu', padding="same")(res_bc)
        res_c_1 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_c)
        res_c_2 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_c_1)
        res_cc = res_c + res_c_2

        tensor_flat = Flatten()(res_cc)

        x = Dense(self.hid_layer[2] , activation = 'relu')(tensor_flat)
        x = BatchNormalization()(x)

        x = Dense(self.hid_layer[2] , activation = 'relu')(x)
        x = BatchNormalization()(x)
        
        x = Dense(self.output_layer , activation = 'sigmoid')(x)
        
        OPCNN = keras.models.Model(inputs = [des_input, body_input], outputs = [x])  

        return OPCNN
        
    ''' DMNN Early Fusion Addtion model '''
    def Early_Add(self):
        # des modality
        des_input = Input(shape = self.des_shape)
        des_input_x2 = Dense(self.hid_layer[2], activation = 'relu')(des_input)
        des_input_x2_batch = BatchNormalization()(des_input_x2)
        
        # body modality
        body_input = Input(shape = self.body_shape) 
        body_input_x2 = Dense(self.hid_layer[2], activation = 'relu')(body_input)
        body_input_x2_batch = BatchNormalization()(body_input_x2)

        merge_layer = add([des_input_x2_batch , body_input_x2_batch])
        
        x = Dense(hid_layer[0] , activation = 'relu')(merge_layer)
        x = BatchNormalization()(x)

        x = Dense(hid_layer[1] , activation = 'relu')(x)
        x = BatchNormalization()(x)

        x = Dense(hid_layer[2] , activation = 'relu')(x)
        x = BatchNormalization()(x)
        
        x = Dense(self.output_layer , activation = 'sigmoid')(x)
        
        early_add = keras.models.Model(inputs = [des_input, body_input], outputs = [x])

        return early_add
    
    ''' DMNN Early Fusion Product model '''
    def Early_Product(self):
        # des modality
        des_input = Input(shape = self.des_shape)
        des_input_x2 = Dense(self.hid_layer[2], activation = 'relu')(des_input)
        des_input_x2_batch = BatchNormalization()(des_input_x2)
        
        # body modality
        body_input = Input(shape = self.body_shape) 
        body_input_x2 = Dense(self.hid_layer[2], activation = 'relu')(body_input)
        body_input_x2_batch = BatchNormalization()(body_input_x2)

        merge_layer = dot([des_input_x2_batch , body_input_x2_batch])
        
        x = Dense(hid_layer[0] , activation = 'relu')(merge_layer)
        x = BatchNormalization()(x)

        x = Dense(hid_layer[1] , activation = 'relu')(x)
        x = BatchNormalization()(x)

        x = Dense(hid_layer[2] , activation = 'relu')(x)
        x = BatchNormalization()(x)
        
        x = Dense(self.output_layer , activation = 'sigmoid')(x)
        
        early_product = keras.models.Model(inputs = [des_input, body_input], outputs = [x])

        return early_product
    
    ''' DMNN Early Fusion Concatenate model '''
    def Early_Concat(self):
        # des modality
        des_input = Input(shape = self.des_shape) 
        des_input_x2 = Dense(self.hid_layer[2], activation = 'relu')(des_input)
        des_input_x2_batch = BatchNormalization()(des_input_x2)
        
        # body modality
        body_input = Input(shape = self.body_shape)
        body_input_x2 = Dense(self.hid_layer[2], activation = 'relu')(body_input)
        body_input_x2_batch = BatchNormalization()(body_input_x2)
           
        merge_layer = concatenate([des_input_x2_batch , body_input_x2_batch])
        
        x = Dense(hid_layer[0] , activation = 'relu')(merge_layer)
        x = BatchNormalization()(x)

        x = Dense(hid_layer[1] , activation = 'relu')(x)
        x = BatchNormalization()(x)

        x = Dense(hid_layer[2] , activation = 'relu')(x)
        x = BatchNormalization()(x)
        
        x = Dense(self.output_layer , activation = 'sigmoid')(x)
        
        early_concat = keras.models.Model(inputs = [des_input, body_input], outputs = [x])

        return early_concat
    
    ''' DMNN Intermediate Fusion Addition model '''
    def Intermediate_Add(self):
        # des modality
        des_input = Input(shape = self.des_shape) 
        des_input_x2 = Dense(self.hid_layer[0], activation = 'relu')(des_input)
        des_input_x2_batch = BatchNormalization()(des_input_x2)
        des_input_x3 = Dense(self.hid_layer[1], activation = 'relu')(des_input_x2_batch)
        des_input_x3_batch = BatchNormalization()(des_input_x3)
        des_input_x4 = Dense(self.hid_layer[2], activation = 'relu')(des_input_x3_batch)
        des_input_x4_batch = BatchNormalization()(des_input_x4)
        
        # body modality
        body_input = Input(shape = self.body_shape) 
        body_input_x2 = Dense(self.hid_layer[0], activation = 'relu')(body_input)
        body_input_x2_batch = BatchNormalization()(body_input_x2)
        body_input_x3 = Dense(self.hid_layer[1], activation = 'relu')(body_input_x2_batch)
        body_input_x3_batch = BatchNormalization()(body_input_x3)
        body_input_x4 = Dense(self.hid_layer[2], activation = 'relu')(body_input_x3_batch)
        body_input_x4_batch = BatchNormalization()(body_input_x4)
        
        addition_layer = keras.layers.add([des_input_x4_batch, body_input_x4_batch])
        addition_input_x = Dense(self.output_layer, activation = 'sigmoid')(addition_layer)
        inter_add = keras.models.Model(inputs = [des_input, body_input], outputs = [addition_input_x])

        return inter_add
    
    ''' DMNN Intermediate Fusion Product model '''
    def Intermediate_Product(self):
        # des modality
        des_input = Input(shape = self.des_shape) 
        des_input_x2 = Dense(self.hid_layer[0], activation = 'relu')(des_input)
        des_input_x2_batch = BatchNormalization()(des_input_x2)
        des_input_x3 = Dense(self.hid_layer[1], activation = 'relu')(des_input_x2_batch)
        des_input_x3_batch = BatchNormalization()(des_input_x3)
        des_input_x4 = Dense(self.hid_layer[2], activation = 'relu')(des_input_x3_batch)
        des_input_x4_batch = BatchNormalization()(des_input_x4)
        
        # body modality
        body_input = Input(shape = self.body_shape) 
        body_input_x2 = Dense(self.hid_layer[0], activation = 'relu')(body_input)
        body_input_x2_batch = BatchNormalization()(body_input_x2)
        body_input_x3 = Dense(self.hid_layer[1], activation = 'relu')(body_input_x2_batch)
        body_input_x3_batch = BatchNormalization()(body_input_x3)
        body_input_x4 = Dense(self.hid_layer[2], activation = 'relu')(body_input_x3_batch)
        body_input_x4_batch = BatchNormalization()(body_input_x4)

        product_layer = multiply([des_input_x4_batch, body_input_x4_batch])       
        product_input_x = Dense(self.output_layer, activation = 'sigmoid')(product_layer)
        inter_product = keras.models.Model(inputs = [des_input, body_input], outputs = [product_input_x])

        return inter_product
    
    ''' DMNN Intermediate Fusion Concatenate model '''
    def Intermediate_Concat(self):
        # des modality
        des_input = Input(shape = self.des_shape) 
        des_input_x2 = Dense(self.hid_layer[0], activation = 'relu')(des_input)
        des_input_x2_batch = BatchNormalization()(des_input_x2)
        des_input_x3 = Dense(self.hid_layer[1], activation = 'relu')(des_input_x2_batch)
        des_input_x3_batch = BatchNormalization()(des_input_x3)
        des_input_x4 = Dense(self.hid_layer[2], activation = 'relu')(des_input_x3_batch)
        des_input_x4_batch = BatchNormalization()(des_input_x4)
        
        # body modality
        body_input = Input(shape = self.body_shape) 
        body_input_x2 = Dense(self.hid_layer[0], activation = 'relu')(body_input)
        body_input_x2_batch = BatchNormalization()(body_input_x2)
        body_input_x3 = Dense(self.hid_layer[1], activation = 'relu')(body_input_x2_batch)
        body_input_x3_batch = BatchNormalization()(body_input_x3)
        body_input_x4 = Dense(self.hid_layer[2], activation = 'relu')(body_input_x3_batch)
        body_input_x4_batch = BatchNormalization()(body_input_x4)

        concat_layer = concatenate([des_input_x4_batch, body_input_x4_batch])        
        concat_input_x = Dense(self.output_layer, activation = 'sigmoid')(concat_layer)
        inter_concat = keras.models.Model(inputs = [des_input, body_input], outputs = [concat_input_x])

        return inter_concat
    
    ''' DMNN Intermediate Fusion TFL model '''
    def Intermediate_TFL(self):
        # bias modality
        bias_input = Input(shape = (1,))
        
        # des modality
        des_input = Input(shape = self.des_shape) 
        des_input_x1 = Dense(self.hid_layer[0], activation = 'relu')(des_input)
        des_input_x1_batch = BatchNormalization()(des_input_x1)
        des_input_x3 = Dense(self.hid_layer[2], activation = 'relu')(des_input_x1_batch)
        des_input_x3_batch = BatchNormalization()(des_input_x3)
        
        # body modality
        body_input = Input(shape = self.body_shape) 
        body_input_x1 = Dense(self.hid_layer[0], activation = 'relu')(body_input)
        body_input_x1_batch = BatchNormalization()(body_input_x1)
        body_input_x3 = Dense(self.hid_layer[2], activation = 'relu')(body_input_x1_batch)
        body_input_x3_batch = BatchNormalization()(body_input_x3)
        
        concat_layer1 = concatenate([bias_input, des_input_x3_batch])
        concat_layer2 = concatenate([bias_input, body_input_x3_batch])
        
        re_concat1 = Reshape((1, 51))(concat_layer1)
        re_concat2 = Reshape((1, 51))(concat_layer2)
        
        dot_layer = dot([re_concat1, re_concat2], axes = 1, normalize=False)
        tensor_flat = Flatten()(dot_layer)
        fusion_dense = Dense(self.hid_layer[2], activation = 'relu')(tensor_flat)
        fusion_out = Dense(self.output_layer, activation = 'sigmoid')(fusion_dense)
        
        inter_TFL = keras.models.Model(inputs = [bias_input, des_input, body_input], outputs = [fusion_out])
        
        return inter_TFL
    
    ''' DMNN Intermediate Fusion MCF model '''
    def Intermediate_MCF(self):
        # des modality
        des_input = Input(shape = self.des_shape) 
        des_input_x1 = Dense(self.hid_layer[0], activation = 'relu')(des_input)
        des_input_x1_batch = BatchNormalization()(des_input_x1)
        des_input_x2 = Dense(self.hid_layer[2], activation = 'relu')(des_input_x1_batch)
        des_input_x2_batch = BatchNormalization()(des_input_x2)

        # body modality
        body_input = Input(shape = self.body_shape)
        body_input_x1 = Dense(self.hid_layer[0], activation = 'relu')(body_input)
        body_input_x1_batch = BatchNormalization()(body_input_x1)
        body_input_x2 = Dense(self.hid_layer[2], activation = 'relu')(body_input_x1_batch)
        body_input_x2_batch = BatchNormalization()(body_input_x2)

        
        fusion = Circulant_layer(self.hid_layer[2])([des_input_x2_batch, body_input_x2_batch])
        fusion_out = Dense(self.output_layer, activation = 'sigmoid')(fusion)
        inter_MCF = keras.models.Model(inputs = [des_input, body_input], outputs = [fusion_out])
        
        return inter_MCF
    
    ''' DMNN Late Fusion Concatenate model '''
    def Late_Concat(self):
        # des modality
        des_input = Input(shape = (self.des_shape,))
        des_input_x1_batch = BatchNormalization()(des_input)
        des_input_x2 = Dense(self.hid_layer[0], activation = 'relu')(des_input_x1_batch)
        des_input_x2_batch = BatchNormalization()(des_input_x2)
        des_input_x3 = Dense(self.hid_layer[1], activation = 'relu')(des_input_x2_batch)
        des_input_x3_batch = BatchNormalization()(des_input_x3)
        des_input_x4 = Dense(self.hid_layer[2], activation = 'relu')(des_input_x3_batch)
        des_input_x4_batch = BatchNormalization()(des_input_x4)
        des_output = Dense(self.output_layer , activation = 'sigmoid')(des_input_x4_batch)

        # body modality
        body_input = Input(shape = (self.body_shape,)) 
        body_input_x1_batch = BatchNormalization()(body_input)
        body_input_x2 = Dense(self.hid_layer[0], activation = 'relu')(body_input_x1_batch)
        body_input_x2_batch = BatchNormalization()(body_input_x2)
        body_input_x3 = Dense(self.hid_layer[1], activation = 'relu')(body_input_x2_batch)
        body_input_x3_batch = BatchNormalization()(body_input_x3)
        body_input_x4 = Dense(self.hid_layer[2], activation = 'relu')(body_input_x3_batch)
        body_input_x4_batch = BatchNormalization()(body_input_x4)
        body_output = Dense(self.output_layer , activation = 'sigmoid')(body_input_x4_batch)

        concat_layer = concatenate([des_output , body_output])
        f_layer = Dense(self.hid_layer[2], activation = 'relu')(concat_layer)
        concat_input_x = Dense(self.output_layer, activation = 'sigmoid')(f_layer)
        late_concat = keras.models.Model(inputs = [des_input, body_input], outputs = [concat_input_x])

        return late_concat