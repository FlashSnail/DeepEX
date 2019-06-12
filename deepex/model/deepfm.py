# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 14:11:47 2019

@author: zhangzehua1
"""

import os
import numpy as np
import tensorflow as tf
from keras.layers import *
from keras.models import Model
from keras.utils import plot_model
from keras import backend as K
os.environ['KERAS_BACKEND'] = 'tensorflow'

def fm(embeddings,numerics,aggregate_flag):
    # First Order
    # sum embedding_fields 
    fields = []
    for field in embeddings:
        tmp = Lambda(lambda x: K.mean(x, axis=-1))(field)
        if aggregate_flag:
            fields.append(Lambda(lambda x: K.sum(x, axis=-1), output_shape=(1,))(tmp))
        else:
            fields.append(tmp)
    
    # sum numerics_fields
    for field in numerics:
        if aggregate_flag:
            fields.append(Lambda(lambda x: K.sum(x, axis=-1), output_shape=(1,))(field))
        else:
            fields.append(field)
    
    # you can use lambda to reduce dimension to (?,1), if so, please set aggregate_flag
    if aggregate_flag:
        first_order = Add()(fields)
        first_order = Reshape((1,), name='first_order')(first_order)
    else:
    # or you can keep the origin tensor
        first_order = concatenate(fields, name='first_order')
    
    # Second Order
    if len(embeddings)>1:
        emb = concatenate(embeddings, axis=1, name='category_embedding_layer')
    else:
        emb = embeddings[0]
        
    summed_features_emb = Lambda(lambda x: K.sum(x, axis=1), name = 'summed_features_emb')(emb) # None * K
    summed_features_emb_square = Multiply(name='summed_features_emb_square')([summed_features_emb,summed_features_emb]) # None * K
    
    squared_features_emb = Multiply(name='squared_features_emb')([emb, emb]) # None * fields * K
    squared_sum_features_emb = Lambda(lambda x: K.sum(x, axis=1), name='squared_sum_features_emb')(squared_features_emb) # Non * K
    
    sub = Subtract()([summed_features_emb_square, squared_sum_features_emb]) # None * K
    sub = Lambda(lambda x:x*0.5)(sub) # None * K
    
    # you can use lambda to reduce dimension to (?,1)
    if aggregate_flag:
        second_order = Lambda(lambda x: K.sum(x, axis=1), output_shape=(1,))(sub) # None * 1
        second_order = Reshape((1,), name='second_order')(second_order)
    else:
    # or you can keep the origin tensor
        second_order = sub
    return first_order, second_order

def deepfm(embeddings,numerics,aggregate_flag,
           deep_model,class_num,inputs,metrics,
           auc,optimizer):
    
    first_order, second_order = fm(embeddings,numerics,aggregate_flag)
    fc7 = concatenate([deep_model,first_order, second_order], name='fc7')
    if class_num > 2:
        model = Dense(class_num, activation='softmax')(fc7)
    elif class_num == 2:
        model = Dense(1, activation='sigmoid')(fc7)
    else:
        model = Dense(1)(fc7)
    model = Model(inputs=inputs,outputs=model)
    
    if class_num > 2:
        loss = 'categorical_crossentropy'
        metrics = ['accuracy'] if metrics is None else metrics
    elif class_num == 2:
        loss = 'binary_crossentropy'
        metrics = [auc] if metrics is None else metrics
    else:
        loss = 'mean_squared_error'
        metrics = ['mse','mae'] if metrics is None else metrics
        
    model.compile(optimizer=optimizer, loss=loss, metrics = metrics)
    model.summary()
    return model, fc7

if __name__ == '__main__':
    pass