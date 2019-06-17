# -*- coding: utf-8 -*-
"""

Created on Thu May 30 17:47:10 2019

@author: zhangzehua

@github:

This file define a universal convenient DeepEX frame

DeepEX is a universal convenient frame with keras and Tensorflow. 

You can get well-known Wide&Deep model such as DeepFM here. 

Or, you can define you custom model use this frame.

"""
import os
import numpy as np
import tensorflow as tf
from keras.layers import *
from keras.models import Model
from keras.utils import plot_model
from keras.models import load_model
from keras import backend as K
os.environ['KERAS_BACKEND'] = 'tensorflow'

class DeepEX:
    def __init__(self, data, feature_dim=None, category_index=None,\
                 embedding_dict_size=1000, embedding_size=64, depths_size = [1024,256,64], \
                 class_num=2, aggregate_flag=False, metrics=None, optimizer='Adam', activation='relu', embedding_way='dense'):
        '''
        
        data: np.array
        
        feature_dim: integer, feature dimension
        
        category_index: 1. Can be a 2D list, like [A,B], A and B also a list, if len(A)>1, that means
                        the element of A belong to a field, and will be input embedding layer together.
                        2. Can be a integer, it specific feature_dim % integer == 0, that means split feature
                        as equal intervals with category_index
                        3. If None, all of feature will be embedding
                        
        embedding_dict_size: embedding dict size of categroy feature
        
        embedding_size: embedding size, it make output size like (?, len(category_index), embedding_size)
        
        depths_size: network of deep part parameter, last dimension means fc7 shape
        
        class_num: multi class or binary class
        
        aggregate_flag: if True, first_order and second_order of FM part output as (?,1)
                        if False, output as (?, len(category_index)) and (?, embedding_size)
        
        metrics: can recive custom metrics, if None, binary class use AUC, multi class use auccary
        
        optimizer: Network optimizer, default adam
        
        activation: Network activation, default relu
        
        embedding_way: How network to do embedding, Embedding layer or Dense layer
        
        '''
        assert data is not None, "Sorry, you need give x_data to DeepEX"
        self.data = data
        self.feature_dim = self.data.shape[1] if self.data is not None else feature_dim
        self.category_index = category_index
        self.embedding_dict_size = embedding_dict_size
        self.embedding_size = embedding_size
        self.depths_size = depths_size
        self.class_num = class_num
        self.aggregate_flag = aggregate_flag
        self.metrics = metrics
        self.optimizer = optimizer
        self.deep_model = None
        self.activation = activation
        self.embedding_way = embedding_way
        
        # set self.inputs, self.numerics, self.embeddings, self.embedding_layer
        self.get_embedding_layer()
        # set self.deep_model
        self.deep()
        
        
    def get_fc7_output(self, model_path = None, layer_name = 'fc7', data = None):
        assert model_path is not None, "Sorry, you need give a model path"
        model = load_model(model_path, compile=False)
        if data is None:
            x = self.data_split
        else:
            x = data
        intermediate_layer_model = Model(inputs=model.input,
                                         outputs=model.get_layer(layer_name).output)
        intermediate_output = intermediate_layer_model.predict(x)
        return intermediate_output
        
    def auc(self, y_true, y_pred):
        try:
            auc = tf.metrics.auc(y_true, y_pred)[1]
            K.get_session().run(tf.local_variables_initializer())
            return auc
        except:
            y_true = tf.constant([1,0])
            y_pred = tf.constant([0,1])
            auc = tf.metrics.auc(y_true, y_pred)[1]
            K.get_session().run(tf.local_variables_initializer())
            return auc
    
    def fit(self, model, y, save_model_path = None, batch_size=None, epochs=1, verbose=1, callbacks=None, 
              validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, 
              sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None):
        
        assert (model is not None) and (y is not None), "Sorry, fit() need model and label"
        # split x to N-d as inputs shape 
        x = self.data_split
        # call keras fit function
        model.fit(x, y, batch_size, epochs, verbose, callbacks, 
              validation_split, validation_data, shuffle, class_weight, 
              sample_weight, initial_epoch, steps_per_epoch, validation_steps)
    
        if save_model_path:
            model.save(save_model_path)
            print ("model has been saved at "+ str(save_model_path))
        else:
            print ("because of save_model_path is None, model is not saved")
    
    
    def get_embedding_layer(self):
        self.data_split = []
        inputs = []
        numerics = []
        embeddings=[]
        if self.category_index:
            # if category_index is integer, split over a specified interval.
            if isinstance(self.category_index,int):
                assert self.feature_dim % self.category_index == 0, "Sorry, if category_index is a integer, condition (feature_dim % category_index == 0) is necessary"
                category_index = []
                start = 0
                for i in range(int(self.feature_dim / self.category_index)):
                    category_index.append(list(range(start, start+self.category_index)))
                    start += self.category_index
                self.category_index = category_index
                
            # if category_index is list, deep will embedding feature in list
            start = 0
            for index, field in enumerate(self.category_index):
                # split input_numeric and input_category, after embedding, concat these layers
                
                if isinstance(field, int):
                    # For fix type category_index, integer must be the last element
                    assert index == len(self.category_index)-1, "Sorry, for fix type category_index, integer must be the last element"
                    assert (self.feature_dim-start) % field == 0, "Sorry, remaining element cannot be evenly divisible by %d." % field
                    for i in range(int((self.feature_dim-start) / field)):
                        self.category_index.append(list(range(start+field*i, start+field*(i+1))))
                    continue
                
                end = field[0]
                #numeric feature
                if end > start:
                    input_numeric = Input(shape=(end-start,), name='input_numeric'+str(index))
                    numerics.append(input_numeric)
                    self.data_split.append(self.data[:,start:end])
                #category feature
                input_category = Input(shape=(len(field),), name='input_category'+str(index))
                if self.embedding_way == 'dense':
                    embedding = Dense(self.embedding_size, activation='tanh', name='embedding'+str(index))(input_category)
                    embedding = Reshape((1,self.embedding_size,))(embedding)
                else:
                    embedding = Embedding(self.embedding_dict_size, self.embedding_size,\
                                          input_length=len(field), name='embedding'+str(index))(input_category)
                embeddings.append(embedding)

                '''
                here is a keras problem, Flatten will make output shape become (?,?).
                
                Flatten is basically Reshape((-1, prod(x.shape[1:]))) the result from the product 
                is a Tensor which the value is not known at the graph creation.
                
                a workaround is working here:
                '''
                shape = int(np.prod(embedding.shape[1:]))
                embedding = Reshape((shape,))(embedding)
                
                if index == 0:
                    if end > start:
                        embedding_layer = concatenate([input_numeric,embedding], name='embedding_layer'+str(index))
                    else:
                        embedding_layer = embedding
                else:
                    if end > start:
                        embedding_layer = concatenate([embedding_layer, input_numeric, embedding], name='embedding_layer'+str(index))
                    else:
                        embedding_layer = concatenate([embedding_layer, embedding], name='embedding_layer'+str(index))
                
                # save in the input list
                if end > start:
                    inputs.append(input_numeric)
                else:
                    pass
                inputs.append(input_category)
                start = field[-1]+1
                self.data_split.append(self.data[:,end:start])
            
            # if there are numeric featture behind last field of category
            start = self.category_index[-1][-1]+1
            end = self.feature_dim
            if end > start:
                input_numeric = Input(shape=(end-start,), name='input_numeric_last')
                inputs.append(input_numeric)
                numerics.append(input_numeric)
                self.data_split.append(self.data[:,start:end])
                embedding_layer = concatenate([embedding_layer, input_numeric], name='embedding_layer_last')
        else:
            # if category_index is None, embedding all of feature
            embeddings_reshape = []
            for i in range(self.feature_dim):
                input_layer = Input(shape=(1,), name='input_layer'+str(i))
                inputs.append(input_layer)
                self.data_split.append(self.data[:,i])
                if self.embedding_way == 'dense':
                    embedding = Dense(self.embedding_size, activation='tanh', name='embedding'+str(i))(input_layer)
                    embedding = Reshape((1,self.embedding_size,))(embedding)
                else:
                    embedding = Embedding(self.embedding_dict_size, self.embedding_size,\
                                          input_length=1, name='embedding'+str(i))(input_layer)
                embeddings.append(embedding)
                shape = int(np.prod(embedding.shape[1:]))
                embedding = Reshape((shape,))(embedding)
                embeddings_reshape.append(embedding)
            
            embedding_layer = concatenate(embeddings_reshape, name='embedding_layer')
        
        self.inputs = inputs
        self.numerics = numerics
        self.embeddings = embeddings
        self.embedding_layer = embedding_layer
        return self.inputs, self.numerics, self.embeddings, self.embedding_layer
    
    def deep(self):
        model = Dense(self.depths_size[0])(self.embedding_layer)
        model = Activation(self.activation)(model)
        for depth in self.depths_size[1:]:
            model = Dense(depth)(model)
            model = Activation(self.activation)(model)
        self.deep_model = model
        return model
    
    def deepfm(self):
        from .model import deepfm
        model, self.fc7 = deepfm.deepfm(self.embeddings,self.numerics,self.aggregate_flag,
                      self.deep_model,self.class_num,self.inputs,self.metrics,
                      self.auc,self.optimizer)
        return model
    
if __name__ == '__main__':
    
    '''
    # HOW TO USE, here is a demo
    # However, if you run this code directly, you should remove dot at "from .model import deepfm" 
    '''
    
    samples = 100000    # set samples num
    feat_dim = 10   # set feat_dim 
    cate = np.random.randint(1,6,samples)   # set a categroy feat randomly
    x = np.random.random((samples,feat_dim))    # generate feat randomly
    x[:,3] = cate   # chose a column to be categroy feat
    y = np.random.randint(0,2,samples)  # generate label
    
    # declare DeepEX objects
    deepEX = DeepEX(data = x, feature_dim=feat_dim, category_index=[[0,1],4], embedding_dict_size=1000, 
                  embedding_size=64, depths_size = [1024,256,64], class_num=2, 
                  aggregate_flag=False, metrics=None, optimizer='Adam', activation='relu', embedding_way='emb')
    
    model = deepEX.deepfm()  # get DeepFM
    plot_model(model,'deepFM.png',show_shapes=True) # show model graph
    
    # train deepfm
    path = 'deepfm.model'
    deepEX.fit(model, y, save_model_path=path, batch_size=None, epochs=1, verbose=1, callbacks=None, 
              validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, 
              sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
    
    # get fc7 output tensor
    fc7 = deepEX.get_fc7_output(model_path=path, layer_name='fc7', data=deepEX.data_split)
    
    
