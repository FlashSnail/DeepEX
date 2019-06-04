# DeepEX

[TOC]

## Overview

**DeepEX**  is a universal convenient frame with keras and Tensorflow,

You can get well-known Wide&Deep model such as DeepFM here. 

Or, you can define you custom model use this frame.





### How to Install

* For Linux:

```shell
pip install DeepEX
```

* For Windows:

You need active cmd first and, 

```powershell
pip install DeepEX
```

Notice: This frame needs keras and tensorflow, maybe it has some problem on windows with python 2.x because of tensorflow.





## DeepEX Class API

In the functional API, given some parmeters, you can instantiate a `DeepEX` object, via:

```python
DeepEX(data = None, feature_dim=None, category_index=None,embedding_dict_size=1000,
      embedding_size=64, depths_size = [1024,256,64],class_num=2,
      aggregate_flag=False, metrics=None, optimizer='Adam')
```

Only data is necessary, other parmeters have default value.

**Arguments**

* **data**: np.array

* **feature_dim**: integer, feature dimension

* **category_index**: 
  * Can be a 2D list, like [A,B], A and B also a list, if len(A)>1, that means the element of A belong to a field, and will be input embedding layer together.
  * Can be a integer, it specific feature_dim % integer == 0, that means split feature as equal intervals with category_index
  * If None, all of feature will be embedding

* **embedding_dict_size**: embedding dict size of categroy feature

* **embedding_size**: embedding size, it make output size like (?, len(category_index), embedding_size)

* **depths_size**: network of deep part parameter, last dimension means fc7 shape

* **class_num**: multi class or binary class

* **aggregate_flag**: 
  * if True, first_order and second_order of FM part output as (?,1)
  * if False, output as (?, len(category_index)) and (?, embedding_size)

* **metrics**: can recive custom metrics, if None, binary class use AUC, multi class use auccary

* **optimizer**: Network optimizer, default adam





## Methods

### **get_embedding_layer**

embedding input data as class parameter. 

**return**

* inputs: A list, which elements are Input layer,  prepare to deep model
* numerics: A list, which elements are numeric feature tensor
* embeddings:  A list, which elements are categroy feature embedding tensor
* embedding_layer: A tensor, which is concate numeric feature tensor and categroy feature embedding tensor

-----

### fm

get fm part

**return**

- A tensor, shape depends on class parameter **aggregate_flag**

-----

### deep

get deep part

**return**

* inputs: A list, which elements are Input layer,  prepare to deep model
* model: A tensor, which is also a keras functional layer

-----

### deepfm

get deepfm model

**return**

* model: A keras functional model

-----

### auc

```python
auc(y_true, y_pred)
```

A custom metrics, when class_num=2, use this metrics to evaluate model

-----

### fit

```python
fit(model, y, save_model_path = None, batch_size=None, epochs=1, verbose=1,
    callbacks=None,validation_split=0.0, validation_data=None,
    shuffle=True, class_weight=None,sample_weight=None, 
    initial_epoch=0, steps_per_epoch=None, validation_steps=None)
```

fit data to train model

**Arguments**

* **model**: a DeepEX model
* **save_model_path**: A string, where model to save, if None, model will not be saved
* **others**: see document [keras fit](https://keras.io/models/model/#fit)





## How to Use

This class is very easy to use,  three steps to go：

```python
# step 1, declare DeepEX object
DeepEX = DeepEX(...)

# step 2, get model you want
model = DeepEX.deepfm()

# step 3, train and save
DeepEX.fit(...)
```

### example

```python
from deepex import *
import numpy as np

samples = 100000    # set samples num
feat_dim = 10   # set feat_dim 
cate = np.random.randint(1,6,samples)   # set a categroy feat randomly
x = np.random.random((samples,feat_dim))    # generate feat randomly
x[:,3] = cate   # chose a column to be categroy feat
y = np.random.randint(0,2,samples)  # generate label

# declare DeepEX objects
deepEX = DeepEX(data = x, feature_dim=feat_dim, category_index=2, embedding_dict_size=1000, 
embedding_size=64, depths_size = [1024,256,64], class_num=2, 
aggregate_flag=False, metrics=None, optimizer='Adam')

model = deepEX.deepfm()  # get DeepFM
plot_model(model,'deepFM.png',show_shapes=True) # show model graph

# train deepfm
deepEX.fit(model, y, save_model_path='deepfm.model', batch_size=None, epochs=1, verbose=1, callbacks=None, 
validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, 
sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
```

