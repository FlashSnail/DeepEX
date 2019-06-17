# Updata Log

[TOC]

## v0.0.15

**update time: 2019.06.17**

**More flexible category_index parameter**: Now, this parameter can be a mix type, like [A,B,c], A and B is list but c is a integer. It can achive more flexible way of split field, A and B will be split as col.1 and the rest of element will be split as col.2 with parameter c. Notice: c must be the last element, and the count of remaining element can be evenly divisible by c

## v0.0.14

**update time: 2019.06.12**

Add regression mode support: if class_num < 2, you can get a regression model

## v0.0.13

**update time: 2019.06.12**

Because of custom metrixs are used when model compile, however, when load exists model, compile will assert a error

## v0.0.12

**update time: 2019.06.10**

Fix a bug, embedding way choise Dense, output should be resize (?, 1, embedding_size)

## v0.0.11

**update time: 2019.06.10**

From now，embedding can chose Embedding layer or Dense layer

## v0.0.10

**update time: 2019.06.10**

Fix a activation bug

## v0.0.9

**update time: 2019.06.05**

Fix a bug

## v0.0.8

**update time: 2019.06.05**

1. **Activation**: When declare a `DeepEX` object, custom activation can give to deep network 
2. **FC7**: FC7 layer output can get from `get_fc7_output()` functionq