 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 17:45:24 2020

@author: Ching
"""
#double brackets return dataframes

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import keras
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.layers import *

from keras.applications import *
from keras.layers.core import Lambda, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from sklearn.metrics import accuracy_score
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.models import *
from keras.optimizers import SGD, Adam
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, History
from PIL import Image
from PIL import ImageFilter
from skimage.io import imshow
from skimage.util import random_noise
from sklearn.utils import class_weight
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix, precision_recall_curve,recall_score,average_precision_score
import segmentation_models as sm
import skimage.transform  
from sklearn.cluster import KMeans

import albumentations
from ImageDataAugmentor.image_data_augmentor import *
import json
from keras_radam import RAdam

keras.backend.set_image_data_format('channels_last')
global defeature, enfeature
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
def class_tversky(y_true, y_pred):
    smooth = 1

    y_true = K.permute_dimensions(y_true, (3,1,2,0))
    y_pred = K.permute_dimensions(y_pred, (3,1,2,0))
    #print(K.eval(y_pred).shape)
    y_true_pos = K.batch_flatten(y_true)
    y_pred_pos = K.batch_flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos, 1)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos), 1)
    false_pos = K.sum((1-y_true_pos)*y_pred_pos, 1)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky(y_true, y_pred, smooth=1e-6, alpha=0.9):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + 0.7 * false_neg + 0.3 * false_pos + smooth)

def focal_tversky_loss(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.sum(K.pow((1-pt_1), gamma))

def dsc(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dsc(y_true, y_pred)
    return loss

def plot_pixels(data, title, colors = None, N = 100000):
    if colors is None:
        colors = data
    
    rng = np.random.RandomState(0)
    i = rng.permutation(data.shape[0])[:N]
    colors = colors[i]
    pixel = data[i].T
    print(pixel.shape)

    R, G, B = pixel[0], pixel[1], pixel[2]

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d', xlabel = 'Red', zlabel = 'Blue', ylabel = 'Green', )
    ax.scatter3D( R, G, B, c=colors, )

    fig.suptitle(title, size=20)
    #plt.show()


IMAGE_H = 256
IMAGE_W = 256
IMAGE_C = 3

train_path = 'cityscapes/train/'
val_path = 'cityscapes/val/'

train_data = next(os.walk(train_path))[2]
val_data = next(os.walk(val_path))[2]

X_train = np.zeros((len(train_data),IMAGE_H,IMAGE_W,IMAGE_C),dtype=np.int32)
Y_train = np.zeros((len(train_data),IMAGE_H,IMAGE_W,3),dtype=np.int32)
X_test = np.zeros((len(val_data),IMAGE_H,IMAGE_W,IMAGE_C),dtype=np.int32)
Y_test = np.zeros((len(val_data),IMAGE_H,IMAGE_W,3),dtype=np.int32)

for n, image in enumerate(train_data, 0):
  temp_image = cv2.imread(train_path + image,1)
  X_train[n] = temp_image[:,:256,:]
  Y_train[n] = temp_image[:,256:,:]

for n, image in enumerate(val_data, 0):
  temp_image = cv2.imread(train_path + image,1)
  X_test[n] = temp_image[:,:256,:]
  Y_test[n] = temp_image[:,256:,:]


X_train = X_train/255.0
X_train = X_train - np.array([0.485, 0.456, 0.406])
X_train = X_train/np.array([0.229, 0.224, 0.225])
X_train = X_train.astype(np.float32)


X_test = X_test/255.0
X_test = X_test - np.array([0.485, 0.456, 0.406])
X_test = X_test/np.array([0.229, 0.224, 0.225])
X_test = X_test.astype(np.float32)

temp_train = np.zeros((len(train_data),IMAGE_H,IMAGE_W,1),dtype=np.int32)
temp_test = np.zeros((len(val_data),IMAGE_H,IMAGE_W,1),dtype=np.int32)

####################No normalization & No Reduced Color Spaces (Best)
num_item = 2000
color_array = np.random.choice(range(256), 3*num_item).reshape(-1,3)
print(len(color_array))
num_classes = 25
label_model = KMeans(n_clusters = num_classes)
s = label_model.fit(color_array)

for idx, i in enumerate(Y_train,0):
    temp_Y = label_model.predict(i.reshape(-1,3)).reshape(256,256)
    temp_train[idx] = np.reshape(temp_Y,(256,256,-1))


for idx, i in enumerate(Y_test,0):
    temp_Y = label_model.predict(i.reshape(-1,3)).reshape(256,256)
    temp_test[idx] = np.reshape(temp_Y,(256,256,-1))
####################No normalization & No Reduced Color Spaces

####################Normalization & Reduced Color Spaces
'''num_item = 2000
color_array = np.random.choice(range(256), 3*num_item).reshape(-1,3)/255.0
num_classes = 30
label_model = KMeans(n_clusters = num_classes)
s = label_model.fit(color_array)

img_data = np.array([Y_train[i].reshape(-1, 3) for i in range(len(Y_train))]) / 255.0
img_data = np.concatenate(img_data).astype(np.float32)
#plot_pixels(img_data, title='all colors in mask')

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS
img_data = img_data.astype(np.float32)
campactness, labels, centers = cv2.kmeans(img_data, 30, None, criteria, 10,flags)
new_colors = centers[labels].reshape((-1, 3))
#plot_pixels(img_data, colors = new_colors, title = "Reduced color space: 30 colors")

new_colors = new_colors.reshape(-1,256,256,3)
new_colors = new_colors.astype(np.float64)

for idx, i in enumerate(new_colors,0):
    temp_Y = label_model.predict(i.reshape(-1,3)).reshape(256,256)
    temp_train[idx] = np.reshape(temp_Y,(256,256,-1))

X2 = temp_train
img_data = np.array([Y_test[i].reshape(-1, 3) for i in range(len(Y_test))]) / 255.0
img_data = np.concatenate(img_data).astype(np.float32)
#plot_pixels(img_data, title='all colors in mask')

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS
img_data = img_data.astype(np.float32)
campactness, labels, centers = cv2.kmeans(img_data, 30, None, criteria, 10,flags)
new_colors = centers[labels].reshape((-1, 3))

new_colors = new_colors.reshape(-1,256,256,3)
new_colors = new_colors.astype(np.float64)

for idx, i in enumerate(new_colors,0):
    temp_Y = label_model.predict(i.reshape(-1,3)).reshape(256,256)
    temp_test[idx] = np.reshape(temp_Y,(256,256,-1))'''

####################Normalization & Reduced Color Spaces


X2_ = temp_test
X2 = temp_train
def inverse_trasform(train_data):
    new = train_data*(np.array([0.229,0.224,0.225]))
    new = new+(np.array([0.485, 0.456, 0.406]))
    print(new[0])
    return new

X1 = X_train
X1_ = X_test

new = inverse_trasform(X1)


x_tra, x_val, y_tra, y_val = train_test_split(X1, X2, test_size = 0.1, shuffle=False)




'''Aug = albumentations.Compose([
            albumentations.OneOf([               
                  albumentations.HorizontalFlip(p=0.5),
                  albumentations.VerticalFlip(p=0.5)])],p=1)

training_datagen = ImageDataAugmentor(
    augment = Aug)
mask_datagen = ImageDataAugmentor(augment= Aug)'''
Aug = albumentations.Compose([])

training_datagen = ImageDataAugmentor()
mask_datagen = ImageDataAugmentor()
validation_training_datagen = ImageDataAugmentor()
validation_mask_datagen = ImageDataAugmentor()
testing_datagen = ImageDataAugmentor()


image_data_augmentator = training_datagen.flow(x_tra, batch_size=8, shuffle=False)
mask_data_augmentator = mask_datagen.flow(y_tra,batch_size=8,shuffle=False)
'''for i in mask_data_augmentator:
    for j in i:
        print(j)
        imshow(j/255)
        plt.show()'''



model = sm.Unet('resnet34', input_shape = (None,None,3), classes=30, activation='softmax', encoder_weights='imagenet')
model.compile(optimizer=Adam(lr=3e-4),loss='sparse_categorical_crossentropy', metrics =['sparse_categorical_accuracy'])

val_image_data_augmentator = validation_training_datagen.flow(x_val, batch_size=8, shuffle=False)
val_mask_data_augmentator = validation_mask_datagen.flow(y_val,batch_size=8,shuffle=False)
#true_y = X2_.astype(np.uint8)
#true_y = true_y.astype(np.float32)

training_data_generator = zip(image_data_augmentator, mask_data_augmentator)
val_training_data_generator = zip(val_image_data_augmentator, val_mask_data_augmentator)
#unet.fit_generator(training_data_generator, steps_per_epoch=len(x_tra)//16, epochs=5)
#from segmentation_models.utils import set_trainable
#set_trainable(unet)
#results = model.fit_generator(training_data_generator, steps_per_epoch=len(x_tra)//8, epochs=60)
results = model.fit_generator(training_data_generator, steps_per_epoch=len(x_tra)//8, epochs=100, validation_data=val_training_data_generator, validation_steps=len(x_val)//8)

#

testing_data_augmentator = testing_datagen.flow(X1_, batch_size=8, shuffle=False)
preds_train = model.predict_generator(testing_data_augmentator, verbose = 1)

#y_preds = np.argmax(preds_train,axis=-1)

'''for i in range(n_samples):
    plt.subplot(3, n_samples, 1 + i)
    plt.axis('off')
    plt.imshow(X_realA[i],cmap='gray')
    # plot generated target image
for i in range(n_samples):
    plt.subplot(3, n_samples, 1 + n_samples + i)
    plt.axis('off')
    plt.imshow(X_fakeB[i],cmap='gray')
    # plot real target image
for i in range(n_samples):
    plt.subplot(3, n_samples, 1 + n_samples*2 + i)
    plt.axis('off')
    plt.imshow(X_realB[i],cmap='gray')
    # save plot to file
filename1 = 'plott2(256*256)_%06d.png' % (step+1)
plt.savefig(filename1)
plt.close()'''

y_preds = np.argmax(preds_train,axis=-1)

'''for i in range(10):
    output = (X2_[i].transpose(2,0,1).reshape(3,-1)).astype(np.uint8)
    cv2.imwrite('cityscapes(gt)_%d(dice).jpg' % (i), output)
    output = (y_preds[i]).astype(np.uint8)
    cv2.imwrite('cityscapes(pred)_%d(dice).jpg' % (i), output) 
    output = (X1_[i]).astype(np.uint8)
    cv2.imwrite('cityscapes(ori)_%d(dice).jpg' % (i), output)'''
new = inverse_trasform(X1_)


counter = 0
for i in new:
    plt.subplot(1, 1, 1)
    plt.axis('off')
    plt.imshow(i)
    plt.savefig('Ori' + '_%d.png' %(counter))
    counter = counter + 1
    plt.close()
    # plot generted target image
counter = 0
for i in X2_:
    plt.subplot(1, 1, 1)
    plt.axis('off')
    plt.imshow(i)
    plt.savefig('Tst' + '_%d.png' %(counter))
    counter = counter + 1
    plt.close()
counter = 0
for i in y_preds:
    plt.subplot(1, 1, 1)
    plt.axis('off')
    plt.imshow(i)
    plt.savefig('Pred' + '_%d.png' %(counter))
    counter = counter + 1
    plt.close()

