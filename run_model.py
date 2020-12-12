# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 21:35:45 2019

@author: Gongfan Chen
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from os import walk
import shutil
import cv2     #resizing th image
from random import shuffle  #shuffle images
from tqdm import tqdm  #professional looping with progressbar
#import tensorflow as tf
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout, Conv2D, MaxPool2D
from tensorflow.keras.activations import relu, softmax, tanh, sigmoid
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
import glob

# Data Augmentation
def data_rotate():
    data_gen = ImageDataGenerator(rotation_range=10, # Amount of rotation
#                                  width_shift_range=0.5,# Amount of shift
#                                  height_shift_range=0.5,
                                  shear_range=0.5,# Shear angle in counter-clockwise direction as radians
#                                  zoom_range=0.1,# Range for random zoom
                                  horizontal_flip=True,# Boolean (True or False). Randomly flip inputs horizontally
                                  vertical_flip=True,
                                  fill_mode='reflect')# Points outside the boundaries of the input are filled
                                                      # according to the given mode
    return data_gen



# Import folder
def load_dataset(path):
    data_gen = data_rotate()
    f = []
    if os.path.exists('Augmentation') == True:
        shutil.rmtree('Augmentation') 
    print('Folder Emplied')
    
    os.mkdir('Augmentation')
    for (dir_path, dir_names, file_names) in walk(path): # Find all images in folder
        f.extend(file_names)
    for item in f: # For each image in folder
        image = Image.open(path+'/'+item).convert('L') # Create a numpy array with shape (1, 500, 500)
        x = img_to_array(image)
        x_label = item.split('.')[-2] #x = np.asarray(x) # Convert to a numpy array with shape (1, 1, 500, 500)
        x = x.reshape((1,) + x.shape)   
        i = 0
        for batch in data_gen.flow(x, save_to_dir='Augmentation', save_prefix=x_label, save_format='jpg'):
            i += 1
            if i > 9 : #determine number of additional images per origional image
                break             
    print('Data Augmentation finished.\n')

def data_reg(path):
    e = 1# 0 is regenerate, 0 is keep origional
    if e == 1:   
        load_dataset(path)
        
data_reg('Training') # run the data regenration
#end of data aug

# train_image
def image_iden():
    TRAIN_DIR = 'Training'
    TRAIN_AUG = 'Augmentation'#image=cv2.imread('Training/000016.jpg') #plt.imshow(image)  #plt.show()
    IMG_SIZE = 100#LR = 1e-3   #MODEL_NAME = 'Soybean'.format(LR, 'classify')   
    dataset = pd.read_csv('train_files.csv')
    file_names = list(dataset['file_name'].values)
    img_labels = list(dataset['annotation'].values)
    return  TRAIN_DIR, TRAIN_AUG, IMG_SIZE, dataset, file_names, img_labels

TRAIN_DIR, TRAIN_AUG, IMG_SIZE, dataset, file_names, img_labels = image_iden()

def label_img_orignal(img):
    word_label = img.split('.')[-2] #eg: 000006.jpg
    for i in range(len(img_labels)):
        if word_label == file_names[i].split('.')[-2]:
            return img_labels[i]
        
def label_img_aug(img):
    word_label=img.split('_')[-3]
    for i in range(len(img_labels)):
        if word_label == file_names[i].split('.')[-2]:
            return img_labels[i]
    
def create_train_data_orignal():
    train_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img_orignal(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        train_data.append([np.array(img), np.array(label)])    
    shuffle(train_data)
    np.save('train_data_orignal.npy', train_data)   #in future we can load this file
    return train_data                           
                           
def create_train_data_aug(): 
    train_data = []
    for img in tqdm(os.listdir(TRAIN_AUG)):
        label = label_img_aug(img)
        path = os.path.join(TRAIN_AUG,img)
        img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        train_data.append([np.array(img), np.array(label)]) 
    shuffle(train_data)
    np.save('train_data_aug.npy', train_data)
    return train_data

def create_test_data(): 
    test_images = []
    for f in glob.iglob('Project_C2_Testing/*'):
        i = cv2.resize(cv2.imread(f,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
        i = i.reshape(IMG_SIZE,IMG_SIZE,1)
        test_images.append(i)
    test_images = np.array(test_images)
    return test_images

def activation_fcn(x):
    if x == "relu":
        return relu
    elif x == "tanh":
        return tanh
    elif x == "softmax":
        return softmax
    elif x == "sigmoid":
        return sigmoid
                               
def model_structure(input, activation_fn, learning_rate,num_neurons_in_dense, num_classes, drop_out, kernel, filter_size):
    
    
# Fully-connect networks
    
#    model=Sequential()
#    model.add(Flatten(input_shape=input))
#    model.add(BatchNormalization())
#    model.add(Dense(units=num_neurons_in_dense, activation=activation_fn))
#    model.add(Dropout(0.2))
#    model.add(Flatten())
#    model.add(BatchNormalization())
#    model.add(Dense(units = num_classes, activation=softmax))
#    model.compile(loss = sparse_categorical_crossentropy, optimizer=Adam(lr=learning_rate), metrics=["accuracy"])
#    model.summary()
    
    
#CNN
    
    model=Sequential()
    model.add(Conv2D(
            filters=filter_size,
            kernel_size=[kernel,kernel],
            input_shape=input,
            padding="same",
            activation=activation_fn
            ))
    
    model.add(BatchNormalization())
    
    model.add(MaxPool2D(
            pool_size=[2,2],
            strides=2))
    
    model.add(Conv2D(
            filters=filter_size,
            kernel_size=[kernel,kernel],
            padding = "same",
            activation = activation_fn
            ))
    
    model.add(BatchNormalization())
    
    model.add(MaxPool2D(
            pool_size=[2,2],
            strides=2
            ))
    
    model.add(Flatten())
    
    model.add(Dense(
               units=num_neurons_in_dense,
               activation=activation_fn))
    model.add(Dropout(drop_out))
    
    model.add(Dense(units = num_classes,
                    activation=softmax))
    
    model.compile(
            loss = sparse_categorical_crossentropy,
            optimizer=Adam(lr=learning_rate),
            
            metrics=["accuracy"])
    
    model.summary()
    
    return model   

def train_model(model, train_images, train_labels, batch_size, num_epochs, valid_images, valid_labels):
#                ,save_callback, tb_callback
#                ):
    history = model.fit(
            x = train_images,
            y = train_labels,
            batch_size = batch_size,
            epochs= num_epochs,
            validation_data=(valid_images, valid_labels),
            shuffle = True)
#            callbacks=[save_callback, tb_callback])
            #verbose = 0
    history_dict=history.history
    train_accuracy=history_dict["acc"]
    train_loss = history_dict["loss"]
    valid_accuracy = history_dict["val_acc"]
    valid_loss = history_dict["val_loss"]
    return train_accuracy, train_loss, valid_accuracy, valid_loss

#def test_model(model, test_images, test_labels):
#    test_loss, test_accuracy = model.evaluate(
#            x = test_images,
#            y = test_labels,
#            verbose = 0
#            )
#    predictions = model.predict_proba(
#            x = test_images,
#            batch_size=None,
#            verbose=0)
#    return test_accuracy, test_loss, predictions
    
# end of train image

def data_split():
    train_data_orig = create_train_data_orignal()
    train_data_aug = create_train_data_aug()
    data=train_data_orig+train_data_aug
    random.shuffle(data)
    #IMG_SIZE = 100
    X=[]
    Y=[]
    for features, labels in data:
        X.append(features)
        Y.append(labels)
    X=np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1) 
    Y=np.array(Y)
    valid_set_size = int(np.floor(len(X)/4))
    split = len (X) - valid_set_size
    valid_images = X[split :]
    valid_labels = Y[split :]
    train_images = X[: split]
    train_labels = Y[: split]
    return train_images, train_labels, valid_images, valid_labels
    
def strucmodel(lr, acf, neurons_size, drop_out, kernel, filter_size):
    learning_rate = lr
    num_neurons_in_dense = neurons_size
    active_fn = acf
    activation_fn = activation_fcn(active_fn)
    num_channels = 1
    input_shape = [IMG_SIZE, IMG_SIZE, num_channels]
    num_classes = 5
#    folder = os.path.join(os.getcwd(), datetime.now().strftime("%d-%m_%Y_%M-%S"), str(active_fn))
#    history_file = folder + "\soybean" + str(active_fn) + ".h5"
#    save_callback = ModelCheckpoint(filepath = history_file, verbose = 1)
#    tb_callback = TensorBoard(log_dir = folder)
    model = model_structure(input_shape, activation_fn, learning_rate, num_neurons_in_dense, num_classes, drop_out, kernel, filter_size)
    #t0 = time.time()
    
    return model

    
def runmodel(train_images, train_labels, valid_images, valid_labels, model, epoch, batch, lr, acf):   
    test_images = create_test_data()
    num_epochs = epoch
    batch_size = batch
    active_fn = acf
    train_accuracy, train_loss, valid_accuracy, valid_loss = train_model(model, train_images, train_labels, batch_size, num_epochs, valid_images, valid_labels)#, save_callback, tb_callback)
    num_epochs_plot = range(1, len(train_accuracy) + 1)
    plt.figure(1)
    plt.plot(num_epochs_plot, train_loss, "b", label="Training Loss")
    plt.plot(num_epochs_plot, valid_loss, "r", label="Validation Loss")
    plt.title("Loss Curves_" + active_fn)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('Figures/' + active_fn + '_loss.png')
    plt.show()
    plt.figure(2)
    plt.plot(num_epochs_plot, train_accuracy, "b", label="Training Accuracy")
    plt.plot(num_epochs_plot, valid_accuracy, "r", label="Validation Accuracy")
    plt.title("Accuracy Curves_" + active_fn)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig('Figures/' + active_fn + '_acc.png')
    plt.show()
    prediction = model.predict_classes(test_images)
    result = np.zeros((test_images.shape[0],np.max(train_labels)+1))
    result[np.arange(test_images.shape[0]),prediction] = 1
    np.savetxt('labels21.csv', result , delimiter=',',fmt='%i')
    
    return train_accuracy, train_loss, valid_accuracy, valid_loss   
    
    
def tun():
    train_images, train_labels, valid_images, valid_labels = data_split()
    acf = 'relu'
    accuracy = ''
    for e in range(3):
        e = e
        epoch = 5+5*e
        for q in range(4):
            q = q
            lr = 0.0001*np.power(10,q)
            for n in range(4):
                n = n
                batch = 128*np.power(2,n)
                model = strucmodel(lr, acf, neurons_size, drop_out, kernel, filter_size)
                valid_accuracy, valid_loss = runmodel(train_images, train_labels, valid_images, valid_labels, model, epoch, batch, lr, acf)
            r = 'The loss is : %s, The accuracy is: %s (Batch: %s, Learning_rate: %s)'%(valid_loss[-1], valid_accuracy[-1], batch, lr)
            print(r)
            accuracy = accuracy+','+r
    accuracy = accuracy
    return accuracy

#one time fit
  
#train_images, train_labels, valid_images, valid_labels = data_split()
#lr = 0.001
#acf = 'relu'
#model = strucmodel(lr, acf)
#epoch = 20
#batch = 32
#runmodel(train_images, train_labels, valid_images, valid_labels, model, epoch, batch, lr, acf)


#hyperparameter tuning
kernel=3
filter_size=32
neurons_size=256
drop_out=0.5
acf = 'relu'
lr =0.00146
epoch = 25
batch_size =256
#accuracy = tun()
train_images, train_labels, valid_images, valid_labels = data_split()

#param_val = batch_size

#parm = "batch_size"

#val_train_loss = np.zeros(len(param_val))
#val_train_acc = np.zeros(len(param_val))
#val_valid_loss = np.zeros(len(param_val))
#val_valid_acc=np.zeros(len(param_val))
#train_time = np.zeros(len(param_val))

#for val, param in enumerate(param_val):
model = strucmodel(lr, acf, neurons_size, drop_out, kernel, filter_size)
    #t0 = time.time()
train_accuracy, train_loss, valid_accuracy, valid_loss = runmodel(train_images, train_labels, valid_images, valid_labels, model, epoch, batch_size, lr, acf)
    
    
    
#    t1 = time.time()
 #   train_time[val] = t1-t0
  #  val_train_loss[val] = train_loss[-1]
  #  val_train_acc[val] = train_accuracy[-1]
  #  val_valid_loss[val] = valid_loss[-1]
  #  val_valid_acc[val] = valid_accuracy[-1]

    
########## plot the output########################
#plt.figure(1)
#plt.plot(param_val, val_train_loss, "b", label="Training Loss")
#plt.plot(param_val, val_valid_loss, "r", label="Validation Loss")
#plt.title("Loss Curves " + parm)
#plt.xlabel("Parameter Values")
#plt.ylabel("Loss")
#plt.legend()
#plt.savefig('Figures/' + parm + '_loss.png')
#plt.show()

# Accuracy curves
#plt.figure(2)
#plt.plot(param_val, val_train_acc, "b", label="Training Accuracy")
#plt.plot(param_val, val_valid_acc, "r", label="Validation Accuracy")
#plt.title("Accuracy Curves " + parm)
#plt.xlabel("Parameter Values")
#plt.ylabel("Accuracy")
#plt.legend()
#plt.savefig('Figures/' + parm + '_acc.png')
#plt.show()


    
    
     
    
    
    
    
    
    
    
    
    
    
    
    
