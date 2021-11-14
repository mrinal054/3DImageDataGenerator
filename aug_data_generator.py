# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 11:03:30 2021

@author: mrinal

Reference:
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
"""
import numpy as np
import keras
import random
from scipy.ndimage.interpolation import shift, zoom
from scipy.ndimage import rotate

#%%
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, labels, batch_size=8, dim=(64,64,64), n_channels=1,
                 n_classes=2, shuffle=False, to_category=True, do_augmentation=False,
                 aug_list=None, flip_axis=None, shift_axis=None, shift_range=None,
                 zoom_axis=None, zoom_range=None, rotate_axis=None, rotate_angle=None):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.data = data 
        self.labels = labels
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.to_category = to_category
        self.do_augmentation = do_augmentation
        self.aug_list = aug_list
        self.flip_axis = flip_axis 
        self.shift_axis = shift_axis
        self.shift_range = shift_range
        self.zoom_axis = zoom_axis 
        self.zoom_range = zoom_range 
        self.rotate_axis = rotate_axis 
        self.rotate_angle = rotate_angle

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data) / self.batch_size)) # mkd changed

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        list_IDs_temp = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
 
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def rotation(self, data, label, rot_angle):
        # rot_angle -> size(1,3) e.g. (15,10,20)
        # Rotate around x-axis
        rot_data = rotate(data, axes=(1,2), angle=rot_angle[0], cval=0.0, reshape=False)
        rot_label = rotate(label, axes=(1,2), angle=rot_angle[0], cval=0.0, reshape=False)
    
        # Rotate around y-axis
        rot_data = rotate(rot_data, axes=(2,3), angle=rot_angle[1], cval=0.0, reshape=False)
        rot_label = rotate(rot_label, axes=(2,3), angle=rot_angle[1], cval=0.0, reshape=False)
        
        # Rotate around z-axis
        rot_data = rotate(rot_data, axes=(3,1), angle=rot_angle[2], cval=0.0, reshape=False)
        rot_label = rotate(rot_label, axes=(3,1), angle=rot_angle[2], cval=0.0, reshape=False)
        
        return rot_data, rot_label
    
    def flip(self, data, label, axis):
        flip_data = np.flip(data, axis)
        flip_label = np.flip(label, axis)
        
        return flip_data, flip_label

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=self.data.dtype)
        y = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=self.labels.dtype)

        # Generate data
        for i in range(len(list_IDs_temp)):
            # Store sample
            X[i,] = self.data[list_IDs_temp[i]]

            # Store class
            y[i,] = self.labels[list_IDs_temp[i]]
            
        # Do augmentation 
        if self.do_augmentation:
            for aug in self.aug_list:
                # Rotation
                if aug == 'rotate':
                    if self.rotate_angle is not None:
                        X, y = self.rotation(X, y, self.rotate_angle)
                # Flip
                elif aug == 'flip':
                    if self.flip_axis is not None:
                        if self.flip_axis == 'random':
                            self.flip_axis = random.randint(1, 3)
                        X, y = self.flip(X, y, self.flip_axis)
                       
        #
        if self.to_category:
            # Convert to categorical
            y = keras.utils.to_categorical(y, num_classes=self.n_classes) # y.dtype = 'float32'

        return X, y
    
