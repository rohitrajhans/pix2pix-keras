#!/usr/bin/env python
# coding: utf-8

from os import listdir
from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed
from numpy import load
import matplotlib.pyplot as plt
import h5py

def load_images(path, size=(256,512)):
    
    # path : Path for the training images directory
    # size : Size of each image. For our case one image contains two separate images of size 256*256.

    # load all images in a directory into memory and returns them as a combined array

    src_list, tar_list = list(), list()

    # enumerate all images in directory
    for filename in listdir(path):
        # load and resize the image
        pixels = load_img(path + filename, target_size=size)
        # convert to numpy array
        pixels = img_to_array(pixels)
        # split into source and target
        s_img, t_img = pixels[:, :256], pixels[:, 256:]
        src_list.append(s_img)
        tar_list.append(t_img)
        print(filename)
        
    print('Done')
    return [asarray(src_list), asarray(tar_list)]

def compress_images(src_images, tar_images, path_dest):
    
    # src_images : source image array
    # tar_images : target image array
    # path_dest : path to store the compressed .npz file
    
    # converts source and targets arrays into a single .npz file

    # [src_images, tar_images] = load_images(path_src)
    print("Loaded: ", src_images.shape, tar_images.shape)
    savez_compressed(path_dest, src_images, tar_images)
    print('Compressed and saved successfully!')

def plot_figures(src_images, tar_images, plot_dest, n_samples=3):

    # n_samples : number of samples to be plotted
    # plot_dest : path to save the plot

    # function to plot figures of dataset
    
    # [src_images, tar_images] = load_images(path_src)

    # Plots first n_samples number of images
    for i in range(n_samples):
        plt.subplot(2, n_samples, 1+i)
        plt.axis('off')
        plt.imshow(src_images[i].astype('uint8'))
    
    for i in range(n_samples):
        plt.subplot(2, n_samples, 1+n_samples+i)
        plt.axis('off')
        plt.imshow(tar_images[i].astype('uint8'))
        
    plt.savefig(plot_dest)

    print('Images saved successfully')

def convert_to_hdf5(src_images, tar_images, path_dest):
    
    # src_images : source image array
    # tar_images : target image array
    # path_dest : path to store the compressed .hdf5 file

    # function to compress arrays to hdf5 file format

    # [src_images, tar_images] = load_images(path_src)
    print('Loaded: ', src_images.shape, tar_images.shape)

    h5f = h5py.File(path_dest, 'w')
    h5f.create_dataset('dataset_1', data=[src_images, tar_images])
    h5f.close()

# USAGE
path_src = './assets/datasets/maps/train/'
path_dest = 'file.npz'
plot_dest = 'img.png'

[src_images, tar_images] = load_images(path_src)
compress_images(src_images, tar_images, path_dest)
plot_figures(src_images, tar_images, plot_dest)