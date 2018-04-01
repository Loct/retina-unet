#==========================================================
#
#  This prepare the hdf5 datasets of the DRIVE database
#
#============================================================

import os
import h5py
import numpy as np
from PIL import Image



def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)


#------------Path of the images --------------------------------------------------------------
#train
original_imgs_train = r'./DRIVE/training/images/'
groundTruth_imgs_train = r'./DRIVE/training/1st_manual/'
borderMasks_imgs_train = r'./DRIVE/training/mask/'
#test
original_imgs_test = r'./DRIVE/test/images/'
groundTruth_imgs_test = r'./DRIVE/test/1st_manual/'
scndgroundTruth_imgs_test = r'./DRIVE/test/2nd_manual/'
borderMasks_imgs_test = r'./DRIVE/test/mask/'
#---------------------------------------------------------------------------------------------

Nimgs = 20
channels = 3
height = 584
width = 565
dataset_path = r'./DRIVE_datasets_training_testing/'

def get_train_datasets(imgs_dir,groundTruth_dir,borderMasks_dir):
    imgs = np.empty((Nimgs,height,width,channels))   
    groundTruth = np.empty((Nimgs,height,width)) 
    border_masks = np.empty((Nimgs,height,width))
    for path, subdirs, files in os.walk(imgs_dir): #list all files, directories in the path
        for i in range(len(files)):
            #original
            img = Image.open(imgs_dir+files[i])
            imgs[i] = np.asarray(img)
            groundTruth_name = files[i][0:2] + "_manual1.gif"
            print("ground truth name: " + groundTruth_name)
            g_truth = Image.open(groundTruth_dir + groundTruth_name)
            groundTruth[i] = np.asarray(g_truth)
            #corresponding border masks
           
            border_masks_name = files[i][0:2] + "_training_mask.gif"
            print("border masks name: " + border_masks_name)
            b_mask = Image.open(borderMasks_dir + border_masks_name)
            border_masks[i] = np.asarray(b_mask)
            
    print("imgs max: " +str(np.max(imgs)))
    print("imgs min: " +str(np.min(imgs)))
    print("ground truth:" +str(np.max(groundTruth)))
    assert(np.max(groundTruth)==255 and np.max(border_masks)==255)
    assert(np.min(groundTruth)==0 and np.min(border_masks)==0)
    print("ground truth and border masks are correctly withih pixel value range 0-255 (black-white)")
    #reshaping for my standard tensors
    imgs = np.transpose(imgs,(0,3,1,2))
    assert(imgs.shape == (Nimgs,channels,height,width))
    groundTruth = np.reshape(groundTruth,(Nimgs,1,height,width))
    border_masks = np.reshape(border_masks,(Nimgs,1,height,width))
    assert(groundTruth.shape == (Nimgs,1,height,width))
    assert(border_masks.shape == (Nimgs,1,height,width))
    return imgs, groundTruth, border_masks
                

def get_test_datasets(imgs_dir,groundTruth_dir,scndgroundTruth_dir,borderMasks_dir):
    imgs = np.empty((Nimgs,height,width,channels))
#    imgs = np.empty((Nimgs,height,width))
#    scndgroundTruth_dir = scndgroundTruth_imgs_test
    
    groundTruth = np.empty((Nimgs,height,width)) 
    scndgroundTruth = np.empty((Nimgs,height,width))
    border_masks = np.empty((Nimgs,height,width))
    for path, subdirs, files in os.walk(imgs_dir): #list all files, directories in the path
        for i in range(len(files)):
            #original
            print("original image: " +files[i])
            img = Image.open(imgs_dir+files[i])
            imgs[i] = np.asarray(img)
            groundTruth_name = files[i][0:2] + "_manual1.gif"
            print("ground truth name: " + groundTruth_name)
            g_truth = Image.open(groundTruth_dir + groundTruth_name)
            groundTruth[i] = np.asarray(g_truth)
            #corresponding border masks 
        
            border_masks_name = files[i][0:2] + "_test_mask.gif"
            print("border masks name: " + border_masks_name)
            b_mask = Image.open(borderMasks_dir + border_masks_name)
            border_masks[i] = np.asarray(b_mask)
            
            scndgroundTruth_name = files[i][0:2] + "_manual2.gif"
            print("2nd manual segmentation ground truth name: " + scndgroundTruth_name)
            scnd_g_truth = Image.open(scndgroundTruth_dir + scndgroundTruth_name)
            scndgroundTruth[i] = np.asarray(scnd_g_truth)
                
    print("imgs max: " +str(np.max(imgs)))
    print("imgs min: " +str(np.min(imgs)))
    print("secondground truth is within pixel value from 0 to 1 ")
    assert(np.max(scndgroundTruth)==1)
    assert(np.min(scndgroundTruth)==0)
    assert(np.max(groundTruth)==255 and np.max(border_masks)==255)
    assert(np.min(groundTruth)==0 and np.min(border_masks)==0)
    print("ground truth and border masks are correctly within pixel value range 0-255 (black-white)")
    #reshaping for my standard tensors
    imgs = np.transpose(imgs,(0,3,1,2))
    assert(imgs.shape == (Nimgs,channels,height,width))
    groundTruth = np.reshape(groundTruth,(Nimgs,1,height,width))
    scndgroundTruth = np.reshape(scndgroundTruth,(Nimgs,1,height,width))
    border_masks = np.reshape(border_masks,(Nimgs,1,height,width))
    assert(groundTruth.shape == (Nimgs,1,height,width))
    assert(scndgroundTruth.shape == (Nimgs,1,height,width))
    assert(border_masks.shape == (Nimgs,1,height,width))
    return imgs, groundTruth, scndgroundTruth, border_masks
            

    
#getting the training datasets
imgs_train, groundTruth_train, border_masks_train = get_train_datasets(original_imgs_train,groundTruth_imgs_train,borderMasks_imgs_train)
print("saving train datasets")
write_hdf5(imgs_train, dataset_path + "DRIVE_dataset_imgs_train.hdf5")
write_hdf5(groundTruth_train, dataset_path + "DRIVE_dataset_groundTruth_train.hdf5")
write_hdf5(border_masks_train,dataset_path + "DRIVE_dataset_borderMasks_train.hdf5")

#getting the testing datasets
imgs_test, groundTruth_test,scndgroundTruth_test, border_masks_test = get_test_datasets(original_imgs_test,groundTruth_imgs_test,scndgroundTruth_imgs_test,borderMasks_imgs_test)
print("saving test datasets")
write_hdf5(imgs_test,dataset_path + "DRIVE_dataset_imgs_test.hdf5")
write_hdf5(groundTruth_test, dataset_path + "DRIVE_dataset_groundTruth_test.hdf5")
write_hdf5(scndgroundTruth_test,dataset_path + "DRIVE_dataset_2ndgroundTruth_test.hdf5")
write_hdf5(border_masks_test,dataset_path + "DRIVE_dataset_borderMasks_test.hdf5")
