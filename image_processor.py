import cv2
import numpy as np
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
import requests

img = None
IMG_HEIGHT = 255
IMG_WIDTH = 255
    
def load_image( img_path):
    img = mpimg.imread(img_path)
    return img

def process_images(img_folder):
    """Processes image required for feeding into a neural network

    Args:
        img_folder (string): The folder containing the images to be processed.
    """
    image_data_array =[]
    class_name = []
    
    #Loop through the image folder, get, resize and normalize image
    for directory in os.listdir(img_folder):
        for file_path in os.listdir(os.path.join(img_folder, directory)):
            image_path = os.path.join(img_folder, directory, file_path) #Initialize the image path
            # Convert Image to grayscale
            image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
            #Resize the image
            if(image.any()):
                image = cv2.resize(image, (255, 255), interpolation = cv2.INTER_AREA)
                #Transform image into a tensor
                image = np.array(image)
                image = image.astype('float32')

                #normalize image
                image = image/255.0
                image_data_array.append(image)
                class_name.append(directory)
            else:
                next
    return image_data_array, class_name
    
def random_images( parent_dir, target_class, binarify=False):
    target_folder = parent_dir+target_class
    image = random.sample(os.listdir(target_folder))
    plt.figure(figsize=(10,7))
    image = plt.imread(image)
    plt.imshow(image,cmap=plt.cm.binary)
    
def improve_image( path, thresh, im_bw):
    """Improves blury or non clear image

    Args:
        path (string): The path to the image file
        thresh (Int): thershold
        im_bw (Int): 
    """
    image = cv2.imread(path)
    image = cv2.bitwise_not(image)
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_scale = cv2.bitwise_not(gray_scale)
    thresh,im_bw = cv2.threshold(gray_scale,thresh,im_bw, cv2.THRESH_BINARY)
    plt.imshow(im_bw)
    plt.show()

def generate_random_images( num:int, save_path:str):
    """Generates and downloads random images from https://picsum.photo/200/200/?random

    Args:
        num (int): Number of images you wish to download
        save_path (str): The path to which the images will be saved
    """
    url = "https://picsum.photo/200/200/?random"
    for i in range(num):
        response = requests.get(url)
        if response.status_code == 200:
            filename = f'img_{i}.jpg'
            file_path = os.path.join(save_path, filename)
            with open(file_path, 'wb') as f:
                print("saving: ", filename)
                f.write(response.content)
