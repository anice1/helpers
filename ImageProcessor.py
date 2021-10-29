import cv2
import numpy as np
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
import pdf2image as p2i

class ImageProcessor:
    def __init__(self):
        self.img = None
        self.IMG_HEIGHT = 255
        self.IMG_WIDTH = 255
    
    def load_image(self, img_path):
        self.img = mpimg.imread(img_path)
        return self.img
    
    def process_images(self, img_folder):
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
        
    def random_images(self, parent_dir, target_class, binarify=False):
        target_folder = parent_dir+target_class
        image = random.sample(os.listdir(target_folder))
        plt.figure(figsize=(10,7))
        image = plt.imread(image)
        plt.imshow(image,cmap=plt.cm.binary if binarify=True)
        
    def improve_image(path, thresh, im_bw):
        """Improves Image data by trying to improve blury or non clear image

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
                