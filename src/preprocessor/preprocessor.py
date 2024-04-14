import cv2
import os
from matplotlib import pyplot as plt

class Preprocessor:
    def __init__(self):
        None
    
    def show_first_and_last(self, images):
        if len(images) > 0:
            
            plt.subplot(1, 2, 1)
            self.show(images[0], 'First')
            plt.subplot(1, 2, 2)
            self.show(images[-1], 'Last imagen')

            plt.show()
        else:
            print("No images were found in the specified folder.")
    
    def load_all_images(self, dir):
        images = []
        for file in os.listdir(dir):
            image = self.load_image(dir, file)
            images.append(image)
        return images
    
    def load_image(self, dir, image_name):
        if image_name.endswith(".jpg") or image_name.endswith(".png"):
            image_path = os.path.join(dir, image_name)
            image = cv2.imread(image_path)
            if image is not None:
                return image

    def convert_to_gray(self, images):
        gray_images = []
        for image in images: 
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            equ_image = cv2.equalizeHist(gray_image)
            gray_images.append(equ_image)

        return gray_images
    
    def convert_bgr_to_hsv(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    def convert_bgr_to_rgb(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def convert_hsv_to_bgr(self, image):
        return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    
    def convert_hsv_to_rgb(self, image):
        return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    
    #TODO: Create resizer
    def cut_and_resize_a_image(image, size = (40, 80)):
        None
    def show(self, image, title=""):
        image = self.convert_bgr_to_rgb(image)
        plt.title(title)
        plt.imshow(image)