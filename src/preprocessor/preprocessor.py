import cv2
import os
from matplotlib import pyplot as plt

class Preprocessor:
    def __init__(self):
        None
    
    def show_first_and_last(self, images):
        if len(images) > 0:
            
            fig, axes = plt.subplots(1, 2)
            
            axes[0].imshow(cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB))
            axes[0].set_title('First')

            axes[1].imshow(cv2.cvtColor(images[-1], cv2.COLOR_BGR2RGB))
            axes[1].set_title('Last imagen')

            plt.show()
        else:
            print("No images were found in the specified folder.")
    
    def load_images(self, images_folder):
        images = []
        for file in os.listdir(images_folder):
            if file.endswith(".jpg") or file.endswith(".png"):
                image_path = os.path.join(images_folder, file)
                image = cv2.imread(image_path)
                if image is not None:
                    images.append(image)
        return images
    
    def convert_to_gray(self, images):
        gray_images = []
        for image in images: 
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            equ_image = cv2.equalizeHist(gray_image)
            gray_images.append(equ_image)

        return gray_images
    
    def convert_bgt_to_hsv(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    def convert_hsv_to_bgr(self, image):
        return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)