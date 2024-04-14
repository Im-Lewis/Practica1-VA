import os
import cv2
from matplotlib import pyplot as plt

class Loader:
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
    def show(self, image, title=""):
        image = self.convert_bgr_to_rgb(image)
        plt.title(title)
        plt.imshow(image)
