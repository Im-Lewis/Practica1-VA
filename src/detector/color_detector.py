import numpy as np
from detector.detector import Detector
import cv2
from matplotlib import pyplot as plt

class ColorDetector(Detector):
    # H: 0 a 179 (elección de color)
    # S: 0 a 255 (de blanco al color)
    # V: 0 a 255 (de negro a menos negro)
    def __init__(self):
        super().__init__()
        
        self.lower_blue_limit = np.array([105, 150, 20])
        self.upper_blue_limit = np.array([130, 255, 255])
        
        self.mask = np.ones((40, 80), dtype=np.float32)

    def show_ideal_mask(self):
        plt.imshow(self.mask, cmap='gray')
        plt.title('Máscara Ideal')
        plt.show()
    
    def cut_zone(self, image, coord):
        #im = cv2.imread('./imagenesTest/00003.png')
        [y, x, h, w] = coord
        im = image[y:h, x:w]
        resized_image = cv2.resize(im, (80,40))
        return resized_image
    
    def cut_detected_zone(self, image, coords):
        detected_subimages = []
        for coord in coords:
            im = self.cut_zone(image, coord)
            detected_subimages.append(im)
        return detected_subimages

    # TODO: Get the resulted mask
    def apply_blue_filter(self, imagenes):
        for image in imagenes:
            plt.subplot(2,1,1)
            plt.imshow(image[:,:,::-1])
    
            img = self.preprocessor.convert_bgr_to_hsv(image)
            mask = cv2.inRange(img, self.lower_blue_limit, self.upper_blue_limit)
            plt.subplot(2,1,2)
            plt.imshow(self.preprocessor.convert_bg_to_rgb(mask)/255)
            plt.show()
